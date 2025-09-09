"""Minimal real-time subtitle overlay with GPT translation."""

from __future__ import annotations

import os
import queue
import threading

import numpy as np
import soundcard as sc
import tkinter as tk
import webrtcvad
from faster_whisper import WhisperModel
from openai import OpenAI

SAMPLERATE = 16000
FRAME_MS = 30
FRAME_SIZE = int(SAMPLERATE * FRAME_MS / 1000)
MAX_CHUNK_FRAMES = int(10_000 / FRAME_MS)

SYSTEM_PROMPT = (
    "You are a real-time subtitle translator. Translate English ASR fragments into "
    "NATURAL Korean. Preserve numbers and proper nouns; keep lines concise (<=42 chars, "
    "<=2 lines); style={style}."
)


def translate_text(client: OpenAI, text: str, style: str) -> str:
    """Translate English text to Korean using the GPT API."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(style=style)},
        {"role": "user", "content": text},
    ]
    response = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
    return response.choices[0].message.content.strip()


class SubtitleOverlay:
    """Desktop overlay that captures audio and shows translated subtitles."""

    def __init__(self, model_size: str = "small", style: str = "honorific") -> None:
        self.model = WhisperModel(model_size, compute_type="int8_float32")
        self.vad = webrtcvad.Vad(2)
        self.audio_queue: "queue.Queue[bytes]" = queue.Queue()
        self.text_queue: "queue.Queue[str]" = queue.Queue()
        self.running = True
        api_key = os.environ.get("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key) if api_key else OpenAI()
        self.style = style

    def _capture_loop(self) -> None:
        """Capture system audio, segment with VAD and enqueue chunks."""
        speaker = sc.default_speaker()
        mic = sc.get_microphone(speaker.name, include_loopback=True)
        with mic.recorder(samplerate=SAMPLERATE) as rec:
            voiced_frames: list[bytes] = []
            while self.running:
                data = rec.record(numframes=FRAME_SIZE)
                mono = np.mean(data, axis=1)
                pcm16 = (mono * 32768).astype(np.int16).tobytes()
                if self.vad.is_speech(pcm16, SAMPLERATE):
                    voiced_frames.append(pcm16)
                    if len(voiced_frames) >= MAX_CHUNK_FRAMES:
                        self.audio_queue.put(b"".join(voiced_frames))
                        voiced_frames = []
                elif voiced_frames:
                    self.audio_queue.put(b"".join(voiced_frames))
                    voiced_frames = []

    def _transcribe_loop(self) -> None:
        """Run ASR and translation on captured chunks."""
        while self.running:
            chunk = self.audio_queue.get()
            data = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
            segments, _ = self.model.transcribe(data)
            text = "".join(seg.text for seg in segments).strip()
            if text:
                try:
                    translation = translate_text(self.client, text, self.style)
                    self.text_queue.put(translation)
                except Exception as exc:  # pragma: no cover - best effort
                    self.text_queue.put(f"[Translation error: {exc}]")

    def _gui_loop(self) -> None:
        """Run the overlay window and update text from the queue."""
        root = tk.Tk()
        root.overrideredirect(True)
        root.attributes("-topmost", True)
        root.configure(bg="black")
        root.geometry("800x120+100+100")
        label = tk.Label(
            root,
            text="Listening...",
            fg="white",
            bg="black",
            font=("Helvetica", 24),
        )
        label.pack(expand=True, fill="both")

        def update_label() -> None:
            if not self.text_queue.empty():
                label.configure(text=self.text_queue.get())
            root.after(100, update_label)

        def on_close() -> None:
            self.running = False
            root.destroy()

        root.protocol("WM_DELETE_WINDOW", on_close)
        update_label()
        root.mainloop()

    def run(self) -> None:
        threading.Thread(target=self._capture_loop, daemon=True).start()
        threading.Thread(target=self._transcribe_loop, daemon=True).start()
        self._gui_loop()


if __name__ == "__main__":
    SubtitleOverlay().run()
