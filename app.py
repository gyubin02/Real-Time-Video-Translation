import threading
import queue
import soundcard as sc
import whisper
import tkinter as tk

SAMPLERATE = 16000
CHUNK_SECONDS = 4

class SubtitleOverlay:
    """Simple desktop overlay that transcribes system audio."""

    def __init__(self, model_size: str = "small") -> None:
        self.model = whisper.load_model(model_size)
        self.text_queue: "queue.Queue[str]" = queue.Queue()
        self.running = True

    def _audio_loop(self) -> None:
        """Capture system audio and put transcripts into the queue."""
        speaker = sc.default_speaker()
        # Create a pseudo microphone that records what the speaker plays
        mic = sc.get_microphone(speaker.name, include_loopback=True)
        with mic.recorder(samplerate=SAMPLERATE) as rec:
            while self.running:
                data = rec.record(numframes=SAMPLERATE * CHUNK_SECONDS)
                # whisper expects float32 numpy array
                result = self.model.transcribe(data, fp16=False)
                text = result.get("text", "").strip()
                if text:
                    self.text_queue.put(text)

    def _gui_loop(self) -> None:
        """Run the overlay window and update text from the queue."""
        root = tk.Tk()
        root.overrideredirect(True)  # Remove window chrome
        root.attributes("-topmost", True)
        root.configure(bg="black")
        root.geometry("800x120+100+100")
        label = tk.Label(root, text="Listening...", fg="white", bg="black", font=("Helvetica", 24))
        label.pack(expand=True, fill="both")

        def update_label() -> None:
            if not self.text_queue.empty():
                label.configure(text=self.text_queue.get())
            root.after(500, update_label)

        def on_close() -> None:
            self.running = False
            root.destroy()

        root.protocol("WM_DELETE_WINDOW", on_close)
        update_label()
        root.mainloop()

    def run(self) -> None:
        threading.Thread(target=self._audio_loop, daemon=True).start()
        self._gui_loop()


if __name__ == "__main__":
    SubtitleOverlay().run()
