# Real-Time-Video-Translation

This repository contains a simple desktop application that listens to your system audio and displays live subtitles in a small overlay window.

## Requirements

- Python 3.9+
- [soundcard](https://pypi.org/project/soundcard/) for capturing system audio
- [openai-whisper](https://pypi.org/project/openai-whisper/) for speech recognition
- Tkinter (usually bundled with Python) for the overlay GUI

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

### Platform notes

- **Windows**: No additional setup is required; the app uses WASAPI loopback to record output audio.
- **macOS**: Install a virtual audio device such as [BlackHole](https://github.com/ExistentialAudio/BlackHole) and set it as the system output.
- **Linux**: Use PulseAudio/pipewire's monitor device or equivalent to make the speaker output recordable.

## Usage

Run the application:

```bash
python app.py
```

A borderless window will appear on top of your screen showing the most recent transcript of what is playing on your computer.

## Limitations

This example uses small audio chunks and a Whisper model, so transcription latency and accuracy depend on the chosen model and the performance of your machine.
