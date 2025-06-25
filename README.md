# 🎙️ Transcribe Videos with Whisper

This script automatically transcribes local or online videos (YouTube, direct `.mp4` links) using [Whisper](https://github.com/openai/whisper).

---

## ✅ Features

- Automatically downloads videos from YouTube or direct `.mp4` URLs
- Extracts audio using MoviePy
- Transcribes using Whisper (GPU supported if available)
- Parallel processing when running on CPU
- Supports `.txt` files with lists of video sources
- Limit parallelism with `--max-workers`

---

## 📦 Installation (Python version)

### 1. Clone the project

```bash
git clone https://github.com/Starbeuck/whisper_transcribe
cd whisper_transcribe
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate       # macOS/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```
---

## ▶️ Usage

```bash
python transcribe_video.py <video.mp4|url|folder|list.txt> [--model model]
```

Examples:

```bash
python transcribe_video.py my_video.mp4
python transcribe_video.py https://www.youtube.com/watch?v=abc123 --model small
python transcribe_video.py path_to_videos/
python transcribe_video.py urls.txt
python transcribe_video.py urls.txt --max-workers 2
```

Available models:
- `tiny`, `base`, `small`, `medium`, `large`

`--max-workers`: number of parallel CPU workers to use (ignored if GPU is available)

---

## 🖥️ Windows Use (No Python)

### 1. Generate the `.exe` file

To create a standalone executable:

```bash
pip install pyinstaller
pyinstaller --onefile transcribe_video.py
```

The `dist/transcribe_video.exe` file is self-contained and distributable.

### 2. Run it

```bash
transcribe_video.exe video.mp4 --model base
```

> ⚠️ You must have `ffmpeg` installed for MoviePy to work correctly on Windows.

---

## 🧪 Type Checking with `mypy`

### 1. Install `mypy`

```bash
pip install mypy
```

### 2. Create a `mypy.ini` at the project root:

```ini
[mypy]
python_version = 3.10
ignore_missing_imports = True
strict = True
disallow_untyped_defs = True
disallow_incomplete_defs = True
warn_unused_ignores = True
warn_return_any = True
check_untyped_defs = True
```

### 3. Run the check

```bash
mypy transcribe_video.py
```

---

## 📁 Sample `.txt` file format

```txt
https://www.youtube.com/watch?v=abc123
https://www.site.com/video1.mp4
C:\Users\User\Videos\local_video.mp4
```

---

## 🧠 Notes

- Whisper models require significant memory (avoid `large` without GPU)
- Transcriptions are saved in the same folder as the video

---

## 🧑‍💻 Authors

Made with ❤️ to simplify video transcription.