# 🎙️ Transcribe Video with Whisper + Rich

A simple and powerful Python tool to **transcribe audio** from a **local MP4 video** or a **YouTube URL**, using OpenAI's Whisper model. Styled terminal output is provided using the `rich` library.

---

## 📦 Features

- 🔗 Supports **YouTube URLs** and **local MP4 files**
- 🎧 Automatically extracts audio
- 🧠 Transcribes speech using [Whisper](https://github.com/openai/whisper)
- 📝 Saves output as a `.txt` file
- 🎨 Beautiful CLI with [`rich`](https://github.com/Textualize/rich)
- ⚙️ Model selection: `tiny`, `base`, `small`, `medium`, `large`

---

## 🚀 Installation

1. Clone this repo or copy `transcribe_video.py`
2. Install required packages:

```bash
python -m venv venv
source venv/bin/activate
pip install pytube moviepy openai-whisper torch rich
```

## 🛠️ Usage

### 🔤 Transcribe a local MP4 file

```bash
python transcribe_video.py /path/to/video.mp4
```

### 📺 Transcribe from YouTube

```bash
python transcribe_video.py "https://www.youtube.com/watch?v=EXAMPLE"
```

### ⚙️ Use a specific Whisper model

```bash
python transcribe_video.py ./video.mp4 --model small
```

Available models: tiny, base (default), small, medium, large

#### 🧠 Which model should I choose?

| Model   |   Size   |    Speed    | Accuracy  |              Use Case              |
| ------- | -------- | ----------- | --------- | ---------------------------------- |
| tiny    |  ~39 MB  | Ultra fast  |    Low    | Quick tests, short audio           |
| base    |  ~74 MB  |    Fast     | Moderate  | General use                        |
| small   | ~244 MB  |   Moderate  |   Good    | Balanced speed and quality         |
| medium  | ~769 MB  |    Slower   | Very Good | For high-quality transcription     |
| large   | ~1550 MB | Slow (CPU)  | Excellent | Best results, recommended with GPU |

## 📂 Output

- Transcript is shown in the terminal using a styled panel.
- A `.txt` file is saved:
    - For MP4 files: `yourvideo_transcription.txt`
    - For YouTube: `transcription.txt` in the current folder

## 🧩 Example

```bash
python transcribe_video.py "./interview.mp4" --model medium

python transcribe_video.py video1.mp4 video2.mp4

python transcribe_video.py "https://youtu.be/abc" "https://www.youtube.com/watch?v=xyz"

python transcribe_video.py video.mp4 "https://youtu.be/xyz" --model small

python transcribe_video.py dossier_videos/ autre_video.mp4 https://youtube.com/...
```
