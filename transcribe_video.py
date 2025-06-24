import os
import whisper
import tempfile
import argparse
import torch
from pytube import YouTube
from moviepy import VideoFileClip
from concurrent.futures import ProcessPoolExecutor, as_completed
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel

console = Console()

def is_gpu_available():
    return torch.cuda.is_available()

def download_youtube_video(url, output_path):
    console.print(f"[bold cyan]🔽 Downloading YouTube video:[/bold cyan] {url}")
    yt = YouTube(url)
    video = yt.streams.filter(only_audio=False, file_extension="mp4").first()
    out_file = video.download(output_path=output_path)
    console.print(f"[green]✅ Downloaded:[/green] {out_file}")
    return out_file

def extract_audio(video_path, audio_path):
    console.print("[bold cyan]🎧 Extracting audio...[/bold cyan]")
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(audio_path, codec='aac')
    console.print(f"[green]✅ Audio saved:[/green] {audio_path}")

def transcribe_audio(audio_path, model_size):
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        console.print(f"[bold green]🚀 GPU detected:[/bold green] {gpu_name}")
    else:
        console.print("[bold yellow]⚠️ No GPU detected. Using CPU.[/bold yellow]")

    console.print(f"[bold cyan]🧠 Transcribing with Whisper ({model_size})...[/bold cyan]")
    model = whisper.load_model(model_size)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        task = progress.add_task("Transcribing...", total=None)
        result = model.transcribe(audio_path)
        progress.update(task, completed=1)

    console.print("[green]✅ Transcription complete.[/green]")
    return result["text"]

def save_transcription(text, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)
    console.print(f"[bold green]💾 Transcription saved to:[/bold green] {output_path}")

def process_video(source, model_size):
    with tempfile.TemporaryDirectory() as tmpdir:
        if source.startswith("http://") or source.startswith("https://"):
            video_path = download_youtube_video(source, tmpdir)
            filename = "transcription_" + YouTube(source).video_id + ".txt"
            output_txt = os.path.join(os.getcwd(), filename)
        elif os.path.isfile(source):
            video_path = source
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            output_txt = os.path.join(os.path.dirname(video_path), f"{base_name}_transcription.txt")
        else:
            console.print(f"[bold red]❌ Invalid source:[/bold red] {source}")
            return

        audio_path = os.path.join(tmpdir, "audio.m4a")
        extract_audio(video_path, audio_path)
        transcription = transcribe_audio(audio_path, model_size)
        save_transcription(transcription, output_txt)
        console.print(Panel.fit(transcription, title=f"📄 Transcription of {source}", border_style="cyan"))

def process_video_wrapper(args):
    source, model_size = args
    try:
        process_video(source, model_size)
    except Exception as e:
        return f"❌ Error processing {source}: {e}"
    return f"✅ Done: {source}"

def main():
    parser = argparse.ArgumentParser(description="Transcribe one or more videos using Whisper.")
    parser.add_argument("sources", nargs="+", help="List of YouTube URLs, MP4 file paths, or folders containing MP4s")
    parser.add_argument("--model", default="base", choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper model to use (default: base)")

    args = parser.parse_args()
    all_sources = []

    for source in args.sources:
        if os.path.isdir(source):
            mp4_files = [os.path.join(source, f) for f in os.listdir(source) if f.lower().endswith(".mp4")]
            if not mp4_files:
                console.print(f"[bold red]⚠️ No MP4 files found in directory:[/bold red] {source}")
            else:
                all_sources.extend(mp4_files)
        else:
            all_sources.append(source)

    use_gpu = is_gpu_available()

    if use_gpu:
        console.print("[bold green]🚀 GPU detected: disabling parallel processing.[/bold green]")
        for src in all_sources:
            console.rule(f"[bold yellow]Processing: {src}")
            process_video(src, args.model)
    else:
        console.print("[bold cyan]🧵 No GPU detected: enabling parallel CPU processing.[/bold cyan]")
        with ProcessPoolExecutor() as executor:
            tasks = [(src, args.model) for src in all_sources]
            futures = [executor.submit(process_video_wrapper, task) for task in tasks]

            for future in as_completed(futures):
                result = future.result()
                console.print(result)

if __name__ == "__main__":
    main()
