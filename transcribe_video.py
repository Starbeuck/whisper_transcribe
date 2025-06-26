import os
import re
import whisper
import tempfile
import argparse
import torch
from pathlib import Path
from typing import List, Literal, Optional
from pytubefix import YouTube
from moviepy import VideoFileClip
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel

console = Console()

ModelSize = Literal["tiny", "base", "small", "medium", "large"]

def is_gpu_available() -> bool:
    return torch.cuda.is_available()

def sanitize_filename(title: str) -> str:
    return re.sub(r'[^a-zA-Z0-9]+', '_', title.strip())

def download_youtube_video(url: str, output_path: Path) -> Optional[Path]:
    console.print(f"[bold cyan]🔽 Downloading YouTube video:[/bold cyan] {url}")
    yt = YouTube(url)
    title_slug = sanitize_filename(yt.title)
    filename = f"{title_slug}.mp4"
    stream = yt.streams.filter(only_audio=False, file_extension="mp4").first()
    out_file = stream.download(output_path=str(output_path), filename=filename)
    console.print(f"[green]✅ Downloaded:[/green] {out_file}")
    return Path(out_file)

def extract_audio(video_path: Path, audio_path: Path) -> bool:
    console.print("[bold cyan]🎧 Extracting audio...[/bold cyan]")
    try:
        clip = VideoFileClip(str(video_path))
        clip.audio.write_audiofile(str(audio_path), codec='aac')
        if not audio_path.exists() or audio_path.stat().st_size == 0:
            console.print("[red]❌ Extracted audio is empty.[/red]")
            return False
        console.print(f"[green]✅ Audio saved:[/green] {audio_path}")
        return True
    except Exception as e:
        console.print(f"[red]❌ Failed to extract audio: {e}[/red]")
        return False

def transcribe_audio(audio_path: Path, model_size: ModelSize) -> str:
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
        result = model.transcribe(str(audio_path))
        progress.update(task, completed=1)

    console.print("[green]✅ Transcription complete.[/green]")
    return result["text"]

def save_transcription(text: str, output_path: Path) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)
    console.print(f"[bold green]💾 Transcription saved to:[/bold green] {output_path}")

def process_video(source: str, model_size: ModelSize, index: int) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        if source.startswith("http://") or source.startswith("https://"):
            try:
                video_path = download_youtube_video(source, tmp_path)
                if not video_path or not video_path.exists():
                    console.print(f"[red]❌ Download failed for {source}[/red]")
                    return
                yt = YouTube(source)
                title_slug = sanitize_filename(yt.title)
                output_txt = Path.cwd() / f"{index}_{title_slug}_transcription.txt"
            except Exception as e:
                console.print(f"[red]❌ YouTube error: {e}[/red]")
                return
        elif Path(source).is_file():
            video_path = Path(source)
            if not video_path.exists() or video_path.stat().st_size == 0:
                console.print(f"[red]❌ Invalid or empty file: {source}[/red]")
                return
            base_name = sanitize_filename(video_path.stem)
            output_txt = video_path.parent / f"{index}_{base_name}_transcription.txt"
        else:
            console.print(f"[bold red]❌ Invalid source:[/bold red] {source}")
            return

        audio_path = tmp_path / "audio.m4a"
        if not extract_audio(video_path, audio_path):
            return

        transcription = transcribe_audio(audio_path, model_size)
        save_transcription(transcription, output_txt)
        console.print(Panel.fit(transcription, title=f"📄 {index}_{video_path.stem}", border_style="cyan"))

def main():
    import concurrent.futures

    parser = argparse.ArgumentParser(description="Transcribe videos using Whisper.")
    parser.add_argument("source", help="Text file with URLs or path to one MP4/YouTube link")
    parser.add_argument("--model", default="base", choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper model to use")
    parser.add_argument("--max-workers", type=int, default=2, help="Maximum number of parallel workers (CPU only)")
    args = parser.parse_args()

    source_path = Path(args.source)
    sources: List[str] = []

    if source_path.is_file() and source_path.suffix == ".txt":
        sources = [line.strip() for line in source_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        sources = [args.source]

    if is_gpu_available():
        for i, src in enumerate(sources, 1):
            console.rule(f"[bold yellow]Processing: {src}")
            process_video(src, args.model, i)
    else:
        console.print(f"[bold cyan]🧵 No GPU detected: using up to {args.max_workers} workers[/bold cyan]")
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {
                executor.submit(process_video, src, args.model, i): src
                for i, src in enumerate(sources, 1)
            }
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    console.print(f"[red]❌ Error during processing: {e}[/red]")

if __name__ == "__main__":
    main()