import os
import tempfile
import argparse
import torch
import whisper
import requests
from typing import cast, List, Optional, Tuple, Literal
from urllib.parse import urlparse
from pathlib import Path
from pytubefix import YouTube
from moviepy import VideoFileClip
from concurrent.futures import ProcessPoolExecutor, as_completed
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel

console = Console()

ModelSize = Literal["tiny", "base", "small", "medium", "large"]


def is_gpu_available() -> bool:
    return torch.cuda.is_available()


def download_youtube_video(url: str, output_path: Path) -> Path:
    console.print(f"[bold cyan]🔽 Downloading YouTube video:[/bold cyan] {url}")
    yt = YouTube(url)
    video = yt.streams.filter(only_audio=False, file_extension="mp4").first()
    out_file = video.download(output_path=str(output_path))
    console.print(f"[green]✅ Downloaded:[/green] {out_file}")
    return Path(out_file)


def download_mp4_direct(url: str, output_path: Path) -> Optional[Path]:
    filename = os.path.basename(urlparse(url).path)
    local_path = output_path / filename
    console.print(f"[cyan]⬇️ Downloading MP4:[/cyan] {url}")
    try:
        with requests.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            with open(local_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        console.print(f"[green]✅ MP4 downloaded:[/green] {local_path}")
        return local_path
    except Exception as e:
        console.print(f"[red]❌ Failed to download {url}: {e}[/red]")
        return None


def extract_audio(video_path: Path, audio_path: Path) -> None:
    console.print("[bold cyan]🎧 Extracting audio...[/bold cyan]")
    clip = VideoFileClip(str(video_path))
    clip.audio.write_audiofile(str(audio_path), codec='aac')
    console.print(f"[green]✅ Audio saved:[/green] {audio_path}")


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
    return cast(str, result["text"])


def save_transcription(text: str, output_path: Path) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)
    console.print(f"[bold green]💾 Transcription saved to:[/bold green] {output_path}")


def process_video(source: str, model_size: ModelSize) -> None:
    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        video_path: Optional[Path] = None
        if source.startswith("http://") or source.startswith("https://"):
            if "youtube.com" in source or "youtu.be" in source:
                video_path = download_youtube_video(source, tmpdir)
                filename = f"transcription_{YouTube(source).video_id}.txt"
                output_txt = Path.cwd() / filename
            else:
                video_path= download_mp4_direct(source, tmpdir)
                if video_path is None:
                    return
                output_txt = Path.cwd() / f"{video_path.stem}_transcription.txt"
        else:
            file_path = Path(source)
            if file_path.is_file():
                video_path = file_path
                output_txt = file_path.with_name(f"{file_path.stem}_transcription.txt")
            else:
                console.print(f"[bold red]❌ Invalid source:[/bold red] {source}")
                return

        audio_path = tmpdir / "audio.m4a"
        extract_audio(video_path, audio_path)
        transcription = transcribe_audio(audio_path, model_size)
        save_transcription(transcription, output_txt)
        console.print(Panel.fit(transcription, title=f"📄 Transcription of {source}", border_style="cyan"))


def process_video_wrapper(args: Tuple[str, ModelSize]) -> str:
    source, model_size = args
    try:
        process_video(source, model_size)
    except Exception as e:
        return f"❌ Error processing {source}: {e}"
    return f"✅ Done: {source}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcribe one or more videos using Whisper.")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=os.cpu_count(),
        help="Maximum number of parallel processes (default: number of CPUs)"
    )
    parser.add_argument("sources", nargs="+", help="YouTube URLs, MP4 file paths, folders, or .txt file containing URLs")
    parser.add_argument("--model", default="base", choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper model to use (default: base)")
    args = parser.parse_args()

    model_size: ModelSize = args.model  # validated by argparse
    all_sources: List[str] = []

    for source in args.sources:
        path = Path(source)
        if path.is_dir():
            mp4_files = [str(p) for p in path.iterdir() if p.suffix.lower() == ".mp4"]
            if not mp4_files:
                console.print(f"[bold red]⚠️ No MP4 files found in directory:[/bold red] {source}")
            else:
                all_sources.extend(mp4_files)
        elif path.is_file() and path.suffix.lower() == ".txt":
            console.print(f"[bold cyan]📖 Reading sources from file:[/bold cyan] {source}")
            with path.open("r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
                all_sources.extend(lines)
        else:
            all_sources.append(source)

    if not all_sources:
        console.print("[bold red]❌ No valid video sources provided.[/bold red]")
        return

    console.print(f"[bold blue]📦 {len(all_sources)} video(s) to process[/bold blue]")

    if is_gpu_available():
        console.print("[bold green]🚀 GPU detected: using sequential mode.[/bold green]")
        for src in all_sources:
            console.rule(f"[bold yellow]Processing: {src}")
            process_video(src, model_size)
    else:
        console.print("[bold cyan]🧵 No GPU detected: using parallel processing.[/bold cyan]")
        with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            tasks: List[Tuple[str, ModelSize]] = [(src, model_size) for src in all_sources]
            futures = [executor.submit(process_video_wrapper, task) for task in tasks]
            for future in as_completed(futures):
                console.print(future.result())


if __name__ == "__main__":
    main()