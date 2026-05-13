from pathlib import Path
from typing import Iterable
import argparse

from huggingface_hub import hf_hub_download
from zipfile import ZipFile
import requests


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"


def _download_files_from_hf(repo_id: str, filenames: Iterable[str], target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for filename in filenames:
        print(f"Downloading {filename} from {repo_id} ...")
        local_path = hf_hub_download(repo_id=repo_id, filename=filename)
        dst = target_dir / filename
        Path(local_path).replace(dst)
        print(f"Saved to {dst}")


def download_conflictqa() -> None:
    """
    Download selected JSON files from the osunlp/ConflictQA repository
    directly via huggingface_hub (without using datasets.load_dataset,
    which no longer supports dataset scripts).

    The full file list is visible at:
    https://huggingface.co/datasets/osunlp/ConflictQA/tree/main
    """
    out_dir = DATA_DIR / "conflictqa"
    repo_id = "osunlp/ConflictQA"

    # You can extend this list with other JSON files from the repo
    files_to_download = [
        "conflictQA-popQA-gpt4.json",
        "conflictQA-strategyQA-gpt4.json",
    ]

    _download_files_from_hf(repo_id, files_to_download, out_dir)

    print("Done. ConflictQA JSON files are stored in 'data/conflictqa/'.")


def download_debateqa() -> None:
    """
    Download DebateQA from GitHub as a zip archive and extract
    the dataset folder into data/debateqa/.
    """
    out_dir = DATA_DIR / "debateqa"
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / "debateqa_main.zip"

    archive_url = "https://github.com/pillowsofwind/DebateQA/archive/refs/heads/main.zip"
    print(f"Downloading DebateQA archive from {archive_url} ...")
    with requests.get(archive_url, timeout=120, stream=True) as resp:
        resp.raise_for_status()
        with zip_path.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    print(f"Saved archive to {zip_path}")

    with ZipFile(zip_path, "r") as zf:
        for member in zf.namelist():
            if not member.startswith("DebateQA-main/dataset/"):
                continue
            relative = member.replace("DebateQA-main/dataset/", "", 1)
            if not relative:
                continue
            target_path = out_dir / "dataset" / relative
            if member.endswith("/"):
                target_path.mkdir(parents=True, exist_ok=True)
                continue
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, target_path.open("wb") as dst:
                dst.write(src.read())

    if zip_path.exists():
        zip_path.unlink()

    print("Done. DebateQA files are stored in 'data/debateqa/dataset/'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download local datasets.")
    parser.add_argument(
        "--dataset",
        choices=("all", "conflictqa", "debateqa"),
        default="all",
        help="Which dataset to download (default: all).",
    )
    args = parser.parse_args()

    if args.dataset in ("all", "conflictqa"):
        download_conflictqa()
    if args.dataset in ("all", "debateqa"):
        download_debateqa()

