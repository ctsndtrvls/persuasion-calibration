from pathlib import Path
from typing import Iterable

from huggingface_hub import hf_hub_download


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


if __name__ == "__main__":
    download_conflictqa()

