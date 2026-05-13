import json
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"


def preview_fever2_adversarial(n: int = 5) -> None:
    """
    Print a few examples from the FEVER 2.0 adversarial dataset
    stored as JSONL under data/fever2_adversarial/.
    """
    # You may need to adjust the filename depending on what you download
    # from https://fever.ai/dataset/adversarial.html
    candidate_files = list((DATA_DIR / "fever2_adversarial").glob("*.jsonl"))
    if not candidate_files:
        print("No FEVER 2.0 adversarial JSONL files found in data/fever2_adversarial/.")
        print("Download the Development Dataset from https://fever.ai/dataset/adversarial.html")
        return

    path = candidate_files[0]
    print(f"Reading FEVER 2.0 adversarial from {path}")

    examples = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            examples.append(obj)

    print(f"Showing {len(examples)} example(s) from FEVER 2.0 adversarial:")
    for i, ex in enumerate(examples, start=1):
        claim = ex.get("claim")
        label = ex.get("label") or ex.get("verdict")
        evidence = ex.get("evidence") or ex.get("evidences")
        print("-" * 80)
        print(f"Example {i}")
        print(f"Claim   : {claim}")
        print(f"Label   : {label}")
        print(f"Evidence: {evidence}")


def preview_conflictqa(n: int = 5) -> None:
    """
    Print a few examples from local ConflictQA JSON/JSONL files
    under data/conflictqa/.
    """
    candidate_files = list((DATA_DIR / "conflictqa").glob("*.json*"))
    if not candidate_files:
        print("No ConflictQA JSON files found in data/conflictqa/.")
        print("Either run download_datasets.py (with HuggingFace auth configured) or")
        print("manually download JSON files from https://huggingface.co/datasets/osunlp/ConflictQA/tree/main")
        return

    path = candidate_files[0]
    print(f"Reading ConflictQA from {path}")
    # ConflictQA files on HuggingFace are JSONL (one object per line)
    df = pd.read_json(path, lines=True)
    print(f"Total train examples: {len(df)}")

    print(f"Showing {min(n, len(df))} example(s) from ConflictQA:")
    for i in range(min(n, len(df))):
        row = df.iloc[i]
        question = row.get("question")
        answer = row.get("answer")
        context = row.get("context") or row.get("passage")
        print("-" * 80)
        print(f"Example {i + 1}")
        print(f"Question: {question}")
        print(f"Answer  : {answer}")
        print(f"Context : {str(context)[:500]}")


def preview_debateqa(n: int = 5) -> None:
    """
    Print a few examples from DebateQA files under data/debateqa/dataset/.
    """
    dataset_dir = DATA_DIR / "debateqa" / "dataset"
    candidate_files = list(dataset_dir.glob("*.json*"))
    if not candidate_files:
        print("No DebateQA files found in data/debateqa/dataset/.")
        print("Run download_datasets.py to fetch DebateQA from GitHub.")
        return

    path = candidate_files[0]
    print(f"Reading DebateQA from {path}")
    df = pd.read_json(path, lines=path.suffix == ".jsonl")
    print(f"Total examples: {len(df)}")

    print(f"Showing {min(n, len(df))} example(s) from DebateQA:")
    for i in range(min(n, len(df))):
        row = df.iloc[i]
        question = row.get("question") or row.get("query")
        partial_answers = row.get("partial_answers") or row.get("answers") or []
        if isinstance(partial_answers, list):
            pa_preview = partial_answers[:2]
        else:
            pa_preview = partial_answers
        print("-" * 80)
        print(f"Example {i + 1}")
        print(f"Question      : {question}")
        print(f"Perspectives  : {len(partial_answers) if isinstance(partial_answers, list) else 'n/a'}")
        print(f"Partial answer: {str(pa_preview)[:500]}")


if __name__ == "__main__":
    print("=== FEVER 2.0 Adversarial preview ===")
    preview_fever2_adversarial()
    print("\n=== ConflictQA preview ===")
    preview_conflictqa()
    print("\n=== DebateQA preview ===")
    preview_debateqa()

