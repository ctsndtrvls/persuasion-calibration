## Persuasion-based approach for evaluating LLM calibration

This repository contains code and experiments for a bachelor thesis on a **persuasion-based approach to evaluating LLM calibration**.

### Datasets

- **FEVER 2.0 Adversarial**: adversarial claims with labels (*Supported*, *Refuted*, *NotEnoughInfo*), created as part of the FEVER 2.0 shared task. See the official page for details and downloads: [FEVER 2.0 Adversarial Dataset](https://fever.ai/dataset/adversarial.html).
- **ConflictQA**: a question answering dataset from the OSU NLP group, available on HuggingFace: [osunlp/ConflictQA](https://huggingface.co/datasets/osunlp/ConflictQA/tree/main).
- **DebateQA**: a dataset for QA on debatable knowledge, available on GitHub: [pillowsofwind/DebateQA](https://github.com/pillowsofwind/DebateQA).

Local data layout:

- `data/fever2_adversarial/` – FEVER 2.0 adversarial JSONL files (downloaded manually from the FEVER site).
- `data/conflictqa/` – ConflictQA splits saved as JSONL via `datasets`.
- `data/debateqa/` – DebateQA files downloaded from GitHub archive.

### Project structure

- `data/`
  - `fever2_adversarial/`
  - `conflictqa/`
- `src/`
  - `download_datasets.py` – script to download and save `osunlp/ConflictQA` locally.
  - `build_debateqa_subsets.py` – creates stratified DebateQA subsets (480 and 600 rows total).
  - `inspect_data.py` – quick preview of FEVER, ConflictQA and DebateQA examples.
- `venv/` – Python virtual environment.
- `requirements.txt` – Python dependencies.

### Setup

From the repository root (the `persuasion-calibration` directory):

```bash
python3 -m venv venv
source venv/bin/activate  # on macOS / Linux
pip install -r requirements.txt
```

