# Persuasion-based approach for evaluating LLM calibration

Code, experiment runners, and results for a **persuasion-based evaluation of LLM calibration**: multi-turn dialogues where a persuader challenges a target model’s answer, and we track confidence, flips, argument quality, and uncertainty.

## What is in this repository

| Path | Contents |
|------|----------|
| `src/` | Python entry points: data prep, baseline calibration, persuasion rollouts, judges, analysis, plots |
| `scripts/` | Shell wrappers for full experiment batches |
| `data/` | Local dataset files (FEVER 2.0 Adversarial, ConflictQA, DebateQA) |
| `output_wood/` | Experiment outputs (CSV/JSON/plots) used in the evaluation |
| `output/` | Older baseline calibration artefacts |
| `requirements.txt` | Python dependencies |

Models covered in the main persuasion runs: **GPT-4o**, **DeepSeek**, **Qwen**, **Gemma** (via OpenAI / DeepSeek / OpenRouter as configured in the scripts).

## Datasets

- **FEVER 2.0 Adversarial** — [fever.ai](https://fever.ai/dataset/adversarial.html) → `data/fever2_adversarial/`
- **ConflictQA (PopQA / StrategyQA)** — [osunlp/ConflictQA](https://huggingface.co/datasets/osunlp/ConflictQA) → `data/conflictqa/`
- **DebateQA** — [pillowsofwind/DebateQA](https://github.com/pillowsofwind/DebateQA) → `data/debateqa/`

Helpers: `src/download_datasets.py`, `src/build_conflictqa_subsets*.py`, `src/build_debateqa_subsets.py`, `src/inspect_data.py`.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate   # macOS / Linux
pip install -r requirements.txt
cp .env.example .env       # then add API keys as needed
```

API keys are read from `.env` (gitignored). Only providers used by a given script need to be filled in.

## Main experiment areas

1. **Baseline calibration** — token-probability / self-reported confidence / epistemic markers / temperature; see `src/collect_conflictqa_ece.py`, `src/run_epistemic_marker_experiment.py`, and `scripts/run_tokenprob_*.sh`, `scripts/run_temp*.sh`, `scripts/run_epistemic_*.sh`. Outputs under `output_wood/{token_prob_confidence,self_reported_confidence,epistemic_markers,temperature_experiments,...}/`.

2. **Persuasion dialogues** — FEVER and QA multi-turn rollouts: `src/run_persuasion_fever_pilot.py`, `src/run_persuasion_qa_multi.py`, wrappers in `scripts/run_persuasion_*.sh`. Results: `output_wood/persuasion/{DeepSeek,GPT-4o,Qwen,Gemma}/`.

3. **Argument quality** — LLM judge + human annotation analysis: `src/run_persuasion_arg_quality_judge.py`, `src/analyze_arg_quality_*.py`, `src/analyze_human_annotator_*.py`.

4. **Persuasion uncertainty / composite scores** — `src/build_persuasion_uncertainty_scores.py`, `src/evaluate_persuasion_uncertainty.py`, `src/run_persuasion_uncertainty_figures.py`. See `output_wood/persuasion/README_uncertainty_scores.md`.

Most plot scripts live in `src/plot_*.py` with matching launchers under `scripts/`.

## Notes

- Large CSV/JSON result files may use Git LFS (see `.gitattributes`).
- Do not commit `.env` or API keys.
- `venv/` is local-only and ignored by git.
