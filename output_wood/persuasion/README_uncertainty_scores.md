# Persuasion uncertainty scores — figures

Canonical output for each model × dataset:

```
output_wood/persuasion/{Model}/{dataset}/uncertainty_scores/
  figures/          ← main comparison PNGs
  csv/              ← metrics tables
  summary.txt
```

## Main figures

1. `metrics_comparison.png` — Brier / AUROC / UCE by method
2. `ablation_comparison.png` — F → +S → +A → full U^pers
3. `bootstrap_brier_ci.png` — paired bootstrap ΔBrier
4. `recalibration_brier_comparison.png` — raw vs isotonic Brier

## Notes

- **DeepSeek / FEVER**: argument-quality judge used for A; includes U_token.
- **Other conditions**: no judge → A = 0.5 on flipped dialogues; no U_token.
- Index: `uncertainty_scores_index.csv`

Rebuild per-condition figures:

```bash
cd src && python3 run_persuasion_uncertainty_figures.py --all
```

## Cross-model comparison pages

One page per dataset, models side-by-side (4 models × FEVER / PopQA / DebateQA):

```
output_wood/persuasion/uncertainty_scores_compare/
  ablation_comparison_{fever,popqa,debateqa}.png
  metrics_comparison_{fever,popqa,debateqa}.png
  bootstrap_brier_ci_{fever,popqa,debateqa}.png
  bootstrap_significance_{fever,popqa,debateqa}.png
  metrics_significance_{fever,popqa,debateqa}.png
  ablation_significance_{fever,popqa,debateqa}.png
  csv/metrics_bootstrap_{dataset}.csv
  csv/ablation_bootstrap_{dataset}.csv
```

```bash
cd src && python3 plot_uncertainty_cross_condition.py
cd src && python3 plot_metrics_ablation_significance.py
```

- `bootstrap_significance_*.png` — method/baseline pairs from the existing bootstrap figure  
- `metrics_significance_*.png` — paired ΔBrier for methods on `metrics_comparison`  
- `ablation_significance_*.png` — paired ΔBrier for consecutive ablation steps on `ablation_comparison`