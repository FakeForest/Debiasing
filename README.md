# Adversarial Debiasing Workshop

**Yuelin Li and Xiaoyu Chen**

**Final report** [Compiled PDF](WORKSHOP.pdf) · [LaTeX source](WORKSHOP.tex) · [Markdown version](WORKSHOP.md)

Predict whether income exceeds $50,000 in UCI Adult. Compare a standard classifier with adversarial training for demographic parity, equalized odds, and equal opportunity. The report answers Q1–Q12 with results from 12 training runs.

## Find the results and code

| What to read | File |
| --- | --- |
| Sex and race experiment results | [Sex table](results/sex/results.csv), [race table](results/race/results.csv) |
| Losses and metrics for each training epoch | [Sex run folders](results/sex), [race run folders](results/race). Open a model's folder for its history.csv. |
| Adversary loss and group score histograms | [Loss curves](results/sex/adversary_loss.png), [scores.png](results/sex/scores.png) |
| Feature correlations and threshold comparison | [Feature correlations](results/sex/sex_correlations.csv), [threshold metrics](results/sex/threshold_comparison.csv), [chosen thresholds](results/sex/thresholds.json) |
| Change data, models, or the training objective | [data.py](data.py), [models.py](models.py), [train.py](train.py) |
| Run the experiments and inspect metric definitions | [main.py](main.py), [metrics.py](metrics.py), [thresholds.py](thresholds.py) |

## Compile the report

Run `tectonic WORKSHOP.tex` from this folder. The source uses the plots in `results/sex/` and generates `WORKSHOP.pdf`.

## Reproduce the runs

Use Python 3.12. Unpack the [UCI Adult archive](https://archive.ics.uci.edu/static/public/2/adult.zip) into the data directory so it contains adult.data and adult.test. Run the following commands from the repository root.

```sh
python -m pip install -r requirements.txt
python main.py --suite --output-dir reproduced
python main.py --protected race --output-dir reproduced
```

Running main.py without arguments trains only the three core sex-based models. Each run uses 20 epochs and seed 42. The commands above save new checkpoints and results under reproduced/. The tracked files under results/ are the published experiment summaries.
