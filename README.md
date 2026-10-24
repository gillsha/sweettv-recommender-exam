# sweet.tv recommender exam

Individual project by Elizaveta Gill, prepared with Codex assistance.

## Current final result after 150 local runs

The user requested 150 additional local runs after the initial Kaggle study. All 150 configurations were actually evaluated and produced distinct prediction lists. Selected run **63**: half-life **28 days**, frequency power **0.25**, popularity weight **0.30**. Hold-out MAP@5 **0.78969841**, previously **0.76081481**. The selection did not use a Kaggle score. New submission **57603427**, **COMPLETE**, private **0.35822**, public **0.37389**.

The root `submission.csv` is now the exact NEW final file; SHA-256 `b6ac7414d200ae4576311e4768bd9fc3a56aa79211530fbeea3c7e1b26ca90b9`. Earlier files remain recoverable in Git history. See [local reproduction instructions](LOCAL_SEARCH.md), `evidence/search_results.csv` (150 rows) and `evidence/search_summary.json`. The linked Kaggle notebook v7 records the initial eight-method study; the additional search and final refit ran locally as requested.

## Initial Kaggle study (preserved comparison)

- [Kaggle Notebook](https://www.kaggle.com/code/liugrowin/sweettv-recommender-exam), executed version 7. The Kaggle notebook is private to the owner; the full reproducible source is public in this repository.
- Final configuration selected on hold-out: **repeat**, MAP@5 **0.76081481**.
- Kaggle submission **57603211**, status **COMPLETE**, private score **0.35566**. The frozen leaderboard position is not used.
- Initial file (Git commit 479c876, superseded at repository root), SHA-256 `65d4c7509b0a8b323e9446ccf33d3f63768c32229fd7972b9c5be02ae5cf9f5c`.

## Fair comparison

| Method | MAP@5, all 1260 users | MAP@5, active users | Fit + scoring seconds |
|---|---:|---:|---:|
| popularity | 0.72939947 | 0.73759497 | 0.085 |
| content | 0.35071429 | 0.35465490 | 2.173 |
| als32 | 0.23865079 | 0.24133226 | 4.813 |
| als64 | 0.16013757 | 0.16193686 | 14.920 |
| repeat | 0.76081481 | 0.76936330 | 0.000 |
| hybrid_0.25 | 0.43625661 | 0.44115837 | 0.158 |
| hybrid_0.5 | 0.64376190 | 0.65099518 | 0.166 |
| hybrid_0.75 | 0.75090741 | 0.75934457 | 0.160 |

Primary metric includes empty-relevance users with AP=0. Active-only MAP is diagnostic, not the selection criterion. Cold-start uses the same popularity fallback in every model. All methods allow repeats and use the same 6384 validation candidates. ALS32 -> ALS64 changes only factor count. All other hyperparameters and the seed are fixed. Hybrid timings exclude component fitting; Repeat's separate timer was not measured (stored as 0). Ranking and metric calculation are outside the fit/score timer. Exact configurations and timings are in `evidence/run_manifest.json` and `evidence/validation_results.json`.

## Reproduce in Kaggle (free CPU)

1. Personally accept [competition rules](https://www.kaggle.com/competitions/sweettv-tv-program-recommender/rules) in your own Kaggle account.
2. Create a Python Notebook, attach `sweettv-tv-program-recommender`, select CPU, and import `sweettv.ipynb`. Internet and GPU are not required. **Run All**. All implementation modules are embedded in the notebook; no private local path or credential is needed.
3. Or use the official Kaggle CLI: copy `kernel-metadata.json`, change its owner/id to your own unused notebook slug, then `kaggle kernels push -p .`. Never overwrite somebody else's notebook.
4. Wait for COMPLETE and download outputs with `kaggle kernels output YOUR_OWNER/YOUR_SLUG -p run_outputs`.
5. Inspect `validation_results.json`, `coverage.json`, `user_examples.json`, and `run_manifest.json`; compare the generated submission SHA with the recorded hash. Minor floating-point differences on different library/hardware versions may affect exact ties.
6. Submit only after checking the generated CSV against the actual template. The source never submits automatically. This repository includes the exact file submitted for the reported score.

The measured environment is Python 3.13.15, Intel(R) Xeon(R) CPU @ 2.20GHz, 4 reported CPU cores. Package versions are pinned in `requirements.txt`. Main pipeline runtime: 177.4 s; final refit+score: 3.4 s. Kaggle startup and previous failed runs are additional.

## Design

- Same-channel interval overlap, half-open intervals; union duplicated/overlapping views per airing; watched threshold 300 seconds. `duraton` is a viewing duration, never a programme duration. `tv_show_id=0` is unlabelled airtime, never a recommendation.
- A 3600-second clock discontinuity on 2020-03-29 was identified in training timestamps. Both schedule and viewing timestamps at/after 04:00 are mapped back by one hour onto a continuous relative axis; no absolute timezone is claimed. All 5,821,987 viewing durations are checked after the correction. Original and corrected mismatch counts are retained in the coverage evidence.
- Train before 2020-05-18; validation from 2020-05-18 to 2020-07-27. Exact prediction cohort is the original template's 1260 user IDs, read as strings. Candidate IDs come from the future airing grid only. Content vocabulary, IDF, popularity and factors use historical data only.
- AP@5 = sum(precision at each hit rank) / min(5, number of unique relevant programmes); empty truth gives 0. MAP is the mean over the fixed cohort.
- Popularity, attribute TF-IDF content filtering, custom weighted implicit ALS, repeat count, and separately evaluated reciprocal-rank hybrids. Confidence is 1+20*log(1+qualified-airing-count), regularization .1, 20 ALS iterations, seed 2026.
- Final training recombines airings crossing the old split before applying the threshold, rather than counting one airing twice.

## Files and verification

- `core.py`, `models.py`, `als.py`, `pipeline.py`: implementation.
- `sweettv.ipynb`: self-contained Kaggle notebook; `build_notebook.py` rebuilds it from modules.
- `test_core.py`: five synthetic contract tests; run `python -m unittest -v test_core`. The Kaggle notebook runs them as well.
- `als.verify_solver()`: independent dense normal-equation check and cold-user check.
- `evidence/`: aggregate measured results, environment and CSV validation. The genuine Kaggle screenshot is included only in the local course report.
- `EXPERIMENT_LOG.md`: actual failures and corrections; `PROTOCOL.md`: fixed validation design.

Raw competition CSV files, credentials and the private AI-session export are excluded. Individual viewing histories and the two illustrative examples are retained in the local course report and private notebook outputs, not published here. The student's personal verification is recorded separately in the course report; agent tests are not claimed as student work. There was no online A/B test, enjoyment measurement or revenue estimate.
