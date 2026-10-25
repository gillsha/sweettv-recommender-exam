# sweet.tv recommender exam — corrected target

Individual project by Elizaveta Gill, with disclosed Codex assistance.

## Current result

The previous >=300-second / all-unique relevance protocol was **wrong for Kaggle model selection**. The author identified the target mismatch. The earlier project is preserved at tag `baseline-before-target-fix-20261025` (commit `89efb09`); historical 150-run results are not evidence for the new choice.

The corrected target reconstructs fresh raw overlaps, qualifies user–airing coverage >=80%, counts qualifying airings per user/programme, and retains the five most frequent IDs. See [the complete protocol](revised/PROTOCOL.md) for explicit assumptions, tie handling, boundary airings, cold users and sensitivity checks. The source does not specify every detail of hidden-label construction; exact organizer equivalence is not claimed.

Final model: **Repeat + Personalized PageRank**, reciprocal rank fusion `0.5/(20+r_repeat)+0.5/(20+r_ppr)`. PPR uses 50 cosine co-view neighbors, damping 0.5 and personal-count restart. The model was selected on window A before evaluating independent window B or submitting to Kaggle.

Kaggle submission **57629703**, **COMPLETE**, public **0.40594**, private **0.39796**. Earlier submission 57603427: public 0.37389, private 0.35822. Scores are measured submissions, not leaderboard positions. Root `submission.csv` is the exact new submitted file; SHA-256:

`a1edc3775457c57ba1e0862a9eb36cec4ee355f07e7505dd2c1405d2027ad01c`

[Kaggle Notebook](https://www.kaggle.com/code/liugrowin/sweettv-recommender-exam), executed version **8**. The run remains private to the owner because its outputs include individual viewing histories. The complete self-contained notebook source and aggregate results are publicly accessible here without teacher authentication. Local/Kaggle final CSV identity: **True**.

## Main comparison

A: train 2020-03-09 to 2020-05-18 exclusive; validation 2020-05-18 to 2020-06-22 exclusive. B: train to 2020-06-22 exclusive; validation 2020-06-22 to 2020-07-27 exclusive. Main evaluation cohorts have at least five qualified unique programmes: **775 / 1078** users; every truth has exactly five IDs. Cohorts are identical across methods within each window. Twenty-seven focused configurations were evaluated on A; only frozen family winners were evaluated on B. Both horizons are five weeks, versus the competition's twelve-week forecast.

| Method | MAP@5 A | MAP@5 B | Fit + score + rank seconds A / B |
|---|---:|---:|---:|
| popularity | 0.29164731 | 0.25087817 | 0.080 / 0.095 |
| repeat | 0.39238710 | 0.36358071 | 0.084 / 0.096 |
| content | 0.22421505 | 0.19586580 | 0.707 / 0.875 |
| implicit64 | 0.17013333 | 0.14968151 | 3.016 / 3.811 |
| count64_bias_reg1 | 0.39009032 | 0.36157390 | 3.398 / 4.433 |
| ppr50 | 0.39382796 | 0.36551639 | 1.760 / 2.356 |
| hybrid_ppr50_0.5 | 0.39540645 | 0.36598639 | 2.196 / 2.882 |
| schedule_0.5 | 0.39195699 | 0.36170377 | 0.229 / 0.281 |


Times are the local measured fit + scoring + Top-5 ranking; hybrids include both component costs. Shared raw preparation, JIT warm-up and metric calculation are excluded consistently. Hardware/software details and separate Kaggle times are in `evidence/corrected/`. Main/forward all-1260 diagnostics (up to five truth IDs, empty AP=0) are recorded separately, not substituted for the strict-five metric.

AP@5 = sum of Precision@rank at hit positions / 5 for the main cohort. MAP is the mean over that fixed cohort; duplicate predictions receive no extra credit. For the diagnostic cohort the denominator is min(5, number of relevant IDs), with empty AP=0. An independent implementation checks each AP against `core.ap_at_k`.

## What changed and what did not work

- Fresh raw reconstruction retained **10,604** qualifying user–airings below 300 seconds. Unioning sessions changes qualification for **48,822** user–airings; exact duplicate sessions are removed. Short-airing checks and independent union arithmetic are recorded.
- At fixed 32 dimensions and lambda 0.1, replacing binary preference with `log1p(count)` improved main MAP from **0.15807 to 0.38492**. Item bias, regularization, factor dimension and confidence were then varied separately; all configurations remain in `main_results.json`.
- The selected count MF minimizes weighted squared errors over observed and unobserved pairs, plus L2 factor and item-bias penalties. Observed weight is `1+20*log1p(count)`; missing target 0 and weight 1. The bias is regularized, not an unpenalized popularity shortcut. Dense normal-equation verification covers nonzero bias and cold rows.
- A schedule-rate correction based only on future airings and historical channel preferences was **worse than Repeat on both windows** and excluded from the final model. The allowed future grid never supplies user behaviour or future content features.
- The final hybrid's improvements over Repeat are small: **+0.003019 / +0.002406** on A/B. Conditional paired-user bootstrap on B: 95% interval [0.000671, 0.004221], 2,000 resamples, seed 2026. This does not resolve label ambiguity.
- Descending-ID frequency ties produce hybrid/Repeat MAP **0.34649/0.34532** on A and **0.31934/0.31825** on B. This target uncertainty is much larger than the hybrid gain. Single-session and qualifying-session-count variants are also reported with frozen predictions on the same cohort. No alternative convention was selected using private scores.
- Only about 10.8% of first-schedule rows have labels. The join covers 50.9% of viewing rows and 53.1% of viewing seconds. Unlabelled airtime is not a negative preference; offline scores do not establish enjoyment or revenue.

## Reproduce

1. Personally accept [competition rules](https://www.kaggle.com/competitions/sweettv-tv-program-recommender/rules), download the five files into a private directory. `.csv.zip` inputs are supported. Never commit raw files or credentials.
2. Free Kaggle CPU: import `sweettv.ipynb`, attach `sweettv-tv-program-recommender`, disable Internet/GPU, Run All. All source modules are embedded; no installation or secret is embedded. It runs target tests, fresh reconstruction, A selection, freeze, B evaluation and full-history refit. The notebook does not submit automatically.
3. Local Python 3.12: create a virtual environment and `pip install -r requirements.txt`. From this repository run:
   ```sh
   OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python revised/test_target.py
   OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python pipeline.py --data /absolute/private/data --output /absolute/private/results
   ```
4. Inspect `final_manifest.json`, validate the SHA, exact template order, unique nonzero candidate IDs and score evidence. Different library/hardware versions may cause numerical variation; both measured environments are saved. Downloaded notebook outputs containing viewing histories must stay private.
5. Optional post-freeze diagnostics: `SWEETTV_DATA=/absolute/private/data SWEETTV_REVISION_OUT=/absolute/private/results python -m revised.diagnostics`. If a private `example_users.json` exists, those same two users are used for all methods; otherwise lower/upper quartiles of final AP are chosen. No per-user examples are included in this public repository.

Current implementation: `pipeline.py`, `revised/`, shared `als.py`, `models.py`, `core.py`; `build_notebook.py` rebuilds the self-contained notebook. Legacy files are clearly marked and are not invoked by the corrected entry point. Exact test CSV is `submission.csv`. Main/forward protocols, full parameters, time, environment and aggregate sensitivity are in `evidence/corrected/`.

The report, genuine private-score screenshot, personal verification and original AI-session export stay in the local course-submission materials. No teacher message was sent. The author personally checked a new AP calculation and discussed the target and ties; automated tests are agent checks, not student work.
