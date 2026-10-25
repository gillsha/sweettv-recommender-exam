> Historical record: the >=300-second/all-unique target below is invalid for Kaggle model selection. See revised/PROTOCOL.md and evidence/corrected for the replacement.

# Additional local search: 150 actual runs

This stage was explicitly requested after the original Kaggle notebook. It tests the student's proposal to mix personal frequency and overall popularity, adding a decay and frequency-exponent grid. It is not 150 Kaggle submissions and does not optimize the hidden private score.

## Fixed grid and selection

- Half-life h: 7, 14, 28, 56 days, or no decay.
- Personal-frequency power p: .25, .5, .75, 1, 1.5, 2.
- Popularity weight w: .05, .15, .30, .50, .75.
- Exactly 5 × 6 × 5 = 150 configurations; every prediction file is distinct.
- Score = (1-w) * (f_ui / max_j f_uj)^p + w * (pop_i / max_j pop_j).
- f_ui sums 2^(-age_days/h) across qualified distinct airings; age is measured from airing start to the training cutoff. No-decay uses weight 1.
- pop_i counts unique training viewers (all training users, not just the prediction cohort).
- Same 1260 users, 6384 eligible validation candidates, repeat policy, deterministic ties and cold-start popularity as notebook v7. All candidates, including unseen ones, remain present.
- Best hold-out selects the configuration; if none beats original Repeat, the original would be retained. The actual winner is run 63: h=28, p=.25, w=.30.

## Reproduce

Use Python 3.12 and `pip install -r requirements-local.txt` in an isolated environment. Run from the repository root. The main original study can still be reproduced with `sweettv.ipynb` on Kaggle as described in README.

1. Run the original notebook under your Kaggle account; download its outputs to `work/final_run`. This supplies `validation_truth.json` and `validation_predictions.json` for exact parity checks. These private per-user outputs are deliberately not published.
2. Download the real competition inputs to `work/data`: `dataset11-30.csv.zip`, all three `export_arh_...csv.zip` schedules, and the uncompressed `submission.csv` template. Personally accept the competition rules first; keep credentials outside the repository.
3. `python prepare_local.py` — reconstruct the same interactions and private caches.
4. `python -m unittest -v test_core test_local_search`
5. `python local_search.py` — run exactly 150 evaluations, refit the selected configuration on all allowed history, validate `work/local150/submission.csv`.

Paths can be overridden by SWEETTV_DATA, SWEETTV_OUTPUT and SWEETTV_REFERENCE environment variables. The code does not submit automatically. The 150 per-user prediction files and pickled interactions are private local artifacts; only aggregate metrics are published.

## Verification and cost

Local preprocessing matched the Kaggle relevance sets for all 1260 users and all original Repeat Top-5 lists exactly. Each run's AP was compared with the original implementation within 1e-12. Two new synthetic tests independently check half-life decay, mixture arithmetic, cold fallback and tie order. A second CSV check confirms template header/user order, five unique nonzero eligible IDs and SHA-256.

Measured local environment: arm64, 8 reported CPU cores; Python 3.12.14, NumPy 2.3.5, pandas 2.2.3. Search plus final refit took 25.0764 seconds after preprocessing; refit plus prediction 0.4181 seconds. Five frequency matrices were shared across 30 configurations each, rather than rebuilding identical features in every run. Per-run timing covers scoring, ranking and metrics; shared construction times are listed separately.

The fixed hold-out was reused for all 150 choices, so the selected score may be optimistic. No independent second-window generalization claim is made. Private score improved from .35566 to .35822; public score fell from .37739 to .37389. Both are recorded; neither was used for selection.
