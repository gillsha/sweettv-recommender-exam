# Evidence log

## Access and audit

- Authenticated Kaggle account: `liugrowin`; GitHub API confirms `gillsha`.
- Initial Kaggle data download returned 403; notebook push explicitly required personal acceptance of competition rules. The student accepted them; template download and audit run then succeeded.
- Audit notebook version 1 failed with `ValueError: invalid literal for int() with base 10: '0 0 0 0 0'`. Cause: code checked only for a `tv_show_id` column, treating both schedule records and the submission's ranked-list field as scalar programme IDs. Fix: recognise a schedule by both `tv_show_id` and `channel_id`; save audit results incrementally. Version 2 produced `data_audit.json`, including the actual 1,260-row template. This is an agent debugging check, not student verification.
- Audit runtime: 42.986856119 seconds; Kaggle Python 3.13.15, pandas 2.3.3, NumPy 2.1.3, 4 reported CPU cores.
- Raw schedule contains a zero-duration row. The main pipeline counts nonpositive durations and excludes those rows from interval matching. Zero-length airtime cannot yield a positive overlap. Counts are retained in `schema_audit.json`.
- User identifiers are uint64 in the diagnostic read and strings in the modelling pipeline, preventing precision loss. Diagnostic numeric quantiles of IDs carry no modelling meaning and are not used as features.
- Diagnostic raw-string min/max of schedule timestamps are lexicographic, not chronological. All split dates are instead computed after explicit `%d.%m.%Y %H:%M:%S` parsing in the main pipeline.

## Agent contract tests

Five synthetic tests pass: AP duplicates/empty truth/cutoff; fixed cohort; interval boundaries/channel/unlabelled rows; union and time clipping; comparison with brute force for 300 random overlap queries (seed 2026). These are independent of real-data quality and do not constitute the student's personal check.

## Local environment limitation

A small optional local model-test dependency installation failed building `implicit==0.7.2` on macOS arm64 (`ld: symbol(s) not found for architecture arm64`). No real-data model training was attempted locally. The requested main computation uses Kaggle's Linux CPU environment. Final Kaggle success and measured versions must be confirmed separately.

## Selection protocol

Before inspecting a Kaggle score: compare popularity, content, implicit ALS32 and ALS64 on the same time hold-out. The controlled ALS iteration changes factors only. Evaluate a repeat-count component and predeclared repeat/ALS rank-fusion weights .25, .5 and .75 separately. Select the highest hold-out MAP@5, then refit on all permitted history and submit. Report all tried variants, including worse ones.



## Actual failures and recovery

- Main versions 1–2 could not install implicit from PyPI (DNS/network unavailable in the Kaggle runtime). The final offline notebook uses an explicit NumPy/Numba weighted ALS solver. Its update was checked against independent dense normal equations (maximum absolute error 1.7763568394002505e-15); all five core tests also pass inside Kaggle.
- Versions 3–4 stopped on the viewing-duration assertion. Investigation found 117 intervals crossing 2020-03-29 with a raw clock difference 3600 seconds longer than duraton. The training schedule has no 03:xx start timestamps that day. Both input time axes are normalized at 04:00, without inferring an absolute timezone. The final run verifies zero remaining mismatches over all 5,821,987 rows; two real crossing intervals and an ordinary interval also pass the separate local regression.
- Version 5's intermediate exclusion of inconsistent intervals was superseded, not selected. Version 6 used corrected time but inefficient Python iteration over every candidate during Top-5 selection. Version 7 vectorizes that filtering; equivalence was checked on 300 random cases. Version 7 is the designated COMPLETE run. Its main pipeline took 177.381844917 s; no real-data model training ran locally.

## Measured controlled iteration and selection

- popularity: MAP@5=0.7293994709; fit+score=0.085472 s.
- content: MAP@5=0.3507142857; fit+score=2.173259 s.
- als32: MAP@5=0.2386507937; fit+score=4.812590 s.
- als64: MAP@5=0.1601375661; fit+score=14.920388 s.
- repeat: MAP@5=0.7608148148; fit+score=0.000000 s.
- hybrid_0.25: MAP@5=0.4362566138; fit+score=0.157684 s.
- hybrid_0.5: MAP@5=0.6437619048; fit+score=0.165690 s.
- hybrid_0.75: MAP@5=0.7509074074; fit+score=0.159555 s.

ALS32→ALS64 changes only factor count; MAP falls by 0.07851323. The cause is not established; more factors alone do not help this protocol. Repeat's separate fit/score timer is unmeasured (stored zero), not instantaneous. Hybrid times include fusion only. The best configuration by hold-out was Repeat; after all-history refitting its exact CSV was independently checked and submitted once. Submission 57603211 is COMPLETE, public 0.37739, private 0.35566. Selection preceded this score; no private-score tuning occurred. Independent MAP recalculation agrees for every method within 1e-12.

Student AP calculation and reflections belong to the course report; the tests and debugging above are agent work.
