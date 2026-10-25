# Corrected target protocol (frozen before model selection)

Primary source, read 2026-10-25: https://www.kaggle.com/competitions/sweettv-tv-program-recommender/data

Kaggle states: discard ID 0; retain screen-time ratios >=0.8; count their frequency per user/programme; select strictly five most frequent programmes. Public/private split is by users (50/50), not by time. Ranking of predictions matters for AP@5; the order of relevance IDs does not enter the standard AP formula. The source contains an inconsistent 'ASC' phrase but clarifies that #1 is most frequent and explicitly uses DESC in its construction steps.

The previous >=300-second/all-unique relevance task is INVALID as a proxy for this competition. Git tag baseline-before-target-fix-20261025 preserves it for diagnosis, not for final selection.

## Unspecified details: explicit assumptions, not organizer-confirmed code

- Viewing unit: (user, unique airing), where airing=(channel, start, end, programme). One qualifying airing contributes one frequency count. Repeated broadcasts of the same programme contribute separately.
- Duplicate schedule keys are removed. Viewing intervals are intersected by channel with every overlapping full airing; ambiguous overlapping schedule entries are not assigned arbitrarily to one programme. Identical session intervals are deduplicated and overlaps are unioned before computing screen-time; exact integer comparison 5*watched >= 4*duration avoids floating threshold errors.
- Fresh raw reconstruction keeps ALL positive overlaps, including qualifying programmes shorter than 300 seconds. No old filtered cache is used.
- In each split, only airings fully contained in the training or validation interval enter that window. Boundary-crossing airings are excluded from both sides of that split; denominators remain full airing durations. Final fitting permits all airings ending by the final history cutoff. This prevents future portions of an airing influencing an earlier model.
- Frequency ties: lower programme ID first, including at fifth place. This is an explicit deterministic assumption. Sensitivity: descending ID and seeded-hash ties (seed 2026).
- Session sensitivity: require a single original session to cover >=80%, still counting once per airing; separately count unique qualifying sessions. These variants diagnose the source's ambiguity; they do not replace the declared main target post hoc.
- Main evaluation cohort: template users with >=5 qualifying unique programmes in the given validation window, so every main relevance list contains exactly five. The same cohort is used by every method within a window. Also report all 1260 template users with available up-to-five relevance IDs, empty AP=0, denominator min(5,|R|); this is diagnostic and explicitly differs from Kaggle's strictly-five population.
- MAP@5 averages AP@5 over the fixed main cohort. Prediction ties use training airing popularity then ID. Repeats allowed. Candidate pool: all nonzero programmes with a positive-duration airing overlapping the forecast window. Cold users and missing/nonpositive model scores use the same training airing-count popularity.

## Time windows and selection

All dates refer to the wall-clock notation in the supplied files; the verified March 29 discontinuity is normalized identically in schedules and logs.

- Main selection: train [2020-03-09, 2020-05-18); validation [2020-05-18, 2020-06-22).
- Independent forward check: train [2020-03-09, 2020-06-22); validation [2020-06-22, 2020-07-27).
- Final: train all allowed history ending before 2020-07-27; forecast the test schedule through October 19.
- Main and independent validation intervals do not overlap. Configurations and the final model identity are frozen using the main window BEFORE opening the second-window metrics. No retuning based on second-window or Kaggle scores. Both local horizons are five weeks, shorter than the twelve-week competition horizon; report this limitation.

## Focused experiments

All fits use seed 2026 and two CPU threads; time includes model construction, fit, scoring and Top-5 ranking, excluding shared raw preprocessing, JIT warm-up and metric calculation. Hybrid cost includes its fitted components, even when cached for experiment efficiency.

1. Global airing-count popularity; personal qualified-airing frequency; content TF-IDF as an auxiliary comparison.
2. Existing binary-preference implicit ALS: 32 factors, confidence 1+20*log1p(count), lambda .1, 20 iterations.
3. Count MF: change target only to log1p(count); add regularized item-popularity bias; then change regularization, dimension and confidence separately. Every change is recorded rather than attributing differences to overfitting without evidence.
4. Personalized PageRank on a training-only programme co-view graph, with personal count restart. Test restart/damping and neighbor count separately.
5. Only after individual models: one-component additions to personal counts, and schedule opportunity adjustment from programme-by-channel airing rates weighted by training channel preferences. Future schedule supplies airings only, never user behaviour or future content attributes.
6. Preserve every measured configuration and failed run. Freeze the best configuration per family and the overall winner before forward evaluation.

## Count-factorization objective

`sum_(u,i) c_ui * (log(1+n_ui) - x_u^T y_i - b_i)^2 + lambda*(||X||_F^2+||Y||_F^2) + lambda_b*sum_i b_i^2`. Missing pairs have n=0, target=0 and c=1; observed pairs have c=1+alpha*log(1+n). The bias is optional and lambda_b=1 throughout. The selected MF comparison has 64 factors, lambda=1, alpha=20, 20 alternating updates, seed=2026. All missing pairs are included algebraically via a shared Gram matrix, not ignored as missing ratings. The reported binary-to-log-count comparison keeps dimensions, factor regularization, confidence and initialization fixed.

## Metadata naming

In the original measured JSON, final_manifest.training_users is the number of matrix rows (including four template cold users), not the number of users with observations. Actual observed training users: 4841; matrix rows: 4845. In forward_sensitivity, map5_fixed_main_cohort means the fixed cohort of that file's window (1078 forward users). These field-name clarifications do not change any prediction or metric.
