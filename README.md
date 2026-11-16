# LCT-pegmatite prospectivity: which data source carries signal? (fuzzy gamma, one model per source)

The fuzzy gamma model (M1) is run **independently** on three feature sets: ASTER-only, Sentinel-2-only and
DEM-only. Each set gets its own feature selection, all three are scored on the **same pixels**, ranked as
**percentiles**, and give the **same number of points**. They are then judged against **held-out known
occurrences**, and the winner goes to a **blinded field test** with criteria written down before any visit.

## Install
    pip install -r requirements.txt

## Before the first run (config.yaml)
1. `paths`, `inputs.aoi`: as before.
2. `inputs.known_occurrences`: your occurrence file (mines, workings, mapped Li-pegmatites, survey records).
   Without it the comparison can only describe stability and agreement, which do not show which source is right.
3. `spectral_check.library_dir`: download USGS splib07a (ASCII, ASD 0.35-2.5 um, https://doi.org/10.5066/F7RR1WDJ).
   Check that `spectral_check.minerals` patterns match file names in your copy; missing minerals are reported.
4. `engineer.vector_sources.lineaments.derived_from`: DEM, imagery or unknown. If the lineaments were drawn on
   Sentinel-2 / ASTER / Landsat, move them out of the DEM set (`feature_set: S2`, or `null` to drop them).
5. Review `name_rules`, `target_rules`, `models.fuzzy_gamma` (directions, AHP priorities). **These are frozen at the
   first `--stage evaluate`.**

## Run
    python run_pipeline.py --config config.yaml --stage features   # spectral check, QA, per-set selection, lineament provenance
    python run_pipeline.py --config config.yaml --stage models     # M1 on ASTER, S2, DEM + one-at-a-time sensitivity
    python run_pipeline.py --config config.yaml --stage compare    # agreement, layer-count test, report
    python run_pipeline.py --config config.yaml --stage evaluate   # occurrences: split, per-layer evidence, curves, FREEZE
    python run_pipeline.py --config config.yaml --stage field_test # blinded list for the winning set
    python run_pipeline.py --config config.yaml --stage field_eval --results filled_results.xlsx
    python run_pipeline.py --stage all                             # features -> models -> compare -> evaluate

## The rules that keep the comparison honest
- **Per-set selection.** Duplicate/collinearity/VIF pruning and family caps run inside each set, so an ASTER layer can
  never eliminate a Sentinel-2 layer. Each sensor's vegetation check uses its own NDVI (its own date).
- **Common domain.** Each sensor's own masks are built (ASTER vegetation from AST-NDVI, S2 from S2-NDVI, water), and
  their union is removed from every set. All sets are scored on identical pixels.
- **Rank percentiles.** Raw gamma scores depend on how many layers a set has (see `03_comparison/layer_count_test.png`),
  so maps, points, tiers and uncertainty are all in percentile units. Never compare `score_raw.tif` across sets.
- **Calibration vs held-out.** Occurrences are declustered and split by spatial blocks. Per-layer evidence uses
  calibration occurrences only: refine weights from `04_evaluation/layer_evidence.xlsx`.
- **Freeze.** The first `--stage evaluate` writes `FREEZE.lock.json`. If you then change rules, weights or features and
  re-run a set, it is detected and that set is scored on the held-out set only, logged in `revision_log.csv`.
  Every new revision uses up a little of the held-out set; after `evaluation.max_holdout_consults` you are warned.
  `--reset-freeze` re-splits but ends the blind test (the old lock is archived, not deleted).
- **Field test.** `success_criteria.json` is written with the blind list and shows the weakest result that would pass.
  Top sites near known occurrences are flagged in the KEY; the evaluation is reported with and without them.

## Outputs
    01_feature_engineering/
        00_spectral_check/   spectral_check.csv (verdict per ratio: discriminates | relabel_generic | non_diagnostic |
                             direction_conflict | target_not_in_library | not_checked), library_swir.png
        01_qa_inventory/     lineament_provenance.json, layer_inventory.csv, ...
        03_audit/<ASTER|Sentinel2|DEM>/   selection per set
        06_feature_store/    feature_dictionary.xlsx (features, all_candidates, spectral_check)
    02_models/M1_Fuzzy_Gamma/<ASTER|Sentinel2|DEM>/
        score_pct.tif, score_pct_sd.tif, tier.tif, score_raw.tif, points.csv, prospective_points.gpkg,
        Prospective_Points.xlsx, contributions.csv, feature_weights.csv, sensitivity_oat.csv, metrics.json, plots/
    03_comparison/   agreement_between_sets.*, maps_side_by_side.png, layer_count_test.*, sensor_summary_label_free.csv
    04_evaluation/   sensor_comparison.xlsx, layer_evidence.xlsx, success_prediction_curves.png,
                     layer_count_test_calibration.*, occurrence_split.csv, FREEZE.lock.json, revision_log.csv,
                     recommendation.md   <- the written summary
    05_field_test/   blind_site_list.csv (give ONLY this to the geologist), KEY_..., success_criteria.json

## Smoke test without real data
    python tools/make_synthetic_data.py syn_in 160
Point `paths.input_root` at `syn_in` and set `inputs.known_occurrences: "Occurrences/occurrences.csv"`. The synthetic
spectral library it writes is invented for testing only; it is not USGS data.

M2 (Isolation Forest) and M3 (TOPSIS+CRITIC) remain in `lct/models.py` and run with `--models M2 M3`; they are not
part of the default run, and consensus across models has been removed.
