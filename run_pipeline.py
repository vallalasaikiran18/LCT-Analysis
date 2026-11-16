#!/usr/bin/env python
"""LCT pegmatite prospectivity - sensor comparison with the fuzzy gamma model (M1).

  python run_pipeline.py --config config.yaml --stage all
  python run_pipeline.py --stage features        # spectral check, QA, per-set selection, lineament provenance
  python run_pipeline.py --stage models          # M1 on ASTER, S2, DEM (+ one-at-a-time sensitivity)
  python run_pipeline.py --stage compare         # agreement, layer-count test, report
  python run_pipeline.py --stage evaluate        # known occurrences: split, per-layer evidence, curves, FREEZE
  python run_pipeline.py --stage field_test      # blinded list for the winning set + pre-registered criteria
  python run_pipeline.py --stage field_eval --results filled_results.xlsx
  python run_pipeline.py --stage spectral        # spectral check only
  python run_pipeline.py --stage models --sets S2 --models M1
Stages in 'all': features, models, compare, evaluate (field_test and field_eval are run on purpose, separately).
"""
import argparse
import sys
from pathlib import Path

from lct import utils as U


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--stage", default="all",
                    choices=["spectral", "features", "models", "compare", "evaluate", "field_test", "field_eval", "all"])
    ap.add_argument("--models", nargs="*", help="model keys (default: run.models in config, i.e. M1)")
    ap.add_argument("--sets", nargs="*", help="subset of feature sets (default: run.sets in config)")
    ap.add_argument("--results", help="filled field results (csv/xlsx) for --stage field_eval")
    ap.add_argument("--reset-freeze", action="store_true",
                    help="archive FREEZE.lock.json and re-split (ends the blind held-out test; logged)")
    a = ap.parse_args()

    cfg = U.load_config(a.config)
    out = Path(cfg["_out"])
    out.mkdir(parents=True, exist_ok=True)
    U.setup_logging(out / "pipeline.log", cfg["project"].get("log_level", "INFO"))
    st = a.stage

    if st == "spectral":
        from lct.features import classify_layers, discover_layers, load_layer_table
        from lct.spectral import run_spectral_check
        table = load_layer_table(cfg)
        run_spectral_check(cfg, table, classify_layers(cfg, discover_layers(cfg), table))
    if st in ("features", "all"):
        from lct.features import run_feature_engineering
        run_feature_engineering(cfg)
    if st in ("models", "all"):
        from lct.models import run_models_stage
        run_models_stage(cfg, a.models, a.sets)
    if st in ("compare", "all"):
        from lct.compare import run_compare
        run_compare(cfg)
    if st in ("evaluate", "all"):
        from lct.evaluation import run_evaluation
        run_evaluation(cfg, reset_freeze=a.reset_freeze)
    if st == "field_test":
        from lct.fieldtest import build_field_test
        build_field_test(cfg)
    if st == "field_eval":
        if not a.results:
            sys.exit("--results is required for --stage field_eval")
        from lct.fieldtest import evaluate_field_results
        evaluate_field_results(cfg, a.results)


if __name__ == "__main__":
    main()
