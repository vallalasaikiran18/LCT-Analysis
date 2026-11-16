"""Stage 5 - blinded field test for ONE feature set (the winner, or the one named in config), and its evaluation.

The geologist receives only Site_IDs and coordinates. The list mixes the set's top points with random controls
(anywhere in the common domain) and low-score controls, shuffled across waves. Success criteria are written to
success_criteria.json BEFORE any visit, with the smallest result that would meet them.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

from . import utils as U

log = U.log


def _far(x, y, ax, ay, d):
    return not len(ax) or np.min(np.hypot(np.asarray(ax) - x, np.asarray(ay) - y)) >= d


def choose_set(cfg):
    fc = cfg["field_test"]
    if fc.get("sensor", "auto") != "auto":
        return fc["sensor"], "named in config"
    p = U.d_stage(cfg, "evaluation") / "sensor_comparison.csv"
    if not p.exists():
        raise RuntimeError("field_test.sensor is 'auto' but there is no evaluation. Name a set (ASTER | S2 | DEM) "
                           "in field_test.sensor, or provide occurrences and run --stage evaluate.")
    c = pd.read_csv(p)
    ok = c[c.carries_signal == True]  # noqa: E712
    if ok.empty:
        raise RuntimeError("No set passes the signal rule on held-out occurrences. Name a set explicitly in "
                           "field_test.sensor if you still want to field-test one.")
    best = ok.sort_values("holdout_auc", ascending=False).iloc[0]
    return best.set, f"best held-out AUC ({best.holdout_auc:.3f})"


def _boundary_p(n_top, n_ctl, hr, ratio):
    """p-value of the weakest result that still meets the hit-rate and ratio thresholds."""
    a = int(np.ceil(hr * n_top))
    c = int(np.floor((a / n_top) / ratio * n_ctl)) if ratio > 0 else n_ctl
    return a, c, float(fisher_exact([[a, n_top - a], [c, n_ctl - c]], alternative="greater")[1])


def build_field_test(cfg):
    from .points import add_geographic
    fc, sep = cfg["field_test"], cfg["points"]["min_separation_m"]
    s, why = choose_set(cfg)
    fd = U.d_stage(cfg, "field_test")
    leaf = U.d_leaf(cfg, "M1", s, create=False)
    pts = pd.read_csv(leaf / "points.csv")
    mp, _, grid = U.read_stack(leaf / "score_pct.tif")
    mp = mp[0]
    rng = np.random.default_rng(cfg["project"]["seed"] + 1)
    chosen, ax, ay = [], [], []

    def take(cat, df, n):
        got = 0
        for _, r in df.iterrows():
            if got >= n:
                break
            if _far(r.Easting, r.Northing, ax, ay, sep):
                chosen.append(dict(Category=cat, Source_ID=r.get("Point_ID", ""), Easting=r.Easting, Northing=r.Northing,
                                   Percentile=r.get("Score_pct", np.nan)))
                ax.append(r.Easting)
                ay.append(r.Northing)
                got += 1
        if got < n:
            log.warning("Field test: only %d of %d requested '%s' sites available", got, n, cat)

    comp = fc["composition"]
    take("Top", pts, comp.get("top", 20))

    def sample_pixels(mask):
        r, c = np.nonzero(mask)
        o = rng.permutation(len(r))
        x, y = grid.xy(r[o], c[o])
        return pd.DataFrame(dict(Easting=x, Northing=y, Score_pct=mp[r[o], c[o]]))
    valid = np.isfinite(mp)
    take("RandomControl", sample_pixels(valid), comp.get("random", 0))
    take("LowScoreControl", sample_pixels(valid & (mp <= fc["low_quantile"])), comp.get("low", 0))
    sites = pd.DataFrame(chosen)
    if sites.empty:
        raise RuntimeError("No field-test sites could be selected.")

    occ = U.load_points_file(cfg, "known_occurrences", grid)
    if occ is not None and len(occ):
        d = np.hypot(sites.Easting.values[:, None] - occ.geometry.x.values[None],
                     sites.Northing.values[:, None] - occ.geometry.y.values[None]).min(1)
        sites["dist_known_occurrence_m"] = d
        sites["near_known_occurrence"] = d <= fc.get("flag_near_known_m", 500)
        nk = int(sites[(sites.Category == "Top") & sites.near_known_occurrence].shape[0])
        if nk:
            log.warning("%d top sites lie within %s m of a known occurrence: a hit there is not new information. "
                        "They are flagged in the KEY; score the test with and without them.", nk, fc.get("flag_near_known_m"))

    sites["Wave"] = 0
    for cat, g in sites.groupby("Category"):
        sites.loc[rng.permutation(g.index), "Wave"] = (np.arange(len(g)) % fc["waves"]) + 1
    sites = sites.sample(frac=1, random_state=int(rng.integers(1 << 31))).sort_values("Wave", kind="stable").reset_index(drop=True)
    sites["Site_ID"] = [f"{fc['id_prefix']}-W{w}-{i + 1:03d}" for i, w in enumerate(sites.Wave)]
    sites = add_geographic(sites, grid)
    blind = sites[["Site_ID", "Wave", "Easting", "Northing", "UTM_zone", "Lat", "Lon"]]
    key = sites[[c for c in ["Site_ID", "Category", "Source_ID", "Percentile", "dist_known_occurrence_m",
                             "near_known_occurrence"] if c in sites]]
    template = blind[["Site_ID"]].copy()
    for c in fc["result_columns"]:
        template[c] = ""
    blind.to_csv(fd / "blind_site_list.csv", index=False)
    key.to_csv(fd / "KEY_do_not_share_with_geologist.csv", index=False)
    with pd.ExcelWriter(fd / "Field_Test.xlsx", engine="openpyxl") as xw:
        blind.to_excel(xw, sheet_name="Blind_Sites", index=False)
        template.to_excel(xw, sheet_name="Results_Template", index=False)

    n_top = int((sites.Category == "Top").sum())
    n_ctl = int(len(sites) - n_top)
    a, c, p = _boundary_p(n_top, n_ctl, fc["success_min_hit_rate"], fc["success_min_ratio"])
    crit = dict(created=datetime.now().isoformat(timespec="seconds"), feature_set=s, chosen_because=why,
                blind_list_hash=U.file_hash(fd / "blind_site_list.csv"), n_top=n_top, n_controls=n_ctl,
                hit_definition=f"{fc['hit_column']} = Y; sites with {fc['outcrop_column']} in "
                               f"{fc['non_informative_values']} are excluded as non-informative",
                criteria=dict(min_hit_rate_top=fc["success_min_hit_rate"], min_ratio_top_over_controls=fc["success_min_ratio"],
                              alpha_one_sided_fisher=fc["success_alpha"]),
                weakest_passing_result=dict(top_hits=a, control_hits=c, fisher_p=p,
                                            meets_alpha=bool(p < fc["success_alpha"])),
                note="Written before any field visit. Do not edit after visits start.")
    if p >= fc["success_alpha"]:
        log.warning("With %d top and %d control sites, the weakest result that meets the hit-rate and ratio "
                    "thresholds (%d/%d vs %d/%d) has p=%.3f >= alpha: add control sites or accept lower power.",
                    n_top, n_ctl, a, n_top, c, n_ctl, p)
    U.jdump(crit, fd / "success_criteria.json")
    log.info("Field test for set %s (%s): %d sites in %d waves. Give the geologist ONLY blind_site_list.csv.",
             s, why, len(sites), fc["waves"])
    return sites


def evaluate_field_results(cfg, results_path):
    fc = cfg["field_test"]
    fd = U.d_stage(cfg, "field_test")
    crit = U.read_json(fd / "success_criteria.json") if (fd / "success_criteria.json").exists() else {}
    key = pd.read_csv(fd / "KEY_do_not_share_with_geologist.csv")
    p = str(results_path)
    res = pd.read_excel(p) if p.lower().endswith((".xlsx", ".xls")) else pd.read_csv(p)
    df = key.merge(res, on="Site_ID", how="inner")
    hit_col, out_col = fc["hit_column"], fc["outcrop_column"]
    df = df[~df[out_col].fillna("").astype(str).str.lower().isin([str(s).lower() for s in fc["non_informative_values"]])]
    df["hit"] = df[hit_col].astype(str).str.upper().str.startswith("Y")

    def score(d):
        top = d[d.Category.isin(fc["top_categories"])]
        ctl = d[~d.Category.isin(fc["top_categories"])]
        a, b, c, e = int(top.hit.sum()), int((~top.hit).sum()), int(ctl.hit.sum()), int((~ctl.hit).sum())
        _, pval = fisher_exact([[a, b], [c, e]], alternative="greater")
        hr_t, hr_c = a / max(a + b, 1), c / max(c + e, 1)
        ratio = hr_t / hr_c if hr_c > 0 else np.inf
        ok = hr_t >= fc["success_min_hit_rate"] and ratio >= fc["success_min_ratio"] and pval < fc["success_alpha"]
        return dict(n_top=a + b, hits_top=a, hit_rate_top=hr_t, n_controls=c + e, hits_controls=c,
                    hit_rate_controls=hr_c, ratio=ratio, fisher_one_sided_p=pval, success_criterion_met=bool(ok))
    out = dict(feature_set=crit.get("feature_set"), **score(df))
    if "near_known_occurrence" in df:
        out["excluding_sites_near_known_occurrences"] = score(df[~df.near_known_occurrence.astype(bool)])
    out["by_category"] = df.groupby("Category").hit.agg(["count", "sum", "mean"]).reset_index().to_dict("records")
    out["note"] = "Non-informative (cover / no outcrop) sites excluded. Every visited site is a new label."
    U.jdump(out, fd / "field_evaluation.json")
    df.to_csv(fd / "field_labels.csv", index=False)
    log.info("Field evaluation (%s): top %.0f%% vs controls %.0f%% (ratio %.1f, p=%.3f) -> criterion %s",
             out["feature_set"], 100 * out["hit_rate_top"], 100 * out["hit_rate_controls"], out["ratio"],
             out["fisher_one_sided_p"], "MET" if out["success_criterion_met"] else "NOT met")
    from .compare import write_report
    write_report(cfg)
    return out
