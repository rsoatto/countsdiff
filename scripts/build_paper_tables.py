"""Aggregate current-metric eval JSONs into camera-ready LaTeX tables.

Scans data/dnadiff/imputed_data/**/results_*.json, keeps only FULL-metric
JSONs (10 keys incl. energy_distance/mmd/swd/pearson), and for each
(method, dataset, masktype, dropout, n_imputations) keeps the MOST RECENT file
(recent overrides old). Excludes revision_mmd_broken (broken MMD). The
blackout/ folder's model is relabeled <model>_blackout.

Emits, per task: an availability matrix, a MAIN table (7 cols, grouped
sample-level | distributional), and an APPENDIX table (all metrics).
Reports mean (standard error) over resamples; scFID and MMD are both shown
as natural log (log(scFID), log(MMD)) to handle their wide dynamic range.
"""
import glob
import json
import math
import os
import re

import numpy as np

BASE = "data/dnadiff/imputed_data"
EXCLUDE_DIRS = {"revision_mmd_broken", "at_submission"}  # at_submission is 6-key anyway
FULL_KEYS = {"r2", "rmse", "mae", "raw_bias", "spearman_corr", "pearson_corr",
             "energy_distance", "scfid", "mmd", "swd"}
FNAME_RE = re.compile(
    r"^results_(.+?)_(fetus|heart|zero_shot)_(MCAR|MNAR_low|MNAR_high|MNAR)_([\d.]+)_(\d+)imputations(?:_(\w+))?\.json$"
)

TASKS = [("fetus", "MCAR", "0.5"), ("fetus", "MNAR_low", "0.25"), ("heart", "MCAR", "0.5")]

# Methods grouped by category; bolding (best/2nd) is computed WITHIN each group.
GROUPS = [
    [  # naive baselines
        ("Zero imputation", "raw", None),
        ("Mean imputation", "mean", None),
        ("Conditional Mean", "conditional_mean", None),
    ],
    [  # non-generative
        ("MAGIC", "magic", None),
        ("scIDPMs, 1-sample", "scidpm", "1"),
        ("scIDPMs, 5-sample", "scidpm", "5"),
        ("GAIN", "gain", None),
        ("Hi-VAE (Poisson)", "hivae", None),
        ("Hi-VAE (Gaussian)", "hivae-gaussian", None),
        ("scGPT (scratch)", "scgpt_scratch", None),
        ("scGPT (pretrained)", "scgpt_pretrained", None),
        ("xTrimoGene", "xtrimogene", None),
    ],
    [  # generative
        ("Forest-Diffusion", "forest", None),
        ("ReMDM, 1-sample", "remdm", "1"),
        ("ReMDM, 5-sample", "remdm", "5"),
        ("Blackout", "countsdiff-blackout", "1"),
        ("\\textbf{CountsDiff (Ours),} 1-sample", "countsdiff", "1"),
        ("\\textbf{CountsDiff (Ours),} 5-sample", "countsdiff", "5"),
    ],
]
ROWS = [r for g in GROUPS for r in g]  # flat, for availability listing


def load_index():
    idx = {}  # (model, ds, mask, drop, nimp) -> (mtime, dict)
    for path in glob.glob(f"{BASE}/**/results_*.json", recursive=True):
        parts = path.split(os.sep)
        folder = parts[-2]
        if folder in EXCLUDE_DIRS:
            continue
        m = FNAME_RE.match(os.path.basename(path))
        if not m:
            continue
        model, ds, mask, drop, nimp, suffix = m.groups()
        if suffix in ("noround", "bkp"):
            continue
        try:
            d = json.load(open(path))
        except Exception:
            continue
        if not FULL_KEYS.issubset(d.keys()):
            continue
        if folder == "blackout":
            model = model + "-blackout"  # match camera_ready's 'countsdiff-blackout'
        if model == "forestdiff":
            model = "forest"  # our .pt-derived forest (from _forest_overnight)
        # IGNORE camera_ready's 'forest' (evaluated against the WRONG mask -> inflated).
        # Only trust the .pt-derived forest from _forest_overnight.
        if model == "forest" and folder != "_forest_overnight":
            continue
        if folder == "hivae_gaussian" and model == "hivae":
            model = "hivae-gaussian"
        key = (model, ds, mask, drop, nimp)
        mt = os.path.getmtime(path)
        if key not in idx or mt > idx[key][0]:
            idx[key] = (mt, d)
    return idx


def ms(vals):
    a = np.array(vals, dtype=float)
    a = a[~np.isnan(a)]
    if len(a) == 0:
        return None
    return a.mean(), a.std(ddof=1) / math.sqrt(len(a)) if len(a) > 1 else 0.0


def fmt(vals, kind="plain"):
    r = ms(vals)
    if r is None:
        return "N/A"
    mu, se = r
    if kind == "logscfid":
        r = ms(np.log(np.clip(np.array(vals, dtype=float), 1e-30, None)))
        mu, se = r
        return f"{mu:.2f}({se:.2f})"
    if kind == "mmd":  # report log(MMD), natural log, like log(scFID)
        r = ms(np.log(np.clip(np.array(vals, dtype=float), 1e-30, None)))
        mu, se = r
        return f"{mu:.2f}({se:.2f})"
    if abs(mu) >= 1e3:
        return "$<\\!-10^3$" if mu < 0 else "$>\\!10^3$"
    if abs(mu) < 0.005:  # avoid an ugly "-0.00"
        mu = 0.0
    return f"{mu:.2f}({se:.2f})"


def get(idx, model, ds, mask, drop, nimp):
    if nimp is not None:
        return idx.get((model, ds, mask, drop, nimp), (None, None))[1]
    # any nimp: prefer 1, then 5, then anything
    for n in ["1", "5", "10", "2"]:
        if (model, ds, mask, drop, n) in idx:
            return idx[(model, ds, mask, drop, n)][1]
    for k, v in idx.items():
        if k[0] == model and k[1] == ds and k[2] == mask and k[3] == drop:
            return v[1]
    return None


CITE_LIST = (
    "Baselines: MAGIC~\\citep{van2018recovering}, scIDPMs~\\citep{zhang2024scidpms}, "
    "GAIN~\\citep{yoon2018gain}, Hi-VAE~\\citep{nazabal2020handling}, "
    "scGPT~\\citep{cui2024scgpt}, xTrimoGene~\\citep{gong2023xtrimogene}, "
    "Forest-Diffusion~\\citep{jolicoeur2024generating}, ReMDM~\\citep{wang2025remaskingdiscretediffusionmodels}, "
    "Blackout Diffusion~\\citep{santos2023blackout}"
    "Methods are grouped into three categories: naive baseline (top), scRNAseq/imputation-specific (middle), "
    "and general generative (bottom). Best performance in each category for each metric is bolded, "
    "and second best is italicized."
)


def cell(d, key, kind="plain"):
    if not d or key not in d:
        return "--"
    return fmt(d[key], kind)


# key -> (display kind, optimize direction). 'absmin' = closest to 0 (bias).
SPEC = {
    "r2": ("plain", "max"),
    "rmse": ("plain", "min"),
    "mae": ("plain", "min"),
    "raw_bias": ("plain", "absmin"),
    "spearman_corr": ("plain", "max"),
    "pearson_corr": ("plain", "max"),
    "energy_distance": ("plain", "min"),
    "scfid": ("logscfid", "min"),
    "mmd": ("mmd", "min"),
    "swd": ("plain", "min"),
}
LOG_KEYS = {"scfid", "mmd"}


def _disp_raw(d, key):
    """(displayed value rounded to 2dp, raw mean). Display is log-space for scFID/MMD."""
    a = np.array(d[key], dtype=float)
    a = a[~np.isnan(a)]
    if a.size == 0:
        return None, None
    raw = float(a.mean())
    v = float(np.log(np.clip(a, 1e-30, None)).mean()) if key in LOG_KEYS else raw
    return round(v, 2), raw


def tiers(rows, key):
    """(bold, italic) sets. Ties are by the *displayed* 2-decimal value: the best
    display value (and its ties) are bold; the next distinct display value (and its
    ties) are italic. Bias ranked by closeness to 0; overflow (>1e3) never bolded."""
    _, direction = SPEC[key]
    cand = []
    for label, d in rows:
        if not d or key not in d:
            continue
        dv, raw = _disp_raw(d, key)
        if dv is None or not np.isfinite(dv) or abs(raw) >= 1e3:
            continue
        cand.append((label, dv))
    if not cand:
        return set(), set()
    rank = (lambda v: abs(v)) if direction == "absmin" else (lambda v: v)
    pick = min if direction in ("min", "absmin") else max
    best = pick(rank(v) for _, v in cand)
    t1 = {l for l, v in cand if rank(v) == best}
    rem = [(l, v) for l, v in cand if l not in t1]
    t2 = set()
    if rem:
        best2 = pick(rank(v) for _, v in rem)
        t2 = {l for l, v in rem if rank(v) == best2}
    return t1, t2


def bcell(d, label, key, tmap):
    kind = SPEC[key][0]
    s = cell(d, key, kind)
    if s in ("--", "N/A") or "10^3" in s:
        return s
    t1, t2 = tmap.get(key, (set(), set()))
    if label in t1:
        return f"\\textbf{{{s}}}"
    if label in t2:
        return f"\\textit{{{s}}}"
    return s


def emit_main_latex(idx):
    """7-col grouped main table per task."""
    out = []
    titles = {("fetus", "MCAR", "0.5"): "human fetus cell atlas with 50\\% MCAR",
              ("fetus", "MNAR_low", "0.25"): "human fetus cell atlas with 25\\% low-biased missingness (MNAR)",
              ("heart", "MCAR", "0.5"): "heart with 50\\% MCAR"}
    labels = {("fetus", "MCAR", "0.5"): "tab:fetus_mcar",
              ("fetus", "MNAR_low", "0.25"): "tab: fetus_mnar",
              ("heart", "MCAR", "0.5"): "tab:heart_mcar"}
    for ds, mask, drop in TASKS:
        out.append("\\begin{table*}[!t]\\centering\\footnotesize\\setlength{\\tabcolsep}{4pt}")
        out.append(f"\\caption{{Benchmarking on scRNA-seq imputation, {titles[(ds,mask,drop)]}. Mean (standard error). {CITE_LIST}}}")
        out.append(f"\\label{{{labels[(ds,mask,drop)]}}}")
        out.append("\\begin{tabular}{l ccc c cccc}")
        out.append("\\toprule")
        out.append(" & \\multicolumn{3}{c}{\\textbf{Sample-level}} & & \\multicolumn{4}{c}{\\textbf{Distributional}} \\\\")
        out.append("\\cmidrule(lr){2-4}\\cmidrule(lr){6-9}")
        out.append("\\textbf{Method} & Spearman$\\uparrow$ & RMSE$\\downarrow$ & Bias & & ED$\\downarrow$ & log(scFID)$\\downarrow$ & log(MMD)$\\downarrow$ & SWD$\\downarrow$ \\\\")
        out.append("\\midrule")
        cols = ["spearman_corr", "rmse", "raw_bias", "energy_distance", "scfid", "mmd", "swd"]
        for gi, group in enumerate(GROUPS):
            rows_data = [(lbl, get(idx, m, ds, mask, drop, n)) for lbl, m, n in group]
            bolds = {k: tiers(rows_data, k) for k in cols}
            for (label, model, nimp), (_, d) in zip(group, rows_data):
                row = (f"{label} & {bcell(d,label,'spearman_corr',bolds)} & {bcell(d,label,'rmse',bolds)} & "
                       f"{bcell(d,label,'raw_bias',bolds)} & & {bcell(d,label,'energy_distance',bolds)} & "
                       f"{bcell(d,label,'scfid',bolds)} & {bcell(d,label,'mmd',bolds)} & {bcell(d,label,'swd',bolds)} \\\\")
                out.append(row)
            if gi < len(GROUPS) - 1:
                out.append("\\midrule")
        out.append("\\bottomrule\\end{tabular}\\end{table*}")
        out.append("")
    return "\n".join(out)


def emit_appendix_latex(idx):
    """Full-metric appendix table per task (10 metrics, grouped)."""
    out = []
    titles = {("fetus", "MCAR", "0.5"): "fetus, 50\\% MCAR",
              ("fetus", "MNAR_low", "0.25"): "fetus, 25\\% MNAR (low-biased)",
              ("heart", "MCAR", "0.5"): "heart, 50\\% MCAR"}
    labels = {("fetus", "MCAR", "0.5"): "tab:fetus_mcar_full",
              ("fetus", "MNAR_low", "0.25"): "tab:fetus_mnar_full",
              ("heart", "MCAR", "0.5"): "tab:heart_mcar_full"}
    for ds, mask, drop in TASKS:
        out.append("\\begin{table*}[!t]\\centering\\tiny\\setlength{\\tabcolsep}{3pt}")
        out.append(f"\\caption{{Full metrics ({titles[(ds,mask,drop)]}). Mean (standard error). {CITE_LIST}}}")
        out.append(f"\\label{{{labels[(ds,mask,drop)]}}}")
        out.append("\\begin{tabular}{l cccccc c cccc}")
        out.append("\\toprule")
        out.append(" & \\multicolumn{6}{c}{\\textbf{Sample-level}} & & \\multicolumn{4}{c}{\\textbf{Distributional}} \\\\")
        out.append("\\cmidrule(lr){2-7}\\cmidrule(lr){9-12}")
        out.append("\\textbf{Method} & R$^2\\uparrow$ & RMSE$\\downarrow$ & MAE$\\downarrow$ & Bias & Spearman$\\uparrow$ & Pearson$\\uparrow$ & & ED$\\downarrow$ & log(scFID)$\\downarrow$ & log(MMD)$\\downarrow$ & SWD$\\downarrow$ \\\\")
        out.append("\\midrule")
        cols = ["r2", "rmse", "mae", "raw_bias", "spearman_corr", "pearson_corr",
                "energy_distance", "scfid", "mmd", "swd"]
        for gi, group in enumerate(GROUPS):
            rows_data = [(lbl, get(idx, m, ds, mask, drop, n)) for lbl, m, n in group]
            bolds = {k: tiers(rows_data, k) for k in cols}
            for (label, model, nimp), (_, d) in zip(group, rows_data):
                row = (f"{label} & {bcell(d,label,'r2',bolds)} & {bcell(d,label,'rmse',bolds)} & {bcell(d,label,'mae',bolds)} & "
                       f"{bcell(d,label,'raw_bias',bolds)} & {bcell(d,label,'spearman_corr',bolds)} & {bcell(d,label,'pearson_corr',bolds)} & & "
                       f"{bcell(d,label,'energy_distance',bolds)} & {bcell(d,label,'scfid',bolds)} & {bcell(d,label,'mmd',bolds)} & {bcell(d,label,'swd',bolds)} \\\\")
                out.append(row)
            if gi < len(GROUPS) - 1:
                out.append("\\midrule")
        out.append("\\bottomrule\\end{tabular}\\end{table*}")
        out.append("")
    return "\n".join(out)


def main():
    idx = load_index()
    tex = emit_main_latex(idx)
    with open("data/dnadiff/imputed_data/_forest_overnight/paper_tables_main.tex", "w") as f:
        f.write(tex)
    apx = emit_appendix_latex(idx)
    with open("data/dnadiff/imputed_data/_forest_overnight/paper_tables_appendix.tex", "w") as f:
        f.write(apx)
    print("WROTE paper_tables_main.tex + paper_tables_appendix.tex\n")
    for ds, mask, drop in TASKS:
        print(f"\n{'='*90}\nTASK: {ds} {mask} {drop}\n{'='*90}")
        # availability
        print("AVAILABILITY:")
        for label, model, nimp in ROWS:
            d = get(idx, model, ds, mask, drop, nimp)
            print(f"  {label:24s} {'FILLED' if d else 'blank'}")
        # appendix table rows (all metrics)
        print("\nAPPENDIX (all metrics) — mean(SE):")
        hdr = ["Method", "r2", "RMSE", "MAE", "Bias", "Spear", "Pears", "ED", "log scFID", "MMD", "SWD"]
        print("  " + " | ".join(hdr))
        for label, model, nimp in ROWS:
            d = get(idx, model, ds, mask, drop, nimp)
            if not d:
                print(f"  {label}: --")
                continue
            row = [label, fmt(d["r2"]), fmt(d["rmse"]), fmt(d["mae"]), fmt(d["raw_bias"]),
                   fmt(d["spearman_corr"]), fmt(d["pearson_corr"]), fmt(d["energy_distance"]),
                   fmt(d["scfid"], "logscfid"), fmt(d["mmd"], "mmd"), fmt(d["swd"])]
            print("  " + " | ".join(row))


if __name__ == "__main__":
    main()
