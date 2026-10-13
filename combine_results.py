import ast
from datetime import datetime
from itertools import zip_longest
from pathlib import Path

import numpy as np
import pandas as pd

#-----------------------------------------------------------------------------------------------------
#Combines the per-sample PFAScreen results tables from one batch into a single table.
#Features seen in more than one sample (or twice in one sample) collapse to one row.
#
#Runs on its own too: `python combine_results.py` asks for the results files and an output folder.

#a feature is a duplicate of the group's anchor when BOTH differences are under these
MZ_DIFF = 0.005  #Da
RT_DIFF = 0.3    #min, compared on PFAScreen's rt(min) column

MZ_COL = "mz"
RT_COL = "rt(min)"
AREA_COL = "mz_area"  #picks the anchor - the most intense instance of a feature
SAMPLE_COL = "sample"
N_SAMPLES_COL = "n_samples"
#PFAScreen stores several suspect candidates per feature as a stringified python list, with
#position N in each of these columns describing the same compound
CANDIDATE_COLS = ["compound_names", "formulas", "SMILES"]

SHEET_COMBINED = "Combined"
SHEET_ALL = "All features"

#-----------------------------------------------------------------------------------------------------


#PFAScreen names its folders/files PFAScreen_<sample> and Results_PFAScreen_<sample>, so the
#sample name is whatever follows that prefix
def sample_name_from_path(path):
    stem = Path(path).stem
    for prefix in ("Results_PFAScreen_", "PFAScreen_"):
        if stem.startswith(prefix):
            return stem[len(prefix):]
    return stem


def read_results(path):
    path = Path(path)
    frame = pd.read_excel(path) if path.suffix.lower() in (".xlsx", ".xls") else pd.read_csv(path)
    #PFAScreen writes its feature index without a header, so it reads back as 'Unnamed: 0'
    first = frame.columns[0]
    if str(first).startswith("Unnamed:") or str(first) == "Index":
        frame = frame.rename(columns={first: "id"})
    missing = [c for c in [MZ_COL, RT_COL, AREA_COL] + CANDIDATE_COLS if c not in frame.columns]
    if missing:
        raise KeyError(f"{path.name} is missing the column(s) {missing}. Found: {list(frame.columns)}")
    return frame


#PFAScreen's Python 3.9 environment ships xlsxwriter and an openpyxl too old for pandas 2.2;
#other environments may only have openpyxl
def excel_engine():
    try:
        import xlsxwriter  # noqa: F401
        return "xlsxwriter"
    except ImportError:
        return "openpyxl"


#'['a', 'b']' -> ['a', 'b'];  blank -> [];  a bare value -> [value]
def as_list(cell):
    if isinstance(cell, list):
        return cell
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return []
    text = str(cell).strip()
    if not text or text == "[]":
        return []
    if text.startswith("["):
        try:
            parsed = ast.literal_eval(text)
            return list(parsed) if isinstance(parsed, (list, tuple)) else [parsed]
        except (ValueError, SyntaxError):
            pass
    return [text]


#merges the candidates of every feature in a group into one list per column. Candidates are
#handled as (name, formula, SMILES) sets so the three columns stay positionally paired, and an
#exact repeat of a set already listed is dropped. The anchor's candidates come first.
def merge_candidates(group):
    seen = []
    for _, row in group.iterrows():
        columns = [as_list(row[c]) for c in CANDIDATE_COLS]
        for candidate in zip_longest(*columns):
            if candidate not in seen:
                seen.append(candidate)
    if not seen:
        return {c: np.nan for c in CANDIDATE_COLS}
    return {c: str([cand[i] for cand in seen]) for i, c in enumerate(CANDIDATE_COLS)}


#anchor grouping: the most intense feature not yet placed takes every unplaced feature inside
#both windows of it, then the next most intense unplaced feature does the same. Every member of
#a group is therefore within the window of the row that's kept.
def group_duplicates(stacked, mz_diff=MZ_DIFF, rt_diff=RT_DIFF):
    order = stacked[AREA_COL].fillna(-np.inf).to_numpy().argsort(kind="stable")[::-1]
    mz = stacked[MZ_COL].to_numpy(dtype=float)
    rt = stacked[RT_COL].to_numpy(dtype=float)
    group_of = np.full(len(stacked), -1)
    #m/z values carry float noise (100.005 - 100.0 = 0.0049999999999955), which would let a pair
    #sitting exactly on the window through; shaving 1e-9 keeps the comparison strictly 'under'
    mz_diff, rt_diff = mz_diff - 1e-9, rt_diff - 1e-9

    group = 0
    for anchor in order:
        if group_of[anchor] != -1:
            continue
        #NaN comparisons are False, so a feature with no m/z or RT only ever groups with itself
        members = (group_of == -1) & (np.abs(mz - mz[anchor]) < mz_diff) & (np.abs(rt - rt[anchor]) < rt_diff)
        members[anchor] = True
        group_of[members] = group
        group += 1
    return group_of


def combine_frames(frames, mz_diff=MZ_DIFF, rt_diff=RT_DIFF):
    #frames: {sample name: results table}, in batch order. Returns (combined, stacked)
    stacked = pd.concat([f.assign(**{SAMPLE_COL: name}) for name, f in frames.items()],
                        ignore_index=True)
    #sample column up front, so the stacked sheet reads sample-by-sample
    stacked = stacked[[SAMPLE_COL] + [c for c in stacked.columns if c != SAMPLE_COL]]
    for c in (MZ_COL, RT_COL, AREA_COL):
        stacked[c] = pd.to_numeric(stacked[c], errors="coerce")

    stacked["_group"] = group_duplicates(stacked, mz_diff, rt_diff)
    batch_order = {name: i for i, name in enumerate(frames)}

    rows = []
    for _, group in stacked.groupby("_group", sort=False):
        #anchor first: every numeric value on the kept row comes from the most intense instance
        group = group.sort_values(AREA_COL, ascending=False, na_position="last", kind="stable")
        row = group.iloc[0].copy()
        #anchor's sample first, the rest in batch order; a sample is listed once even when
        #two of its own features collapsed together
        others = sorted(set(group[SAMPLE_COL]) - {row[SAMPLE_COL]}, key=batch_order.get)
        row[SAMPLE_COL] = "; ".join([row[SAMPLE_COL]] + others)
        row[N_SAMPLES_COL] = 1 + len(others)
        for c, value in merge_candidates(group).items():
            row[c] = value
        rows.append(row)

    combined = pd.DataFrame(rows).drop(columns="_group")
    combined = combined[[SAMPLE_COL, N_SAMPLES_COL] +
                        [c for c in combined.columns if c not in (SAMPLE_COL, N_SAMPLES_COL)]]
    #PFAScreen sorts its own tables by m/C, most PFAS-like first
    if "m/C" in combined.columns:
        combined = combined.sort_values("m/C", ascending=False, kind="stable")
    combined = combined.reset_index(drop=True)
    return combined, stacked.drop(columns="_group")


def combine_results(result_files, out_dir, mz_diff=MZ_DIFF, rt_diff=RT_DIFF, name="PFAScreen_batch"):
    #result_files: list of Results_PFAScreen_<sample>.csv/.xlsx paths, or {sample name: path}.
    #Writes <out_dir>/<timestamp>_<name>_combined.xlsx (+ .csv of the combined sheet) and returns
    #the .xlsx path.
    if not isinstance(result_files, dict):
        result_files = {sample_name_from_path(p): p for p in result_files}
    if not result_files:
        raise ValueError("No PFAScreen results files to combine.")

    frames = {sample: read_results(path) for sample, path in result_files.items()}
    combined, stacked = combine_frames(frames, mz_diff, rt_diff)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
    out_xlsx = out_dir / f"{timestamp}_{name}_combined.xlsx"

    with pd.ExcelWriter(out_xlsx, engine=excel_engine()) as xl:
        combined.to_excel(xl, sheet_name=SHEET_COMBINED, index=False)
        stacked.to_excel(xl, sheet_name=SHEET_ALL, index=False)
    combined.to_csv(out_xlsx.with_suffix(".csv"), index=False)

    print(f"Combined {len(stacked)} features from {len(frames)} samples into {len(combined)} rows "
          f"({(combined[N_SAMPLES_COL] > 1).sum()} seen in more than one sample) "
          f"-> {out_xlsx}")
    return out_xlsx


#----------------------
if __name__ == "__main__":
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    try:
        picked = filedialog.askopenfilenames(
            title="Select the PFAScreen results files to combine",
            filetypes=[("PFAScreen results", "Results_PFAScreen_*.csv Results_PFAScreen_*.xlsx"),
                       ("Table files", "*.csv *.xlsx"), ("All files", "*.*")])
        if not picked:
            raise SystemExit("No results files selected.")
        out_dir = filedialog.askdirectory(title="Select where to save the combined file")
        if not out_dir:
            raise SystemExit("No output folder selected.")
        combine_results(list(picked), out_dir)
    finally:
        root.destroy()
