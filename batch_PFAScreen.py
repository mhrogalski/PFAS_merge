import os
import shutil
import sys
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import combine_results

#-----------------------------------------------------------------------------------------------------
#Batch wrapper for PFAScreen: runs FeatureFinding + PFASPrioritization on every selected sample
#.mzML with one set of settings, then combines the per-sample results into one table.
#
#NOTE: PFAScreen needs Python 3.9 (pyopenms 3.1.0) - start this with Run_PFAScreen_batch.bat or
#`py -3.9 batch_PFAScreen.py`. PFAScreen-main itself is only imported, never modified.

#where PFAScreen-main lives; can also be changed in the GUI
PFASCREEN_DIR = Path.home() / "Documents" / "PFAScreen-main"

#put sample and blank .mzML files in Input - the file pickers open there. Each batch run is
#exported to its own timestamped folder inside Output (changeable in the GUI).
#kept relative to this script so the paths survive the project folder being moved or renamed
PROJECT_DIR = Path(__file__).resolve().parent
INPUT_DIR = PROJECT_DIR / "Input"
OUTPUT_DIR = PROJECT_DIR / "Output"

#PFAScreen reads these by bare file name, i.e. from the working directory. The batch runs inside
#its output folder, so a copy of each goes there (and doubles as a record of the lists used).
PFASCREEN_DATA_FILES = ["suspect_list.csv", "diagnostic_fragments.csv"]

#PFAScreen's own GUI defaults (run_PFAScreen.py), plus the duplicate windows for combining.
#Space-separated entries are split into lists the same way PFAScreen's GUI does it.
DEFAULT_SETTINGS = {
    #FeatureFinding
    "mass_error_ppm": 10,
    "intensity_threshold": 1000,
    "isotope_model": True,
    "mass_tolerance_MSMS_assignment": 0.005,  #Da
    "RT_tolerance_MSMS_assignment": 12.0,     #s
    "mass_tolerance_blank_correction": 0.002, #Da
    "RT_tolerance_blank_correction": 6.0,     #s
    "fold_change_blank_correction": 5,
    #PFASPrioritization
    "mC_range": "0 inf",
    "MDC_range": "-0.5 0.5",
    "MD_range": "-0.5 0.5",
    "diffs_KMD": "CF2",
    "mass_tolerance_KMD": 0.002,
    "n_homologues": 3,
    "diffs": "CF2 C2F4 HF",
    "number_of_fragments": 1,
    "mass_tolerance_FindPFAS": 0.002,
    "intensity_threshold_MSMS": 1000,
    "save_MSMS_spectra": False,
    "adducts": 1,  #1 = M-H, 2 = M+H, 3 = M+
    "tol_suspect": 0.002,
    #combining
    "combine_mz_diff": combine_results.MZ_DIFF,
    "combine_rt_diff": combine_results.RT_DIFF,
}

#-----------------------------------------------------------------------------------------------------


def import_pfascreen(pfascreen_dir):
    #PFAScreen's modules import each other by bare name, so its folder goes on sys.path
    pfascreen_dir = Path(pfascreen_dir)
    if not (pfascreen_dir / "PFAS_feature_prioritization.py").exists():
        raise FileNotFoundError(f"{pfascreen_dir} doesn't look like PFAScreen-main "
                                f"(no PFAS_feature_prioritization.py).")
    if str(pfascreen_dir) not in sys.path:
        sys.path.insert(0, str(pfascreen_dir))
    global MS1_feature_finding, read_feature_map, blank_correc, ismembertol, plotting
    global PFAS_feature_prioritization
    from MS1_feature_finding import MS1_feature_finding
    from read_feature_map import read_feature_map
    from blank_correc import blank_correc
    from ismembertol import ismembertol
    import plotting
    from PFAS_feature_prioritization import PFAS_feature_prioritization


def isotope_model_name(isotope_model_bool):
    return "metabolites (5% RMS)" if isotope_model_bool else "none"


def find_blank_features(blank_path, s):
    #feature finding on a blank, done once per batch and reused for every sample
    _, feature_map = MS1_feature_finding(str(blank_path), float(s["mass_error_ppm"]),
                                         float(s["intensity_threshold"]),
                                         isotope_model_name(s["isotope_model"]))
    return read_feature_map(feature_map)


def getMS2RawData(exp):
    #copied from PFAScreen's feature_preproccessing so the batch gives the same MS2 table
    precursor_mz, precursor_RT, precursor_intens, MS2_spec_mz, MS2_spec_intens = [], [], [], [], []
    for spec in exp:
        if spec.getMSLevel() == 2:
            precursor_mz.append(spec.getPrecursors()[0].getMZ())
            precursor_RT.append(spec.getRT())
            precursor_intens.append(spec.getPrecursors()[0].getIntensity())
            MS2_spec_mz.append(spec.get_peaks()[0])
            MS2_spec_intens.append(spec.get_peaks()[1])
    return (np.array(precursor_mz), np.array(precursor_RT), np.array(precursor_intens),
            MS2_spec_mz, MS2_spec_intens)


def feature_preprocessing_multi_blank(sample_path, blank_features, s, Results_folder):
    #PFAScreen's feature_preproccessing, except the blank correction runs against every blank:
    #a sample feature is dropped if it matches a feature in ANY blank, each blank judged with
    #PFAScreen's own m/z + RT tolerance and fold-change rule. Returns what PFAScreen's version
    #returns (minus the raw experiments, which the batch doesn't need).
    exp, feature_map = MS1_feature_finding(str(sample_path), float(s["mass_error_ppm"]),
                                           float(s["intensity_threshold"]),
                                           isotope_model_name(s["isotope_model"]))
    #made only once the file has loaded, so a sample that fails here leaves no empty folder behind
    os.makedirs(os.path.join(Results_folder, "plots"), exist_ok=True)
    Df_FeatureData = read_feature_map(feature_map)
    print(f"{len(Df_FeatureData)} features detected")

    if blank_features:
        keep = np.ones(len(Df_FeatureData), dtype=bool)
        for blank_name, Df_Blank in blank_features.items():
            idx_not_in_blank, _, _ = blank_correc(
                Df_FeatureData["mz"], Df_FeatureData["rt"], Df_FeatureData["mz_area"],
                Df_Blank["mz"], Df_Blank["rt"], Df_Blank["mz_area"],
                float(s["mass_tolerance_blank_correction"]),
                float(s["RT_tolerance_blank_correction"]),
                float(s["fold_change_blank_correction"]))
            print(f"  {(~idx_not_in_blank & keep).sum()} removed by blank {blank_name}")
            keep &= np.asarray(idx_not_in_blank, dtype=bool)
        Df_FeatureData = Df_FeatureData[keep].reset_index(drop=True)
        print(f"{len(Df_FeatureData)} features after blank correction")

    precursor_mz, precursor_RT, precursor_intens, MS2_spec_mz, MS2_spec_intens = getMS2RawData(exp)
    Df_MS2RawData = pd.DataFrame(data={"mz": precursor_mz, "rt": precursor_RT,
                                       "intens": precursor_intens, "ms2_spec_mz": MS2_spec_mz,
                                       "ms2_spec_intens": MS2_spec_intens})

    #MS2 spectra to MS1 features, as in PFAScreen
    _, _, M_bool_mz = ismembertol(Df_FeatureData["mz"], Df_MS2RawData["mz"],
                                  float(s["mass_tolerance_MSMS_assignment"]))
    _, _, M_bool_RT = ismembertol(Df_FeatureData["rt"], Df_MS2RawData["rt"],
                                  float(s["RT_tolerance_MSMS_assignment"]))
    M_combined = np.logical_and(M_bool_mz, M_bool_RT)
    idx_in_MS2RawData = np.where(M_combined)[0]
    idx_in_features = np.where(M_combined)[1]
    print(f"{len(idx_in_MS2RawData)} of {len(Df_MS2RawData)} MS2 spectra assigned to features")
    print(f"{len(np.unique(idx_in_features))} of {len(Df_FeatureData)} features have MS2 information")

    plotting.mz_RT_MSMS(Df_FeatureData, Df_MS2RawData, idx_in_features, idx_in_MS2RawData,
                        Results_folder)

    Df_MS2RawData["idx_ms1"] = np.nan
    Df_MS2RawData.loc[idx_in_MS2RawData, "idx_ms1"] = idx_in_features
    Df_MS2RawData = Df_MS2RawData.dropna(subset=["idx_ms1"])
    Df_MS2RawData = Df_MS2RawData.sort_values("intens").drop_duplicates("idx_ms1").sort_index()

    return Df_FeatureData, Df_MS2RawData, idx_in_features


def run_sample(sample_path, blank_features, s):
    #one sample through both PFAScreen stages; returns the results .csv it wrote
    Results_folder = f"PFAScreen_{Path(sample_path).stem}"
    Df_FeatureData, Df_MS2RawData, idx_in_features = feature_preprocessing_multi_blank(
        sample_path, blank_features, s, Results_folder)

    PFAS_feature_prioritization(
        Df_FeatureData, Df_MS2RawData, idx_in_features,
        Results_folder=Results_folder,
        diffs=str(s["diffs"]).split(),
        number_of_fragments=float(s["number_of_fragments"]),
        mass_tolerance_FindPFAS=float(s["mass_tolerance_FindPFAS"]),
        intensity_threshold=float(s["intensity_threshold_MSMS"]),
        diffs_KMD=str(s["diffs_KMD"]).split(),
        mass_tolerance_KMD=float(s["mass_tolerance_KMD"]),
        n_homologues=float(s["n_homologues"]),
        adducts=int(s["adducts"]),
        tol_suspect=float(s["tol_suspect"]),
        save_MSMS_spectra=bool(s["save_MSMS_spectra"]),
        mC_range=str(s["mC_range"]).split(),
        MDC_range=str(s["MDC_range"]).split(),
        MD_range=str(s["MD_range"]).split())

    return Path(Results_folder) / f"Results_{Results_folder}.csv"


def write_log(batch_dir, samples, blanks, s, results, failures):
    lines = [f"PFAScreen batch run {datetime.now():%Y-%m-%d %H:%M}", "",
             "Samples:"] + [f"  {p}" for p in samples] + \
            ["", "Blanks:"] + ([f"  {p}" for p in blanks] or ["  (none)"]) + \
            ["", "Settings:"] + [f"  {k} = {v}" for k, v in s.items()] + \
            ["", f"Succeeded: {len(results)} of {len(samples)}"] + \
            [f"  FAILED {name}: {err}" for name, err in failures.items()]
    (batch_dir / "batch_log.txt").write_text("\n".join(lines), encoding="utf-8")


def run_batch(samples, blanks, out_parent, settings=None, pfascreen_dir=PFASCREEN_DIR, status=print):
    #samples/blanks: lists of .mzML paths. Creates <out_parent>/PFAScreen_batch_<timestamp>/
    #holding each sample's normal PFAScreen results folder plus the combined table.
    #Returns (combined .xlsx path or None, {sample: error} for samples that failed).
    s = dict(DEFAULT_SETTINGS, **(settings or {}))
    samples = [Path(p) for p in samples]
    blanks = [Path(p) for p in blanks]

    if not samples:
        raise ValueError("No sample .mzML files selected.")
    #checked up front so a typo'd path fails in seconds, not after an hour of feature finding
    missing = [str(p) for p in samples + blanks if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"These files don't exist: {missing}")
    #PFAScreen names each results folder after the file, so two samples with the same name
    #would write into one folder
    stems = [p.stem for p in samples]
    repeated = sorted({x for x in stems if stems.count(x) > 1})
    if repeated:
        raise ValueError(f"Samples share a file name, so their results would overwrite each other: {repeated}")
    overlap = {p.resolve() for p in samples} & {p.resolve() for p in blanks}
    if overlap:
        raise ValueError(f"These files were picked as both sample and blank: {sorted(map(str, overlap))}")

    pfascreen_dir = Path(pfascreen_dir)
    import_pfascreen(pfascreen_dir)

    batch_dir = Path(out_parent) / f"PFAScreen_batch_{datetime.now():%Y_%m_%d_%H_%M_%S}"
    batch_dir.mkdir(parents=True)
    for name in PFASCREEN_DATA_FILES:
        shutil.copy2(pfascreen_dir / name, batch_dir / name)

    results, failures = {}, {}
    start_dir = os.getcwd()
    os.chdir(batch_dir)
    try:
        blank_features = {}
        for i, blank in enumerate(blanks, 1):
            status(f"Blank {i}/{len(blanks)}: feature finding on {blank.name}")
            blank_features[blank.stem] = find_blank_features(blank, s)

        for i, sample in enumerate(samples, 1):
            status(f"Sample {i}/{len(samples)}: {sample.name}")
            #one bad file shouldn't sink the rest of the batch
            try:
                results[sample.stem] = (batch_dir / run_sample(sample, blank_features, s)).resolve()
            except Exception as err:
                traceback.print_exc()
                failures[sample.stem] = f"{type(err).__name__}: {err}"
                status(f"Sample {sample.name} FAILED - continuing with the rest")
    finally:
        os.chdir(start_dir)
        write_log(batch_dir, samples, blanks, s, results, failures)

    combined = None
    if results:
        status("Combining results")
        combined = combine_results.combine_results(results, batch_dir,
                                                   mz_diff=float(s["combine_mz_diff"]),
                                                   rt_diff=float(s["combine_rt_diff"]))
    status(f"Done: {len(results)} of {len(samples)} samples processed"
           + (f", {len(failures)} failed (see batch_log.txt)" if failures else "")
           + f". Output in {batch_dir}")
    return combined, failures


#-----------------------------------------------------------------------------------------------------
#GUI

#(settings key, label) per section; a key holding a bool becomes a checkbox
GUI_SECTIONS = [
    ("Peak finding", [("mass_error_ppm", "Mass error (ppm)"),
                      ("intensity_threshold", "Intensity threshold"),
                      ("isotope_model", "Isotope model")]),
    ("MS/MS alignment", [("mass_tolerance_MSMS_assignment", "Mass tolerance (Da)"),
                         ("RT_tolerance_MSMS_assignment", "RT tolerance (s)")]),
    ("Blank correction", [("mass_tolerance_blank_correction", "Mass tolerance (Da)"),
                          ("RT_tolerance_blank_correction", "RT tolerance (s)"),
                          ("fold_change_blank_correction", "Fold change")]),
    ("MD/C-m/C & MD filtering", [("mC_range", "m/C range"),
                                 ("MDC_range", "MD/C range"),
                                 ("MD_range", "MD range")]),
    ("Kendrick mass defect", [("diffs_KMD", "KMD difference"),
                              ("mass_tolerance_KMD", "KMD mass tolerance (Da)"),
                              ("n_homologues", "Number of homologues")]),
    ("MS2 differences and DFs", [("diffs", "Mass differences"),
                                 ("number_of_fragments", "Number of fragments"),
                                 ("mass_tolerance_FindPFAS", "Mass tolerance frags (Da)"),
                                 ("intensity_threshold_MSMS", "Intensity threshold MS/MS"),
                                 ("save_MSMS_spectra", "Save HTML MS/MS spectra")]),
    ("Suspect screening", [("adducts", "Adduct"),
                           ("tol_suspect", "Suspect mass tol (Da)")]),
    ("Combining duplicates", [("combine_mz_diff", "Δm/z under (Da)"),
                              ("combine_rt_diff", "ΔRT under (min)")]),
]
ADDUCTS = {1: "M-H", 2: "M+H", 3: "M+"}


def launch_gui():
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox

    #created on first launch so a fresh copy of the folder works without setup
    INPUT_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    root = tk.Tk()
    root.title("PFΔScreen batch")

    files = ttk.Frame(root, padding=10)
    files.grid(row=0, column=0, sticky="nsew")
    settings_frame = ttk.Frame(root, padding=10)
    settings_frame.grid(row=0, column=1, sticky="nsew")

    #---- file lists
    def file_list(parent, row, title):
        ttk.Label(parent, text=title, font=("TkDefaultFont", 10, "bold")).grid(row=row, column=0, sticky="w")
        box = tk.Listbox(parent, width=60, height=8, selectmode="extended")
        box.grid(row=row + 1, column=0, sticky="nsew")
        buttons = ttk.Frame(parent)
        buttons.grid(row=row + 1, column=1, sticky="n", padx=(6, 0))

        def add():
            for p in filedialog.askopenfilenames(title=f"Select {title.lower()}", initialdir=INPUT_DIR,
                                                 filetypes=[("mzML files", "*.mzML *.mzml"), ("All files", "*.*")]):
                if p not in box.get(0, "end"):
                    box.insert("end", p)

        def remove():
            for i in reversed(box.curselection()):
                box.delete(i)

        ttk.Button(buttons, text="Add...", command=add).pack(fill="x")
        ttk.Button(buttons, text="Remove", command=remove).pack(fill="x", pady=2)
        ttk.Button(buttons, text="Clear", command=lambda: box.delete(0, "end")).pack(fill="x")
        return box

    sample_box = file_list(files, 0, "Sample .mzML files")
    ttk.Label(files, text="").grid(row=2, column=0)
    blank_box = file_list(files, 3, "Blank .mzML files (optional - features in ANY blank are removed)")

    def folder_row(row, title, initial):
        ttk.Label(files, text=title, font=("TkDefaultFont", 10, "bold")).grid(row=row, column=0, sticky="w", pady=(12, 0))
        var = tk.StringVar(value=str(initial))
        ttk.Entry(files, textvariable=var, width=60).grid(row=row + 1, column=0, sticky="ew")
        ttk.Button(files, text="Browse...",
                   command=lambda: var.set(filedialog.askdirectory(title=title) or var.get())
                   ).grid(row=row + 1, column=1, padx=(6, 0))
        return var

    out_var = folder_row(6, "Output folder (a timestamped batch folder is made inside)",
                         OUTPUT_DIR)
    pfas_var = folder_row(8, "PFAScreen-main folder", PFASCREEN_DIR)

    #---- settings, pre-filled with PFAScreen's defaults
    vars_ = {}
    row = 0
    for section, fields in GUI_SECTIONS:
        ttk.Label(settings_frame, text=section, font=("TkDefaultFont", 10, "bold")).grid(
            row=row, column=0, columnspan=2, sticky="w", pady=(8 if row else 0, 2))
        row += 1
        for key, label in fields:
            default = DEFAULT_SETTINGS[key]
            if key == "adducts":
                var = tk.StringVar(value=ADDUCTS[default])
                widget = ttk.Combobox(settings_frame, textvariable=var, values=list(ADDUCTS.values()),
                                      state="readonly", width=10)
            elif isinstance(default, bool):
                var = tk.BooleanVar(value=default)
                widget = ttk.Checkbutton(settings_frame, variable=var)
            else:
                var = tk.StringVar(value=str(default))
                widget = ttk.Entry(settings_frame, textvariable=var, width=13)
            ttk.Label(settings_frame, text=label).grid(row=row, column=0, sticky="w", padx=(10, 6))
            widget.grid(row=row, column=1, sticky="w")
            vars_[key] = var
            row += 1

    status_var = tk.StringVar(value="Add samples, then Run batch.")
    ttk.Label(root, textvariable=status_var, padding=(10, 0)).grid(row=1, column=0, columnspan=2, sticky="w")

    def status(message):
        print(message)
        status_var.set(message)
        root.update()

    def read_settings():
        s = {}
        for key, var in vars_.items():
            value = var.get()
            if key == "adducts":
                value = {v: k for k, v in ADDUCTS.items()}[value]
            elif isinstance(DEFAULT_SETTINGS[key], (int, float)) and not isinstance(DEFAULT_SETTINGS[key], bool):
                try:
                    value = float(value)
                except ValueError:
                    raise ValueError(f"'{value}' isn't a number (setting: {key})")
            s[key] = value
        return s

    def on_run():
        try:
            s = read_settings()
            run_button.config(state="disabled")
            combined, failures = run_batch(sample_box.get(0, "end"), blank_box.get(0, "end"),
                                           out_var.get(), s, pfas_var.get(), status)
        except Exception as err:
            traceback.print_exc()
            messagebox.showerror("Batch failed", str(err))
            status("Batch failed - see the console.")
            return
        finally:
            run_button.config(state="normal")
        if combined is None:
            messagebox.showerror("Batch finished", "Every sample failed - see batch_log.txt and the console.")
        elif failures:
            messagebox.showwarning("Batch finished",
                                   f"{len(failures)} sample(s) failed: {', '.join(failures)}\n\n"
                                   f"Combined file (from the rest):\n{combined}")
        else:
            messagebox.showinfo("Batch finished", f"Combined file:\n{combined}")

    run_button = tk.Button(root, text="Run batch", bg="#F7EFEA", font=("TkDefaultFont", 10, "bold"),
                           command=on_run)
    run_button.grid(row=2, column=0, columnspan=2, pady=10, ipadx=20, ipady=6)

    root.mainloop()


if __name__ == "__main__":
    launch_gui()
