"""Process every child with resting-state EEG in one HBN release.

Saves standard features and LaBraM embeddings (pretrained and random weights)
per child, deleting the raw recording afterwards to save disk.
Usage: python run_release.py R1 /path/to/data_dir /path/to/out_dir [max_children]
"""
import os
import sys
import time

import numpy as np
import pandas as pd

import hbn_pipeline as hp

release, data_dir, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
limit = int(sys.argv[4]) if len(sys.argv) > 4 else None
os.makedirs(out_dir, exist_ok=True)

parts = hp.get_participants(release, data_dir)
parts = parts[parts["RestingState"] == "available"]
if limit:
    parts = parts.head(limit)

pretrained, random_init = hp.load_labram(), hp.load_labram(random_weights=True)
log = []
for i, sub in enumerate(parts["participant_id"]):
    out = os.path.join(out_dir, f"{sub}.npz")
    if os.path.exists(out):
        continue
    t = time.time()
    try:
        paths = hp.download_resting_state(release, sub, data_dir)
        windows, n_total = hp.eyes_closed_windows(paths["eeg.set"], paths["events.tsv"])
        if len(windows) < 2:
            raise ValueError(f"only {len(windows)} clean windows of {n_total}")
        feats = hp.standard_features(windows)
        np.savez(out,
                 feat_names=np.array(list(feats)), feats=np.array(list(feats.values()), float),
                 labram=hp.labram_embeddings(pretrained, windows),
                 labram_random=hp.labram_embeddings(random_init, windows),
                 n_windows=len(windows), n_total=n_total)
        status = f"ok {len(windows)}/{n_total} windows"
    except Exception as e:  # keep going, record the failure
        status = f"FAILED {e!r}"[:200]
    finally:
        for suffix in ("eeg.set", "events.tsv"):
            p = os.path.join(data_dir, f"{sub}_RestingState_{suffix}")
            if os.path.exists(p):
                os.remove(p)
    log.append((sub, status))
    print(f"[{i + 1}/{len(parts)}] {sub} {status} ({time.time() - t:.0f}s)", flush=True)

pd.DataFrame(log, columns=["participant_id", "status"]).to_csv(
    os.path.join(out_dir, f"log_{release}.tsv"), sep="\t", index=False, mode="a")
