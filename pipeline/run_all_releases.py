"""Reprocess HBN releases 1-3 in the cloud workspace: standard features, full log spectrum,
and pretrained LaBraM embeddings per child. Skips children already saved.
Usage: python run_all_here.py OUT_DIR DATA_DIR [max_children]
"""
import os, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np, pandas as pd, torch
from scipy.signal import welch
import hbn_pipeline as hp

out_dir, data_dir = sys.argv[1], sys.argv[2]
limit = int(sys.argv[3]) if len(sys.argv) > 3 else None
os.makedirs(out_dir, exist_ok=True)
torch.set_num_threads(2)

subs = []
for rel in ('R1', 'R2', 'R3'):
    p = hp.get_participants(rel, data_dir)
    p = p[p['RestingState'] == 'available']
    subs += [(rel, s) for s in p['participant_id']]
seen = set(); subs = [x for x in subs if not (x[1] in seen or seen.add(x[1]))]
todo = [x for x in subs if not os.path.exists(os.path.join(out_dir, f'{x[1]}.npz'))]
if limit:
    todo = todo[:limit]
print(f'{len(subs)} children with resting EEG, {len(todo)} to process', flush=True)

def prepare(rel, sub):
    """Download, clean, window, standard features, full spectrum (I/O heavy, runs in threads)."""
    try:
        paths = hp.download_resting_state(rel, sub, data_dir)
        windows, n_total = hp.eyes_closed_windows(paths['eeg.set'], paths['events.tsv'])
        if len(windows) < 2:
            return sub, None, f'only {len(windows)} clean windows of {n_total}'
        feats = hp.standard_features(windows)
        freqs, psd = welch(windows, fs=hp.SFREQ, nperseg=2 * hp.SFREQ, axis=-1)
        return sub, dict(windows=windows, n_total=n_total, feats=feats, freqs=freqs,
                         log_psd=np.log10(psd.mean(0) + 1e-30)), 'ok'
    except Exception as e:
        return sub, None, f'FAILED {e!r}'[:150]
    finally:
        for suffix in ('eeg.set', 'events.tsv'):
            f = os.path.join(data_dir, f'{sub}_RestingState_{suffix}')
            if os.path.exists(f):
                os.remove(f)

model = hp.load_labram()
t0, done = time.time(), 0
log = open(os.path.join(out_dir, 'log.tsv'), 'a')
with ThreadPoolExecutor(max_workers=3) as pool:
    for fut in as_completed([pool.submit(prepare, r, s) for r, s in todo]):
        sub, d, status = fut.result(); done += 1
        if d is not None:
            np.savez(os.path.join(out_dir, f'{sub}.npz'),
                     feat_names=np.array(list(d['feats'])), feats=np.array(list(d['feats'].values()), float),
                     labram=hp.labram_embeddings(model, d['windows']).astype('float32'),
                     freqs=d['freqs'].astype('float32'), log_psd=d['log_psd'].astype('float32'),
                     n_windows=len(d['windows']), n_total=d['n_total'])
        log.write(f'{sub}\t{status}\n'); log.flush()
        print(f'[{done}/{len(todo)}] {sub} {status} ({(time.time() - t0) / 60:.1f} min)', flush=True)
print('DONE', flush=True)
