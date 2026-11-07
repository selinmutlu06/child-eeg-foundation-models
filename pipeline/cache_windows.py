"""Cache every child's cleaned eyes-closed resting windows (volts, 19 x 3000 at 200 Hz) so that
every foundation model sees exactly the same data. Only children already in OUT (the 373) are cached.
Usage: python cache_windows.py REST_OUT_DIR RAW_DIR WINDOW_DIR
"""
import glob, os, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import hbn_pipeline as hp

rest_out, raw_dir, win_dir = sys.argv[1:4]
os.makedirs(win_dir, exist_ok=True)
ids = [os.path.basename(f)[:-4] for f in sorted(glob.glob(os.path.join(rest_out, 'sub-*.npz')))]
release_of = {}
for rel in ('R1', 'R2', 'R3'):
    for s in hp.get_participants(rel, raw_dir)['participant_id']:
        release_of.setdefault(s, rel)
todo = [s for s in ids if not os.path.exists(os.path.join(win_dir, f'{s}.npy'))]
print(len(ids), 'children,', len(todo), 'to cache', flush=True)

def one(sub):
    try:
        paths = hp.download_resting_state(release_of[sub], sub, raw_dir)
        w, _ = hp.eyes_closed_windows(paths['eeg.set'], paths['events.tsv'])
        np.save(os.path.join(win_dir, f'{sub}.npy'), w.astype('float32'))
        return sub, f'ok {len(w)}'
    except Exception as e:
        return sub, f'FAILED {e!r}'[:120]
    finally:
        for suffix in ('eeg.set', 'events.tsv'):
            f = os.path.join(raw_dir, f'{sub}_RestingState_{suffix}')
            if os.path.exists(f):
                os.remove(f)

t0 = time.time(); n = 0
with ThreadPoolExecutor(max_workers=3) as pool:
    for fut in as_completed([pool.submit(one, s) for s in todo]):
        sub, st = fut.result(); n += 1
        if n % 25 == 0 or not st.startswith('ok'):
            print(f'[{n}/{len(todo)}] {sub} {st} ({(time.time() - t0) / 60:.1f} min)', flush=True)
print('DONE', flush=True)
