"""Cache cleaned eyes-closed windows for held-out HBN releases (same QC: at least 2 clean windows).
Usage: python cache_new_release.py RAW_DIR WINDOW_DIR R4,R5
"""
import os, sys, time, glob
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import hbn_pipeline as hp

raw, win, rels = sys.argv[1], sys.argv[2], sys.argv[3].split(',')
os.makedirs(win, exist_ok=True)
old = {os.path.basename(f)[:-4] for f in glob.glob('/home/claude/hbn_work/windows/sub-*.npy')}
todo = []
for rel in rels:
    p = hp.get_participants(rel, raw)
    p = p[p['RestingState'].astype(str).str.lower().isin(['available', 'true', '1', 'yes'])] if 'RestingState' in p else p
    for s in p['participant_id']:
        if s not in old and not os.path.exists(os.path.join(win, f'{s}.npy')) and not os.path.exists(os.path.join(win, f'{s}.skip')):
            todo.append((rel, s))
print(len(todo), 'to cache', flush=True)

def one(rs):
    rel, sub = rs
    try:
        paths = hp.download_resting_state(rel, sub, raw)
        w, _ = hp.eyes_closed_windows(paths['eeg.set'], paths['events.tsv'])
        if len(w) < 2:
            open(os.path.join(win, f'{sub}.skip'), 'w').write(f'{len(w)} windows'); return sub, f'skip {len(w)}'
        np.save(os.path.join(win, f'{sub}.npy'), w.astype('float32')); return sub, f'ok {len(w)}'
    except Exception as e:
        open(os.path.join(win, f'{sub}.skip'), 'w').write(repr(e)[:200]); return sub, f'FAILED {e!r}'[:120]
    finally:
        for suf in ('eeg.set', 'events.tsv'):
            f = os.path.join(raw, f'{sub}_RestingState_{suf}')
            if os.path.exists(f): os.remove(f)

t0 = time.time(); n = 0
with ThreadPoolExecutor(max_workers=4) as pool:
    for fut in as_completed([pool.submit(one, x) for x in todo]):
        sub, st = fut.result(); n += 1
        if n % 25 == 0 or not st.startswith('ok'):
            print(f'[{n}/{len(todo)}] {sub} {st} ({(time.time() - t0) / 60:.1f} min)', flush=True)
print('DONE', flush=True)
