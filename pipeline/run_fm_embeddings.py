"""Extract per-layer embeddings from each EEG foundation model for every cached child.
Uses each child's first 4 clean 15 s windows. Restartable: skips children already in the output.
Usage: python run_fm_embeddings.py WINDOW_DIR OUT_DIR model1,model2,... [pretrained|random]
"""
import glob, os, sys, time, warnings
import numpy as np, torch
import fm_adapters as F

warnings.filterwarnings('ignore')
win_dir, out_dir, models = sys.argv[1], sys.argv[2], sys.argv[3].split(',')
mode = sys.argv[4] if len(sys.argv) > 4 else 'pretrained'
os.makedirs(out_dir, exist_ok=True)
torch.set_num_threads(2)
files = sorted(glob.glob(os.path.join(win_dir, 'sub-*.npy')))

for name in models:
    out = os.path.join(out_dir, f'{name}_{mode}.npz')
    done = dict(np.load(out, allow_pickle=True)) if os.path.exists(out) else {'ids': np.array([]), 'emb': None}
    have = set(done['ids'].tolist())
    ids, embs = list(done['ids']), ([e for e in done['emb']] if done['emb'] is not None else [])
    load, prep, fwd = F.spec(name)
    model = load().eval()
    if mode == 'random':
        model = F.randomize(model).eval()
    t0 = time.time()
    for k, f in enumerate(files):
        sub = os.path.basename(f)[:-4]
        if sub in have:
            continue
        w = np.load(f)[:4].astype(np.float64)
        try:
            e = F.embed(model, fwd, prep(w), time_first=(name == 'bendr'))
        except Exception as ex:
            print(f'{name} {sub} FAILED {ex!r}'[:200], flush=True)
            continue
        ids.append(sub); embs.append(e.astype('float32'))
        if len(ids) % 40 == 0:
            np.savez(out, ids=np.array(ids), emb=np.stack(embs))
            print(f'{name} {mode}: {len(ids)}/{len(files)} ({(time.time() - t0) / 60:.1f} min)', flush=True)
    np.savez(out, ids=np.array(ids), emb=np.stack(embs))
    print(f'{name} {mode} DONE {len(ids)} children, emb {np.stack(embs).shape}, {(time.time() - t0) / 60:.1f} min', flush=True)
print('ALL DONE', flush=True)
