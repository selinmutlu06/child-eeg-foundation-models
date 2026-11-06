"""Process HBN movie-watching EEG (The Present, Despicable Me) for releases 1-3.

Per child and movie, saves:
- standard features, full log spectrum, pretrained LaBraM embeddings (same as resting state)
- a time-locked 1-20 Hz signal at 50 Hz (artifacts set to NaN) for inter-subject correlation
- 1-second log alpha and theta power envelopes (artifacts NaN) for envelope synchrony
Usage: python run_movie_here.py OUT_DIR DATA_DIR [max_children]
"""
import os, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
import mne, numpy as np, pandas as pd, torch
from scipy.signal import butter, sosfiltfilt, hilbert, welch
import hbn_pipeline as hp

mne.set_log_level('error')
MOVIES = {'ThePresent': 200, 'DespicableMe': 168}       # seconds kept from video_start
out_dir, data_dir = sys.argv[1], sys.argv[2]
limit = int(sys.argv[3]) if len(sys.argv) > 3 else None
os.makedirs(out_dir, exist_ok=True)
torch.set_num_threads(2)


def download(rel, sub, task):
    s3 = hp.s3_client()
    base = hp.PREFIX.format(release=rel) + f'{sub}/eeg/{sub}_task-{task}_'
    out = {}
    for suffix in ('eeg.set', 'events.tsv'):
        local = os.path.join(data_dir, f'{sub}_{task}_{suffix}')
        s3.download_file(hp.BUCKET, base + suffix, local)
        out[suffix] = local
    return out


def clean_movie(set_path, events_path, seconds):
    """Same cleaning as resting state, then the segment from video_start, (19, seconds*200)."""
    raw = mne.io.read_raw_eeglab(set_path, preload=True)
    raw.filter(0.5, 75.0); raw.notch_filter(60.0)
    raw.set_montage(mne.channels.make_standard_montage('GSN-HydroCel-129'))
    egi = [c for c in raw.ch_names if c != 'Cz']
    log_sd = np.log(raw.get_data(picks=egi).std(axis=1) + 1e-12)
    mad = np.median(np.abs(log_sd - np.median(log_sd))) + 1e-12
    z = (log_sd - np.median(log_sd)) / (1.4826 * mad)
    raw.info['bads'] = [c for c, zz in zip(egi, z) if abs(zz) > 3]
    if raw.info['bads']:
        raw.interpolate_bads(reset_bads=True)
    raw.set_eeg_reference('average', projection=False)
    raw.pick(list(hp.EGI_TO_1020)); raw.rename_channels(hp.EGI_TO_1020); raw.reorder_channels(hp.CH_1020)
    raw.resample(hp.SFREQ)
    data = raw.get_data()
    ev = pd.read_csv(events_path, sep='\t')
    start = ev.loc[ev['value'] == 'video_start', 'onset']
    if start.empty:
        raise ValueError('no video_start marker')
    s = int(round(float(start.iloc[0]) * hp.SFREQ)); n = seconds * hp.SFREQ
    seg = np.full((19, n), np.nan)
    avail = data[:, s:s + n]
    seg[:, :avail.shape[1]] = avail
    return seg


def features(seg):
    fs = hp.SFREQ
    # 15 s windows for spectra and LaBraM, same rejection as resting state
    wins = []
    for st in range(0, seg.shape[1] - hp.WIN + 1, hp.WIN):
        w = seg[:, st:st + hp.WIN]
        if not np.isnan(w).any() and np.ptp(w, axis=1).max() * 1e6 < hp.PTP_REJECT_UV:
            wins.append(w)
    wins = np.stack(wins) if wins else np.empty((0, 19, hp.WIN))
    # time-locked signals for synchrony
    x = np.nan_to_num(seg)
    sos = butter(4, [1, 20], btype='band', fs=fs, output='sos')
    bb = sosfiltfilt(sos, x, axis=1)
    sd = 1.4826 * np.median(np.abs(bb - np.median(bb, axis=1, keepdims=True)), axis=1, keepdims=True) + 1e-12
    bad = (np.abs(bb) > 6 * sd) | np.isnan(seg)
    bad = np.convolve(bad.any(0), np.ones(fs // 2), mode='same') > 0          # widen by 0.25 s each side
    sig = bb.copy(); sig[:, bad] = np.nan
    sig50 = sig[:, ::4]                                                        # 50 Hz (already low-passed at 20 Hz)
    env = {}
    for name, (lo, hi) in {'theta': (4, 8), 'alpha': (8, 13)}.items():
        a = np.abs(hilbert(sosfiltfilt(butter(4, [lo, hi], btype='band', fs=fs, output='sos'), x, axis=1), axis=1))
        a[:, bad] = np.nan
        n_sec = a.shape[1] // fs
        with np.errstate(all='ignore'):
            env[name] = np.log(np.nanmean(a[:, :n_sec * fs].reshape(19, n_sec, fs), axis=2) + 1e-30)
    return wins, sig50.astype('float32'), env, float(bad.mean())


def prepare(rel, sub, task):
    try:
        paths = download(rel, sub, task)
        seg = clean_movie(paths['eeg.set'], paths['events.tsv'], MOVIES[task])
        wins, sig50, env, frac_bad = features(seg)
        if len(wins) < 2:
            return sub, task, None, f'only {len(wins)} clean windows'
        feats = hp.standard_features(wins)
        freqs, psd = welch(wins, fs=hp.SFREQ, nperseg=2 * hp.SFREQ, axis=-1)
        return sub, task, dict(wins=wins, sig50=sig50, env=env, frac_bad=frac_bad, feats=feats,
                               freqs=freqs, log_psd=np.log10(psd.mean(0) + 1e-30)), 'ok'
    except Exception as e:
        return sub, task, None, f'FAILED {e!r}'[:150]
    finally:
        for suffix in ('eeg.set', 'events.tsv'):
            f = os.path.join(data_dir, f'{sub}_{task}_{suffix}')
            if os.path.exists(f):
                os.remove(f)


jobs = []
for rel in ('R1', 'R2', 'R3'):
    p = hp.get_participants(rel, data_dir)
    for task in MOVIES:
        for s in p.loc[p[task] == 'available', 'participant_id']:
            if not os.path.exists(os.path.join(out_dir, f'{s}_{task}.npz')):
                jobs.append((rel, s, task))
jobs = list(dict.fromkeys(jobs))
if limit:
    jobs = jobs[:limit]
print(f'{len(jobs)} recordings to process', flush=True)

model = hp.load_labram()
log = open(os.path.join(out_dir, 'log.tsv'), 'a')
t0, done = time.time(), 0
with ThreadPoolExecutor(max_workers=3) as pool:
    for fut in as_completed([pool.submit(prepare, *j) for j in jobs]):
        sub, task, d, status = fut.result(); done += 1
        if d is not None:
            np.savez(os.path.join(out_dir, f'{sub}_{task}.npz'),
                     feat_names=np.array(list(d['feats'])), feats=np.array(list(d['feats'].values()), float),
                     labram=hp.labram_embeddings(model, d['wins']).astype('float32'),
                     freqs=d['freqs'].astype('float32'), log_psd=d['log_psd'].astype('float32'),
                     sig50=d['sig50'], env_theta=d['env']['theta'].astype('float32'),
                     env_alpha=d['env']['alpha'].astype('float32'),
                     n_windows=len(d['wins']), frac_bad=d['frac_bad'])
        log.write(f'{sub}\t{task}\t{status}\n'); log.flush()
        if done % 20 == 0 or status != 'ok':
            print(f'[{done}/{len(jobs)}] {sub} {task} {status} ({(time.time() - t0) / 60:.1f} min)', flush=True)
print('DONE', flush=True)
