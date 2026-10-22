"""Tug-of-war smoothing of the opportunity label and a spectral (STFT) look at the same market.

1. Impulse u(t) = side * clip((R - 2) / 2, 0, 1) from the side strengths R (move before the own -G stop, in G,
   within H = 2 (G / sigma)^2): zero unless a real one-sided 2G opportunity starts at t. Three versions of the direction line per scale G: raw u, symmetric fading
   (EMA, half-life tau) and the tug of war (same tau, opposition fades 5x faster), both integrated from the
   future to the present; tau = half the median horizon of the scale. Smoothness (hysteresis flips per hour,
   roughness = mean |dS| / std S, share of power at periods < 60 s, autocorrelation at 30 s) and coverage.
   No fidelity score against the label: the rope is built from the label, any agreement would be circular.
   The same on the sign-flipped null path.
2. Variance-ratio spectrum: the periodogram of 1-second mid increments divided by its flat expectation
   under random signs (sum of tapered squared increments). Above 1 at a period = one-sided pulling (trend),
   below 1 = return. Whole-session spectra on every A day; trailing 30-minute STFT frames every minute
   give a causal band map (<10 s, 10-60 s, 1-5 min, 5-30 min).
3. The four STFT band ratios added to the volatility sense (opportunity R >= 2): does the forecast improve?
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, log_loss
sys.path.insert(0, str(Path(__file__).parent))
import opportunity_sense as osn                       # noqa: E402
from spectrum.cognition import race as rc             # noqa: E402
from spectrum.cognition.hotpath import base_day       # noqa: E402

OUT = Path('reports/race_v1')
TAU = {12: 31, 18: 70, 27: 159, 40: 348}
BANDS = ((2, 10), (10, 60), (60, 300), (300, 1800))
SPEC_BANDS = ((2, 5), (5, 10), (10, 30), (30, 60), (60, 180), (180, 600), (600, 1800), (1800, 5400))
WIN, HOP = 1800, 60


def band_ratio(d, bands, n_fft=None):
    """Variance-ratio spectrum of increments d (Hann taper), averaged inside each band of periods (s)."""
    w = np.hanning(len(d)); x = d * w
    p = np.abs(np.fft.rfft(x)) ** 2; flat = np.sum(x ** 2)
    f = np.fft.rfftfreq(len(d), 1.)
    out = []
    for lo, hi in bands:
        m = (f > 1 / hi) & (f <= 1 / lo)
        out.append(float(p[m].mean() / flat) if m.any() and flat > 0 else np.nan)
    return out


def stft_frames(mid, k):
    """Causal trailing-window band ratios at every HOP seconds; frame value applies from its end onward."""
    d = np.diff(mid, prepend=np.nan); d = np.where(np.isfinite(d), d, 0.)
    ends = np.arange(k[0], k[-1] + 1, HOP); rows = []
    for e in ends:
        seg = d[max(0, e - WIN + 1):e + 1]
        rows.append(band_ratio(seg, BANDS) if len(seg) == WIN and np.sum(seg != 0) > 100 else [np.nan] * len(BANDS))
    frames = np.asarray(rows)
    idx = np.clip(np.searchsorted(ends, k, side='right') - 1, 0, None)
    return ends, frames, frames[idx]


def hysteresis_flips(s, band=.05):
    state = 0; flips = 0
    for v in s:
        if v > band and state != 1:
            flips += state == -1; state = 1
        elif v < -band and state != -1:
            flips += state == 1; state = -1
    return flips


def smooth_stats(s, RL, RS):
    """Smoothness and coverage of a line. No agreement-with-the-label score: the rope integrates the label itself."""
    s = np.nan_to_num(s); sd = s.std()
    f = np.fft.rfft(s - s.mean()); p = np.abs(f) ** 2; fr = np.fft.rfftfreq(len(s), 1.)
    a, b = s[30:], s[:-30]
    side, R = rc.strength(RL, RS)
    return dict(flips_h=hysteresis_flips(s) / (len(s) / 3600), rough=float(np.mean(np.abs(np.diff(s))) / sd) if sd else np.nan,
                hf=float(p[fr > 1 / 60].sum() / p[1:].sum()) if p[1:].sum() else np.nan,
                acf30=float(np.corrcoef(a, b)[0, 1]) if sd else np.nan,
                active=float((np.abs(s) >= .1).mean()), impulse_share=float(np.nanmean(np.abs(np.nan_to_num(rc.impulse(RL, RS))) > 0)),
                opp_rate=float(np.nanmean(R[np.isfinite(R)] >= 2)))


def day(date):
    out = osn.day(date)
    b, m = base_day(date); k = out['k']
    ends, frames, per_sec = stft_frames(b['mid'], k)
    out['spec'] = per_sec.astype(np.float32); out['frame_t'] = ends - k[0]; out['frames'] = frames.astype(np.float32)
    d = np.diff(b['mid'][k[0]:k[-1] + 1]); d = np.where(np.isfinite(d), d, 0.)
    out['day_spectrum'] = band_ratio(d, SPEC_BANDS)
    out['abs_spectrum'] = band_ratio(np.abs(d) - np.abs(d).mean(), SPEC_BANDS)
    null = rc.sign_flipped(b['mid'], b['seg'], np.random.default_rng(41 * 100000 + int(date.replace('-', '')) % 100000))
    out['null'] = {}
    for G in rc.GRID_G:
        e = rc.excursion(null, b['seg'], k, b['sigma'][k], G)
        out['null'][G] = (e['R_long'].astype(np.float32), e['R_short'].astype(np.float32))
    return out


def main():
    dates = osn.dates_of(('2025-10', '2025-11', '2025-12'))
    with ProcessPoolExecutor(max_workers=6) as pool:
        days = list(pool.map(day, dates))
    test = [d for d in days if d['date'] in osn.TEST]; train = [d for d in days if d['date'] not in osn.TEST]
    res = dict(tau=TAU, bands=BANDS, spec_bands=SPEC_BANDS)
    # 1. smoothing
    rows = []
    for d in test:
        for G in rc.GRID_G:
            for kind, (RL, RS) in (('real', (d[f'L{G}'], d[f'S{G}'])), ('null', d['null'][G])):
                u = np.nan_to_num(rc.impulse(RL, RS))
                for name, s in (('raw', u), ('ema', rc.tug_of_war(u, TAU[G], 1.)), ('tug', rc.tug_of_war(u, TAU[G], 5.))):
                    rows.append(dict(date=d['date'], G=G, kind=kind, line=name, **smooth_stats(s, RL, RS)))
    sm = pd.DataFrame(rows); res['smoothing'] = sm.groupby(['G', 'line', 'kind']).mean(numeric_only=True).round(4).reset_index().to_dict('records')
    # 2. spectra
    spec = np.array([d['day_spectrum'] for d in days]); aspec = np.array([d['abs_spectrum'] for d in days])
    res['spectrum'] = dict(mean=np.nanmean(spec, 0).tolist(), q10=np.nanquantile(spec, .1, 0).tolist(), q90=np.nanquantile(spec, .9, 0).tolist(),
                           days=len(days), abs_mean=np.nanmean(aspec, 0).tolist())
    # frames vs the next 30 minutes of opportunity
    rel = []
    for d in days:
        R = np.fmax(d['L18'], d['S18']); opp = (R >= 2).astype(float); opp[~np.isfinite(R)] = np.nan
        t = d['k'] - d['k'][0]
        for ft, fr in zip(d['frame_t'], d['frames']):
            m = (t > ft) & (t <= ft + 1800)
            if m.sum() > 600 and np.isfinite(fr).all():
                rel.append(dict(date=d['date'], **{f'b{i}': np.log(fr[i]) for i in range(len(BANDS))}, opp_next=float(np.nanmean(opp[m]))))
    rel = pd.DataFrame(rel)
    res['frames_vs_next30'] = {f'b{i}': float(rel[f'b{i}'].corr(rel.opp_next)) for i in range(len(BANDS))}
    # 3. volatility sense with and without STFT bands
    res['sense'] = {}
    for G in rc.GRID_G:
        def xy(ds, stride, spec):
            X = np.concatenate([np.c_[d['X'], np.log(np.maximum(d['spec'], 1e-6))][::stride] if spec else d['X'][::stride] for d in ds])
            R = np.concatenate([np.fmax(d[f'L{G}'], d[f'S{G}'])[::stride] for d in ds]); ok = np.isfinite(R)
            return np.nan_to_num(X[ok]), (R[ok] >= 2).astype(int)
        r = {}
        for spec in (False, True):
            Xtr, ytr = xy(train, 10, spec); Xte, yte = xy(test, 1, spec)
            mdl = HistGradientBoostingClassifier(max_depth=3, learning_rate=.05, max_iter=250, min_samples_leaf=500, l2_regularization=1., random_state=7).fit(Xtr, ytr)
            p = mdl.predict_proba(Xte)[:, 1]
            r['with_stft' if spec else 'base'] = dict(auc=float(roc_auc_score(yte, p)), ll=float(log_loss(yte, p)), ll_rate=float(log_loss(yte, np.full(len(yte), ytr.mean()))))
            # spectral-only model: is the band map informative by itself?
        Xtr, ytr = xy(train, 10, True); Xte, yte = xy(test, 1, True)
        cols = list(range(len(osn.FEATS), len(osn.FEATS) + len(BANDS)))
        mdl = HistGradientBoostingClassifier(max_depth=3, learning_rate=.05, max_iter=250, min_samples_leaf=500, l2_regularization=1., random_state=7).fit(Xtr[:, cols], ytr)
        p = mdl.predict_proba(Xte[:, cols])[:, 1]; r['stft_only'] = dict(auc=float(roc_auc_score(yte, p)), ll=float(log_loss(yte, p)))
        res['sense'][str(G)] = r
        print(G, json.dumps(r), flush=True)
    (OUT / 'tug_spectrum.json').write_text(json.dumps(res, indent=1), encoding='utf8')
    # page data: test week, 5 s
    page = dict(grid=list(rc.GRID_G), tau=TAU, bands=[list(b) for b in BANDS], spectrum=res['spectrum'], spec_bands=[list(b) for b in SPEC_BANDS], days=[])
    for d in test:
        sub = slice(None, None, 5); t = (d['k'] - d['k'][0])
        dd = dict(date=d['date'], t=t[sub].tolist(), mid=np.round(d['mid'][sub] / 10, 2).tolist(), frame_t=d['frame_t'].tolist(),
                  frames=[[None if not np.isfinite(v) else round(float(np.log2(v)), 3) for v in fr] for fr in d['frames']], G={})
        for G in rc.GRID_G:
            RL, RS = d[f'L{G}'], d[f'S{G}']; u = np.nan_to_num(rc.impulse(RL, RS))
            nRL, nRS = d['null'][G]; nu = np.nan_to_num(rc.impulse(nRL, nRS))
            dd['G'][str(G)] = dict(u=[int(round(v * 100)) for v in u[sub]], tug=[int(round(v * 1000)) for v in rc.tug_of_war(u, TAU[G], 5.)[sub]],
                                   ema=[int(round(v * 1000)) for v in rc.tug_of_war(u, TAU[G], 1.)[sub]],
                                   ntug=[int(round(v * 1000)) for v in rc.tug_of_war(nu, TAU[G], 5.)[sub]])
        page['days'].append(dd)
    page['smoothing'] = res['smoothing']; page['sense'] = res['sense']; page['frames_vs_next30'] = res['frames_vs_next30']
    (OUT / 'tug_data.json').write_text(json.dumps(page, separators=(',', ':')), encoding='utf8')


if __name__ == '__main__':
    main()
