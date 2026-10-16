"""Read-only source audit: fixed wall-clock hypotheses, never timezone fitting by PnL.

Run: .venv/Scripts/python.exe -X utf8 scripts/audit_telegram_labels.py
Bundled archive programs are never executed. No signal is promoted to a fill.
"""
from pathlib import Path
import hashlib
import html
import json
import time
import zipfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.collections import LineCollection

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/telegram_labels_v1'
DATA = ROOT / 'data/derived/telegram_labels_v1'
COLORS = {'wall_UTCplus2': '#0b9e88', 'wall_UTCplus0': '#e37a22', 'archive_reference': '#7b60bd'}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def wall(value):
    if pd.isna(value) or not str(value).strip():
        return pd.NaT
    return pd.Timestamp(value).tz_localize(None)


def utc(value):
    return pd.to_datetime(value, utc=True, errors='coerce')


def safe_sources():
    base = (DATA / 'sources').resolve()
    base.mkdir(parents=True, exist_ok=True)
    manifests, paths = [], {}
    for archive in sorted((ROOT / 'data/labels').glob('*.zip')):
        members = []
        with zipfile.ZipFile(archive) as z:
            for item in z.infolist():
                if item.is_dir() or any(x in item.filename for x in ('__pycache__/', '.pytest_cache/')):
                    continue
                target = (base / item.filename).resolve()
                if not target.is_relative_to(base):
                    raise ValueError(f'Unsafe archive member: {item.filename}')
                raw = z.read(item)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(raw)
                members.append({'member': item.filename, 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)})
                paths[(archive.name.split('-')[0].split('_KUMAR')[0], target.name)] = target
        manifests.append({'archive': archive.name, 'sha256': digest(archive), 'members': members})
    (OUT / 'source_manifest.json').write_text(json.dumps(manifests, indent=2), encoding='utf8')
    return paths


def normalize(paths):
    records = []
    counts = {}

    def add(trader, r, ident, local, reference, low, high, role='call_publication', public_local=None, ready=True, text='', sl=None, tps=None):
        w = wall(local)
        rec = dict(trader=trader, signal_id=str(ident), clock_role=role, wall_time=str(w) if pd.notna(w) else None,
                   archive_reference_utc=utc(reference), wall_UTCplus2_utc=w.tz_localize('UTC') - pd.Timedelta(hours=2) if pd.notna(w) else pd.NaT,
                   wall_UTCplus0_utc=w.tz_localize('UTC') if pd.notna(w) else pd.NaT,
                   public_wall_time=str(wall(public_local)) if public_local is not None else None,
                   side=str(r['side']), entry_low=float(low), entry_high=float(high),
                   price_ready=bool(ready) and bool(np.isfinite([float(low), float(high)]).all()),
                   timezone_status='UNRESOLVED_HYPOTHESES', execution_status='PROPOSAL_NOT_FILL',
                   training_admitted=False, raw_text_role='PROVENANCE_ONLY_MAY_CONTAIN_LATER_OUTCOME_ANNOTATIONS',
                   historical_text_integrity='UNVERIFIED_POSSIBLY_EDITED' if trader == 'T3' else 'ORIGINAL_EDIT_HISTORY_NOT_PRESENT',
                   sl=float(sl) if sl is not None and pd.notna(sl) else None, tp_values=str(tps), raw_text=str(text))
        records.append(rec)

    d = pd.read_csv(paths['TRADER_1', 'signal_calls.csv'])
    counts['T1_raw_call_events'] = len(d)
    counts['T1_logical_call_ids'] = int(d.trade_instance_id.nunique())
    # Preserve an earlier attention-only event, but NEVER backdate later entry prices.
    first_attention = d.sort_values('timestamp_utc').drop_duplicates('trade_instance_id').set_index('trade_instance_id')
    d['has_entry_prices'] = np.isfinite(d[['proposed_entry_low', 'proposed_entry_high']]).all(axis=1)
    d = d.sort_values(['has_entry_prices', 'timestamp_utc'], ascending=[False, True]).drop_duplicates('trade_instance_id', keep='first').sort_values('timestamp_utc')
    for _, r in d.iterrows():
        add('T1', r, r.trade_instance_id, r.timestamp_display_calendar, r.timestamp_utc, r.proposed_entry_low, r.proposed_entry_high,
            text=r.text_raw, sl=r.sl, tps=r.tp_values_json)
        records[-1]['literal_export_utc'] = utc(r.timestamp_title_literal_utc)
        records[-1]['source_timestamp_text'] = r.datetime_title_raw
        early = first_attention.loc[r.trade_instance_id]
        records[-1]['first_attention_archive_utc'] = utc(early.timestamp_utc)
        records[-1]['first_attention_wall_time'] = str(wall(early.timestamp_display_calendar))
        records[-1]['first_attention_event_id'] = early.event_id
        records[-1]['price_event_id'] = r.event_id
        records[-1]['price_time_semantics'] = 'FIRST_PRICE_BEARING_CALL_NOT_EARLIER_ATTENTION' if r.has_entry_prices else 'ATTENTION_ONLY_NO_ENTRY_PRICE'
    d = pd.read_csv(paths['TRADER_2', 'canonical_signal_calls.csv'])
    counts['T2_canonical_calls'] = len(d)
    for _, r in d.iterrows():
        add('T2', r, r.canonical_signal_id, r.signal_time_warsaw, r.signal_time_utc, r.proposed_entry_low, r.proposed_entry_high,
            ready=r.source_value_consistency == 'CONSISTENT' and bool(r.analysis_price_ready), text=r.primary_text_raw, sl=r.sl, tps=r.tp_values_json)
    d = pd.read_csv(paths['TRADER_3', 'canonical_signal_calls.csv'])
    counts['T3_calls'] = len(d)
    for _, r in d.iterrows():
        for role, local, ref in [('forwarded_source', r.source_call_time_warsaw, r.source_call_time_utc),
                                 ('public_repost', r.public_repost_time_warsaw, r.public_repost_time_utc)]:
            add('T3', r, r.signal_id, local, ref, r.entry_low, r.entry_high, role=role,
                public_local=r.public_repost_time_warsaw, text=r.raw_text, sl=r.sl, tps=[r.tp1, r.tp2, r.tp3])
    d = pd.read_csv(paths['TRADER_6', 'canonical_xau_labels.csv.gz'])
    counts['T6_calls'] = len(d)
    for _, r in d.iterrows():
        # Strict source-field whitelist. No best horizon, future path, claims, or quality labels.
        add('T6', r, r.call_id, r.signal_ts_local, r.signal_ts_m1_utc, r.entry_low_raw, r.entry_high_raw,
            text=r.raw_text, sl=r.sl_raw, tps=r.tp_values)
        records[-1]['literal_export_utc'] = utc(r.signal_ts_export_utc)
        records[-1]['side_source'] = r.side_source
    result = pd.DataFrame(records)
    result['record_id'] = result.trader + ':' + result.signal_id + ':' + result.clock_role
    result.to_csv(OUT / 'normalized_calls.csv', index=False)
    return result, counts


def market():
    source = ROOT / 'data/raw/ctrader/OHLC/xauusd_m1_full.csv'
    stat = source.stat()
    key = dict(path=str(source), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns,
               start='2025-12-18', end_exclusive='2026-08-21')
    cache, receipt = DATA / 'xauusd_audit_slice.parquet', DATA / 'ohlc_cache_identity.json'
    if cache.exists() and receipt.exists() and json.loads(receipt.read_text()) == key:
        d = pd.read_parquet(cache)
    else:
        lo, hi = [int(pd.Timestamp(x, tz='UTC').timestamp()*1000) for x in (key['start'], key['end_exclusive'])]
        parts = []
        for c in pd.read_csv(source, chunksize=500000):
            sub = c.loc[(c.timestamp_ms >= lo) & (c.timestamp_ms < hi)]
            if len(sub):
                parts.append(sub)
        d = pd.concat(parts, ignore_index=True)
        d.to_parquet(cache, index=False)
        receipt.write_text(json.dumps(key, indent=2))
    ts = pd.to_datetime(d.timestamp_utc, utc=True)
    assert (ts.astype('int64').to_numpy() // 1000000 == d.timestamp_ms.to_numpy()).all(), 'ISO/epoch mismatch'
    assert d.timestamp_ms.is_monotonic_increasing and not d.timestamp_ms.duplicated().any()
    assert (d.timestamp_ms % 60000 == 0).all()
    assert (d.low <= d[['open', 'close']].min(axis=1)).all() and (d.high >= d[['open', 'close']].max(axis=1)).all()
    d.index = pd.DatetimeIndex(ts)
    gaps = d.index.to_series().diff().dt.total_seconds().div(60).sub(1)
    pd.DataFrame({'next_bar_utc': d.index[gaps.gt(0)], 'missing_minutes': gaps[gaps.gt(0)].to_numpy()}).to_csv(OUT / 'ohlc_gaps.csv', index=False)
    info = dict(**key, rows=len(d), first=str(d.index[0]), last=str(d.index[-1]),
                subset_sha256=digest(cache), gaps=int(gaps.gt(0).sum()),
                clock_checks='ISO timestamp == epoch_ms; unique ordered minute grid',
                bar_clock_semantics='Assumed cTrader minute open; within-bar event sequence unavailable',
                quote_semantics='XAUUSD OHLC; no synchronized ask and no execution proof; not GC futures')
    return d, info


def distance(price, lo, hi):
    return max(lo-price, price-hi, 0.0)


def compare(calls, bars):
    rows = []
    for _, r in calls.iterrows():
        for candidate in COLORS:
            t = r[candidate + '_utc']
            q = {k: r[k] for k in ['record_id', 'trader', 'signal_id', 'clock_role', 'side', 'entry_low', 'entry_high', 'price_ready']}
            q.update(candidate=candidate, timestamp_utc=t, season='winter' if pd.notna(t) and t < pd.Timestamp('2026-03-29', tz='UTC') else 'summer')
            q['status'] = 'UNKNOWN_SOURCE_TIME' if pd.isna(t) else 'CONFLICT_OR_MISSING_PRICE' if not r.price_ready or not np.isfinite([r.entry_low, r.entry_high]).all() else 'OK'
            if q['status'] == 'OK':
                minute = t.floor('min')
                prev = minute - pd.Timedelta(minutes=1)
                if minute not in bars.index or prev not in bars.index:
                    q['status'] = 'NO_CONTIGUOUS_M1_SUPPORT'
                else:
                    b, p = bars.loc[minute], bars.loc[prev]
                    q.update(previous_complete_close=float(p.close), previous_bar_utc=prev,
                             distance_previous_close=distance(p.close, r.entry_low, r.entry_high),
                             distance_containing_bar=max(r.entry_low-b.high, b.low-r.entry_high, 0.0),
                             containing_bar_low=float(b.low), containing_bar_high=float(b.high),
                             containing_bar_warning='RETROSPECTIVE_ONLY_NOT_AVAILABLE_AT_PUBLICATION')
            rows.append(q)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT / 'clock_candidates.csv', index=False)
    paired_rows, summaries = [], []
    for (trader, role), group in metrics.groupby(['trader', 'clock_role']):
        for season in ['all', 'winter', 'summer']:
            g = group if season == 'all' else group[group.season == season]
            if g.empty:
                continue
            p = g.pivot(index='record_id', columns='candidate', values='distance_previous_close').dropna(subset=['wall_UTCplus2', 'wall_UTCplus0'])
            if not len(p):
                continue
            delta = p.wall_UTCplus0 - p.wall_UTCplus2
            # Paired support only; identical calls, no per-event clock cherry-picking.
            item = dict(trader=trader, role=role, season=season, paired_n=len(p),
                        median_distance_plus2=float(p.wall_UTCplus2.median()), median_distance_plus0=float(p.wall_UTCplus0.median()),
                        median_distance_archive=float(p.archive_reference.median()),
                        plus2_closer=int((delta > 1e-9).sum()), plus0_closer=int((delta < -1e-9).sum()), ties=int((delta.abs() <= 1e-9).sum()),
                        median_paired_improvement_plus2=float(delta.median()),
                        within_2usd_plus2=int((p.wall_UTCplus2 <= 2).sum()), within_2usd_plus0=int((p.wall_UTCplus0 <= 2).sum()))
            summaries.append(item)
            if season == 'all':
                for rid, pp in p.iterrows():
                    paired_rows.append(dict(record_id=rid, trader=trader, role=role, d_plus2=pp.wall_UTCplus2, d_plus0=pp.wall_UTCplus0, improvement_plus2=pp.wall_UTCplus0-pp.wall_UTCplus2))
    pd.DataFrame(summaries).to_csv(OUT / 'clock_comparison_summary.csv', index=False)
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(OUT / 'paired_clock_comparison.csv', index=False)
    return metrics, summaries, paired


def candles(ax, b):
    x = mdates.date2num(b.index.to_pydatetime())
    colors = np.where(b.close >= b.open, '#16876f', '#be4d60')
    ax.vlines(x, b.low, b.high, colors=colors, lw=.7, alpha=.85)
    seg = [((a, o), (a, c if o != c else c+.001)) for a, o, c in zip(x, b.open, b.close)]
    ax.add_collection(LineCollection(seg, colors=colors, linewidths=2))


def make_charts(calls, bars, metrics, paired):
    charts = OUT / 'charts'
    charts.mkdir(exist_ok=True)
    chosen = {}
    rng = np.random.default_rng(20261015)
    for trader in ['T1', 'T2', 'T3', 'T6']:
        g = paired[paired.trader == trader]
        if trader == 'T3':
            g = g[g.role == 'forwarded_source']
        for rid in rng.choice(g.record_id.to_numpy(), min(5, len(g)), replace=False):
            chosen[rid] = 'seeded_random_20261015'
        for label, sub in [('largest_plus2_advantage', g.nlargest(2, 'improvement_plus2')), ('largest_plus0_advantage', g.nsmallest(2, 'improvement_plus2'))]:
            for rid in sub.record_id:
                chosen.setdefault(rid, label)
    for _, r in calls[(calls.trader == 'T2') & (calls.wall_time < '2026-03-29')].head(2).iterrows():
        chosen.setdefault(r.record_id, 'winter_fixed_offset_vs_Warsaw')
    for _, r in calls[(calls.trader == 'T3') & calls.wall_time.isna()].iterrows():
        chosen[r.record_id] = 'missing_source_time_show_publication_only'
    selections = []
    for number, (rid, reason) in enumerate(chosen.items(), 1):
        r = calls[calls.record_id == rid].iloc[0]
        clocks = {k: r[k+'_utc'] for k in COLORS if pd.notna(r[k+'_utc'])}
        public_w = wall(r.public_wall_time) if r.public_wall_time else pd.NaT
        public = public_w.tz_localize('UTC')-pd.Timedelta(hours=2) if pd.notna(public_w) else pd.NaT
        if not clocks and pd.notna(public):
            clocks = {'wall_UTCplus2': public, 'wall_UTCplus0': public+pd.Timedelta(hours=2)}
        if not clocks:
            continue
        start, end = min(clocks.values())-pd.Timedelta(minutes=45), max(clocks.values())+pd.Timedelta(minutes=45)
        b = bars.loc[start:end]
        fig, (ax, lower) = plt.subplots(2, 1, figsize=(14, 7.8), gridspec_kw={'height_ratios':[4, 1.3]})
        if len(b):
            candles(ax, b)
        ax.axhspan(r.entry_low, r.entry_high, color='#3766ab', alpha=.13, label=f'Proposed entry {r.entry_low:g}–{r.entry_high:g}; NOT fill')
        if pd.notna(r.sl):
            ax.axhline(r.sl, color='#bd5365', lw=.8, ls=':', label=f'SL proposal {r.sl:g}')
        for k, t in clocks.items():
            duplicate = k == 'archive_reference' and any(t == clocks[j] for j in clocks if j != k)
            if not duplicate:
                ax.axvline(t, color=COLORS[k], lw=1.6, ls='--', label=f'{k}: {t:%H:%M:%S} UTC')
            # Explicit hollow marker at zone mid represents a proposal, not a trade price.
            ax.plot(t, (r.entry_low+r.entry_high)/2, marker='o', ms=6, mfc='none', mec=COLORS[k])
        if r.trader == 'T3' and r.clock_role == 'forwarded_source' and pd.notna(public):
            if start <= public <= end:
                ax.axvline(public, color='#202b3b', ls=':', label=f'Public repost (+2 hypothesis): {public:%H:%M:%S}')
        ax.set_xlim(start, end)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=__import__('datetime').timezone.utc))
        ax.set_ylabel('XAUUSD OHLC price (USD/oz)')
        ax.grid(alpha=.15)
        ax.legend(fontsize=8, loc='best', ncol=2)
        title_role = 'SOURCE TIME UNKNOWN — lines show public repost only' if pd.isna(r.wall_UTCplus2_utc) else r.clock_role
        ax.set_title(f'{r.trader} | {r.signal_id} | {r.side}\n{start:%Y-%m-%d} UTC | {title_role}', fontsize=11, loc='left')
        lower.axis('off')
        lines = [f'Selection: {reason}. Wall clock: {r.wall_time}.', 'Hypotheses: UTC+2 means UTC = displayed time − 2 hours; UTC+0 means UTC = displayed time.']
        for k in ['wall_UTCplus2', 'wall_UTCplus0']:
            q = metrics[(metrics.record_id == rid) & (metrics.candidate == k)].iloc[0]
            lines.append(f'{k}: previous complete M1 close distance to zone = {q.distance_previous_close:.2f} USD/oz' if q.status == 'OK' else f'{k}: {q.status}')
        if r.trader == 'T3':
            lines.append(f'Public wall time: {r.public_wall_time}; edited text unverified. Source time is NOT public availability.')
        lines.append('Candles after each candidate are future context. Full current-minute OHLC is unavailable at publication. No execution/PnL inferred.')
        lower.text(0, 1, '\n'.join(lines), va='top', fontsize=9, linespacing=1.7)
        fig.tight_layout()
        name = f'{number:02d}_{r.trader}.png'
        fig.savefig(charts / name, dpi=125)
        plt.close(fig)
        selections.append(dict(record_id=rid, trader=r.trader, reason=reason, file='charts/'+name))
    pd.DataFrame(selections).to_csv(OUT / 'visual_sample_manifest.csv', index=False)
    return selections


def html_report(summary, calls, metrics, selections, receipt):
    links = {s['record_id']: s['file'] for s in selections}
    rows = []
    for _, r in calls.iterrows():
        desc = html.escape(f'{r.trader} {r.signal_id} {r.clock_role}')
        link = f'<a href="{links[r.record_id]}" target="_blank">график</a>' if r.record_id in links else 'CSV'
        rows.append('<tr><td>'+desc+'</td><td>'+html.escape(str(r.wall_time))+'</td><td>'+html.escape(str(r.wall_UTCplus2_utc))+'</td><td>'+html.escape(str(r.wall_UTCplus0_utc))+'</td><td>'+html.escape(f'{r.side} {r.entry_low:g}–{r.entry_high:g}')+'</td><td>'+link+'</td></tr>')
    cards = ''.join(f'<figure><a href="{s["file"]}"><img loading="lazy" src="{s["file"]}"></a><figcaption>{html.escape(s["record_id"])} · {s["reason"]}</figcaption></figure>' for s in selections)
    table = pd.DataFrame(summary).to_html(index=False, float_format=lambda v:f'{v:.2f}', classes='summary')
    winter = next(s for s in summary if s['trader'] == 'T2' and s['season'] == 'winter')
    winter_summary = f"Зимняя выборка T2: медиана для архивной Europe/Warsaw (UTC+1 зимой) ${winter['median_distance_archive']:.2f}; для UTC+0 ${winter['median_distance_plus0']:.2f}; для UTC+2 ${winter['median_distance_plus2']:.2f}."
    page = '''<!doctype html><html lang="ru"><meta charset="utf-8"><title>Telegram — аудит времени и цены</title><style>
body{font:15px system-ui;color:#182438;background:#f3f5f8;margin:30px auto;max-width:1500px;padding:0 20px}h1{font-size:30px}p{max-width:1100px;line-height:1.6}.note{border-left:5px solid #e37a22;background:white;padding:15px}table{border-collapse:collapse;font-size:12px;background:white}td,th{padding:7px;border:1px solid #dbe2ea;text-align:left}section{overflow:auto}figure{margin:22px 0;background:white;padding:8px}img{width:100%;max-width:1400px}input{padding:10px;width:420px;max-width:90%}figcaption{padding:8px}a{color:#075fc1}</style>
<h1>Telegram calls: UTC+2 или UTC+0?</h1>
<p class="note">Показаны плановые зоны, а не доказанные исполнения. Часовой пояс остается гипотезой; совпадение цены — проверка согласованности, а не доказательство. XAUUSD spot сравнивается с XAUUSD OHLC. GC futures не подменяет spot.</p>
<p>Для обеих гипотез берем <b>одни и те же calls</b>, только при наличии текущей и предыдущей M1. Расстояние: от close предыдущей полной минуты до ближайшей границы entry-зоны (ноль внутри). Это цена в USD/oz, не прибыль. Свеча текущей минуты содержит будущее и хранится только как ретроспективная диагностика. Никаких сдвигов по TP, MFE или PnL.</p>
<p>UTC+2: UTC = видимые часы − 2 часа; UTC+0: UTC = видимые часы. Зимой Europe/Warsaw = UTC+1, поэтому архивная версия отдельно сохранена в CSV. Предположение о начале минуты cTrader требует подтверждения исходным экспортером; ISO и epoch_ms внутри CSV проверены.</p>
<p>T1: повторы одного trade_instance объединены по первой публикации цены; более ранний precall без цены сохранен отдельно и не получает позднюю цену задним числом. T2: четыре конфликта цен исключены из парных метрик. T3: source и public repost разделены, исходное время одного image-only call неизвестно; все 12 calls имеют риск редактирования. Kumar: исходный literal UTC и архивный −60m сохранены; поля будущей траектории не включены.</p>
<h2>Парное сравнение часов</h2><section>SUMMARY</section>
<p><b>Сезонность:</b> WINTER_SUMMARY Сравнение фиксированных +2 / +0 не заменяет проверку сезонной зоны. Все версии сохранены; окончательная зона не объявлена установленной.</p>
<p>plus2_closer / plus0_closer / ties — число calls по меньшему расстоянию. within_2usd — число calls не дальше $2/oz от зоны. Это зависимые события внутри дней, не независимые статистические доказательства.</p>
<p><a href="normalized_calls.csv">Нормализованные calls</a> · <a href="clock_candidates.csv">Все clock-кандидаты и расстояния</a> · <a href="clock_comparison_summary.csv">Сводка CSV</a> · <a href="audit_receipt.json">Воспроизводимость</a> · <a href="INDEPENDENT_AUDIT_SOL61_RU.md">Независимый аудит Sol 6.1</a></p>
<p>Все строки имеют training_admitted=false. raw_text сохранен только для проверки источника: он может содержать поздние заявления об исходе и не допускается в признаки модели.</p>
<h2>Все метки</h2><input id="filter" placeholder="Фильтр: трейдер, дата, signal_id…"><section style="max-height:440px"><table id="calls"><thead><tr><th>Метка / роль времени</th><th>Видимые часы</th><th>Гипотеза UTC+2 → UTC</th><th>Гипотеза UTC+0 → UTC</th><th>План</th><th>Просмотр</th></tr></thead><tbody>ROWS</tbody></table></section>
<h2>Визуальная выборка</h2><p>По пять случайных сцен на трейдера с seed 20261015; дополнительно наиболее сильные расхождения в обе стороны, зимние примеры и неизвестное время источника. Причина отбора указана; галерея намеренно не является оценкой доли успешных сигналов.</p>CARDS
<script>document.getElementById('filter').addEventListener('input',e=>{const q=e.target.value.toLowerCase();document.querySelectorAll('#calls tbody tr').forEach(r=>r.hidden=!r.textContent.toLowerCase().includes(q));});</script></html>'''
    (OUT/'index.html').write_text(page.replace('WINTER_SUMMARY', winter_summary).replace('SUMMARY', table).replace('ROWS', ''.join(rows)).replace('CARDS', cards), encoding='utf8')


def main():
    t0 = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    paths = safe_sources()
    calls, counts = normalize(paths)
    print('Normalized', counts, flush=True)
    bars, market_receipt = market()
    print('Loaded OHLC', len(bars), flush=True)
    metrics, summary, paired = compare(calls, bars)
    selections = make_charts(calls, bars, metrics, paired)
    receipt = dict(schema='telegram_clock_audit_v1', counts=counts, normalized_rows=len(calls),
                   distinct_calls=int(calls[['trader','signal_id']].drop_duplicates().shape[0]),
                   market=market_receipt, summary=summary, chart_count=len(selections),
                   candidate_status_counts=metrics.groupby(['trader','clock_role','candidate','status']).size().to_dict(),
                   policy='No clock selection using future outcomes; no fills or economic admission; no training',
                   script_sha256=digest(Path(__file__)), seconds=time.perf_counter()-t0)
    receipt['candidate_status_counts'] = {'|'.join(k):int(v) for k,v in receipt['candidate_status_counts'].items()}
    (OUT/'audit_receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf8')
    html_report(summary, calls, metrics, selections, receipt)
    print(json.dumps({'counts': counts, 'charts':len(selections), 'seconds':receipt['seconds'], 'summary':summary}, indent=2))


if __name__ == '__main__':
    main()
