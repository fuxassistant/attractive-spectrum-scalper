"""Join independent scan receipts without promoting a clean hash to market truth."""
from collections import Counter
from pathlib import Path
import argparse,html,json
import numpy as np
import pandas as pd
from spectrum.data_properties_audit import Distribution
from spectrum.core.storage import sha
from spectrum.ingest import save_json


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def merged(items):
    result=Distribution()
    for histogram,stats in items:
        result.hist+=histogram
        result.negative+=stats['negative'];result.maximum=max(result.maximum,stats['max_ns'])
        result.total_ns+=round(stats['total_interval_seconds']*10**9)
        for k in result.thresholds:
            result.thresholds[k]+=stats['tail_counts_gt_ns'][str(k)]
            result.tail_ns[k]+=round(stats['tail_duration_seconds'][str(k)]*10**9)
    return result


def flags(rows):
    counts=Counter()
    for r in rows:counts.update(r['flags'])
    return dict(combinations=dict(counts),
        snapshot=sum(v for k,v in counts.items() if int(k)&32),
        maybe_bad_book=sum(v for k,v in counts.items() if int(k)&4),
        bad_receive_snapshot=sum(v for k,v in counts.items() if int(k)&8 and int(k)&32),
        bad_receive_non_snapshot=sum(v for k,v in counts.items() if int(k)&8 and not int(k)&32))


def report(root):
    root=Path(root)
    assert read(root/'scan_complete.json')['status']=='COMPLETE'
    assert read(root/'semantics/scan_complete.json')['status']=='COMPLETE'
    input_audit=read(root/'player_inputs/summary.json')
    clock_flags=read(root/'clock_flags/summary.json')
    legacy=read(root/'legacy_slices.json')
    legacy_loss=read(root/'legacy_snapshot_loss.json')
    geometry=read(root/'unconditioned_support/summary.json')
    invalid_probes=read(root/'invalid_book_probes.json')
    clipping=read(root/'model_input_clipping.json')
    assert clock_flags['status']=='COMPLETE'
    source=[read(p) for p in sorted((root/'raw').glob('member_*.json'))]
    raw={r['key']:r for m in source for r in m['captures']}
    market={r['key']:r for r in [read(p) for p in sorted((root/'market').glob('*.json'))]}
    semantics={r['key']:r for m in [read(p) for p in sorted((root/'semantics').glob('member_*.json'))] for r in m['captures']}
    rates=read(root/'rates/summary.json')
    master=read('data/derived/full_mbo_v1/manifest.json')
    reasons=Counter()
    for c in master['captures']:
        if c['status']=='COMPLETE':reasons.update(read(Path(c['market'])/'manifest.json')['reasons'])
    admission=read('reports/full_mbo_v1/audit_20261016/training_admission_overlay.json')
    distributions={}
    for name in ('all_record_interarrival','non_snapshot_record_interarrival','non_snapshot_recv_minus_event'):
        npkey={'all_record_interarrival':'all_raw','non_snapshot_record_interarrival':'non_snapshot_raw','non_snapshot_recv_minus_event':'latency'}[name]
        distributions[name]=merged([(np.load(root/'raw'/(r['member']+'_hist.npz'))[npkey],r[name]) for r in source])
    for name,npkey in (('selected_records','records'),('selected_publications','publications')):
        statkey='non_snapshot_record_interarrival' if npkey=='records' else 'non_snapshot_publication_interarrival'
        distributions[name]=merged([(np.load(root/'raw'/(key.split('/')[0]+'_hist.npz'))[key.replace('/','_')+'_'+npkey],r[statkey]) for key,r in raw.items()])
    for name in ('all_publications','valid_same_epoch','reference_supported','london_ny','bbo_price_changes','bbo_price_or_size_changes'):
        distributions[name]=merged([(np.load(root/'market'/(key.replace('/','_')+'_hist.npz'))[name],r['distributions'][name]) for key,r in market.items()])
    # Merge scalar counters independently; use the already summed histogram once.
    unconditioned=merged([(np.zeros(8194,np.int64),r['interval_stats']) for r in geometry['captures']])
    unconditioned.hist=np.load(root/'unconditioned_support/unconditioned_interarrival_hist.npy')
    distributions['london_ny_before_core_filters']=unconditioned
    publication_mismatches=[key for key,r in market.items() if r['publication_identity_sha256']!=raw[key]['publication_identity_sha256'] or r['quote_rows']!=raw[key]['publications']]
    integrity=dict(raw_members=len(source),source_hash_failures=[r['member'] for r in source if not r['source_hash_matches']],
        source_records=sum(r['records'] for r in source),selected_captures=len(raw),selected_records=sum(r['records'] for r in raw.values()),
        selected_partition_failures=[key for key,r in raw.items() if not r['raw_partition_matches']],
        reconstructed_captures=len(market),reconstructed_publications=sum(r['quote_rows'] for r in market.values()),
        reconstruction_payload_failures=[key for key,r in market.items() if not all(r['payload_hashes_match'].values())],
        publication_identity_mismatches=publication_mismatches,
        source_ordinal_nonincreasing=sum(r['source_ordinal_nonincreasing'] for r in market.values()),
        publication_clock_regressions=sum(r['publication_regressions'] for r in market.values()),
        valid_crossed_or_empty=sum(r['valid_crossed_or_empty'] for r in market.values()))
    totals=Counter()
    for r in semantics.values():
        for k in ('records','mutation_undefined_prices','zero_receive','recv_regressions','event_regressions','sequence_regressions','sequence_forward_jumps','negative_recv_event'):totals[k]+=r[k]
    trades={k:dict(sum((Counter(r[k]) for r in semantics.values()),Counter())) for k in ('trade_count','trade_volume','fill_count','fill_volume')}
    probes=[dict(key=key,**r['reference_book']) for key,r in semantics.items() if 'reference_book' in r]
    anomaly_dates=[]
    for key,r in raw.items():
        f=flags([r]);m=market.get(key,{})
        if f['maybe_bad_book'] or m.get('valid_london_ny_quotes_without_spot_reference',0):
            anomaly_dates.append(dict(key=key,date=r['date'],symbol=r['symbol'],maybe_bad_book=f['maybe_bad_book'],
                valid_london_ny_quotes_without_spot_reference=m.get('valid_london_ny_quotes_without_spot_reference',0),
                invalid_quotes=m.get('invalid_quotes')))
    result=dict(status='COMPLETE_DATA_CHARACTERIZATION_NOT_TRAINING_ADMISSION',integrity=integrity,
        cached_player_inputs=input_audit,legacy_overlapping_slices=legacy,
        confirmed_legacy_snapshot_loss=legacy_loss,unconditioned_support_geometry=geometry,
        invalid_book_reconstruction_probes=invalid_probes,model_input_clipping=clipping,
        archive_inventory=read(root/'archive_inventory.json'),
        field_semantics=dict(price='Integer nanodollars per troy ounce for selected GC; divide by 1e9',
            size='Native order quantity in contracts; not ounces and not number of messages',
            GC_contract='100 troy ounces; price tick USD0.10 per ounce',
            clocks='ts_recv is capture receive time, ts_event is exchange event time; causal replay uses published completed ts_recv/ordinal',
            grouping='F_LAST completes an instrument event; T/F/C can belong to one economic execution',
            volume='Count T trade size for traded volume; F is passive fill information, C resting-book removal; never sum all three as traded volume',
            missing_side='N preserved; no guessed aggressor side',
            simulation='1 oz, fees, latency and action policies are simulation assumptions, not raw market facts'),
        selected_non_snapshot_bad_clock_records=clock_flags,
        reconstruction_quality_reasons=dict(reasons),prior_admitted_captures=admission['eligible_captures'],
        source_flags=flags(source),selected_flags=flags(list(raw.values())),selected_clocks=dict(totals),trades=trades,
        distributions={name:d.result() for name,d in distributions.items()},wall_second_rates=rates,
        independent_book_probes=probes,
        independent_book_checks=sum(p['valid_publications_checked'] for p in probes),
        independent_book_errors=sum(len(p['errors']) for p in probes),anomaly_dates=anomaly_dates,
        persisted_core_support_ns=sorted({r['persisted_support_ns'] for r in market.values()}),
        valid_quotes_without_spot_reference=sum(r['valid_quotes_without_spot_reference'] for r in market.values()),
        valid_london_ny_quotes_without_spot_reference=sum(r['valid_london_ny_quotes_without_spot_reference'] for r in market.values()),
        limitations=['Population quantile intervals are bounded histogram values, not point estimates or samples.',
            'Recv-minus-event latency quantiles describe nonnegative values only; negative deltas are explicitly counted separately and must not be interpreted as interarrival times.',
            'Normal non-snapshot raw cadence still contains scheduled closures; trading-hours cadence is separately reported.',
            'Valid/same-epoch subsets are conditioned on the old 30-second core rule: their lack of >30-second gaps cannot validate that rule.',
            'OHLC reference absence is not exchange closure. Rates explicitly condition on known reference seconds.',
            'Sequence forward jumps in a filtered instrument/universe do not prove packet loss.',
            'Independent book replay checks only stated reset prefixes; not every order in history.',
            'Source hashes prove preserved local bytes, not completeness of vendor capture or achievable fills.'],
        code_findings=['Gym 5-second support contract conflicts with persisted 30-second reconstruction contract.',
            'Prepared player quote valid mask combines source book validity with spot-reference availability.',
            'Data quote publications include every F_LAST; not just changed bid/ask.',
            'Unchanged BBO, absence of messages, book corruption and scheduled closure need separate fields.'])
    save_json(root/'summary.json',result)
    pd.DataFrame([dict(key=k,date=r['date'],symbol=r['symbol'],**r['london_ny_per_second'],
        gaps_over_5s=len(r['gaps_over_5s_london_ny']),invalid_quotes=r['invalid_quotes'],
        missing_reference_valid_quotes=r['valid_london_ny_quotes_without_spot_reference']) for k,r in market.items()]).to_csv(root/'capture_properties.csv',index=False)
    rows=[]
    for name,d in distributions.items():
        if name=='non_snapshot_recv_minus_event':continue
        r=d.result();row=dict(stream=name,intervals=r['count'],zero_fraction=r['zeros']/max(1,r['count']))
        for q in ('0.5','0.9','0.99','0.999'):
            b=r['quantile_ns_bounds'][q];row['p'+q+'_milliseconds']=None if b is None else [v/10**6 for v in b]
        row['over_5s']=r['tail_counts_gt_ns'][str(5*10**9)];rows.append(row)
    save_json(root/'cadence_table.json',rows)
    plot(root,distributions,rates)
    body='<h1>Свойства исходных MBO и реконструкции</h1><p>Полный проход по доступным файлам. Это аудит данных, не допуск стратегии.</p>'
    body+='<p><b>Подтверждённые ошибки обработки:</b> независимые пороги поддержки 5/30 секунд; превращение отсутствующих spot-минут в invalid фьючерсного стакана; потеря snapshot в старой недельной вырезке. Месячные исходники и текущая реконструкция проверены отдельно.</p>'
    body+='<h2>Частота</h2><p>Медиана ничего не отбрасывает. Интервалы сообщений и число сообщений за календарную секунду — разные распределения. Нулевые интервалы и нулевые секунды сохранены.</p>'
    body+=pd.DataFrame(rows).to_html(index=False,escape=True)
    body+='<p>Квантили интервалов даны границами точных гистограммных ячеек; выборочного семплирования нет. Медиана публикаций в календарную секунду: '+str(rates['quantiles']['0.5'])+'.</p>'
    body+='<img src="cadence.png"><h2>Сохранность и соответствие</h2><pre>'+html.escape(json.dumps(integrity,indent=2))+'</pre>'
    body+='<h2>Признаки качества выбранных контрактов</h2><pre>'+html.escape(json.dumps(result['selected_flags'],indent=2))+'</pre>'
    body+='<h2>Даты, требующие отдельного обращения</h2>'+pd.DataFrame(anomaly_dates).to_html(index=False,escape=True)
    body+='<h2>Ошибки и ограничения обработки</h2><ul>'+''.join('<li>'+html.escape(s)+'</li>' for s in result['code_findings']+result['limitations'])+'</ul>'
    body+='<h2>Сделки и исполнения</h2><p>T, F и C не складываются как независимые объёмы сделок. Неизвестная сторона T остаётся отдельной категорией.</p><pre>'+html.escape(json.dumps(trades,indent=2))+'</pre>'
    body+='<h2>Независимая проверка стакана</h2><p>Проверено публикаций: '+str(result['independent_book_checks'])+'; несовпадений: '+str(result['independent_book_errors'])+'. Область — первые 20 000 исходных записей каждого заранее выбранного фрагмента.</p>'
    body+='<h2>Входы модели</h2><pre>'+html.escape(json.dumps(input_audit,indent=2))+'</pre><p>12 мая 20:15–20:24 UTC: 10 отсутствующих spot-минут превратили 7672 валидные публикации фьючерса в invalid. Это ошибка объединения масок, не доказательство закрытия биржи.</p>'
    body+='<h2>Дефект старой вырезки</h2><p>В GC_MBO_2026-05-04_to_2026-05-09.dbn.zst отсутствуют 13119 snapshot-записей, из них 3685 относятся к GCM6. Записи вне snapshot побайтно совпадают с исходником. Дефект находится в локальной вырезке, а не в месячном источнике; нынешний полный датасет её не использует.</p>'
    body+='<h2>Проверка без прежнего фильтра 30 секунд</h2><p>Число интервалов длиннее 30 секунд в торговом окне с известными spot-минутами: '+str(len(geometry['gaps_over_30s']))+'. Эти интервалы сохранены с точными часами и исходными ordinal; автоматически считать их потерей потока нельзя.</p>'
    body+='<h2>Невалидный стакан — отдельная категория</h2><p>В торговом окне с известной OHLC-опорой сохранено '+str(geometry['invalid_london_ny_known_reference'])+' невалидных публикаций. Три эпизода 2025-10-17, 2026-01-30 и 2026-05-22 независимо восстановлены от reset: перекрещенный стакан воспроизводится из сырых событий; валидные состояния до/после совпадают. Без биржевого status-потока нельзя доказать конкретную причину этого состояния.</p>'
    body+='<p>В старом формате invalid обнуляет bid/ask, а причины сохраняются только общими счётчиками фрагмента. Для дальнейшей диагностики требуется сохранять причину и наблюдаемое состояние отдельно от разрешения исполнения.</p>'
    body+='<h2>Единицы и преобразования</h2><pre>'+html.escape(json.dumps(result['field_semantics'],ensure_ascii=False,indent=2))+'</pre><p>Медиана не фильтрует исходные сообщения. На входе модели существует отдельное насыщение z-score до ±10: fast 0.0253%, base 0.0430%, slow 0.0427% известных ячеек на 30 днях; anchors 0%. Оно меняет амплитуды фич, не удаляет MBO. Динамические event-контексты в этой проверке clipping не включены.</p>'
    body+='<p>Определения: <a href="https://databento.com/docs/schemas-and-data-formats/mbo">MBO</a>, <a href="https://databento.com/docs/standards-and-conventions/mbo-snapshot">snapshot</a>. Полные числа и ограничения — summary.json; воспроизводимые результаты — raw/, market/, semantics/, rates/.</p>'
    (root/'index.html').write_text('<!doctype html><html lang="ru"><meta charset="utf-8"><title>Аудит MBO</title><style>body{font:16px system-ui;max-width:1400px;margin:32px auto;padding:0 24px;background:#101827;color:#e6edf5}table{border-collapse:collapse;font-size:13px;width:100%;overflow:auto}td,th{padding:7px;border:1px solid #364457}pre{white-space:pre-wrap;background:#1b283a;padding:18px}a{color:#7ec8ff}img{max-width:100%}h2{margin-top:36px}</style>'+body+'</html>',encoding='utf-8')
    save_json(root/'report_receipt.json',dict(status='COMPLETE',summary_sha256=sha(root/'summary.json'),html_sha256=sha(root/'index.html'),
        implementations={str(p):sha(p) for p in [Path(__file__),*Path('spectrum').glob('data_*audit.py'),Path('spectrum/data_support_geometry.py'),Path('spectrum/player_input_audit.py'),Path('spectrum/data_flag_evidence.py')]},
        training_admission=False,reason='Data characterization complete; observed code defects remain unresolved in the simulator'))
    print(json.dumps(integrity,indent=2))


def plot(root,distributions,rates):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(15,9))
    for name in ('selected_records','london_ny_before_core_filters','bbo_price_changes'):
        h=distributions[name].hist;x=np.exp2((np.arange(1,len(h))-1)/128)/10**6
        label={'selected_records':'Selected raw records (all hours)','london_ny_before_core_filters':'Publications London-NY, before age filters','bbo_price_changes':'BBO price changes, valid London-NY'}[name]
        axes[0,0].semilogx(x,np.cumsum(h)[1:]/max(1,h.sum()),label=label)
    axes[0,0].set(xlabel='Interarrival, ms (zero mass retained in CDF)',ylabel='Fraction of intervals',xlim=(.001,1e5));axes[0,0].legend()
    d=distributions['london_ny_before_core_filters'];h=d.hist;x=np.exp2((np.arange(1,len(h))-1)/128)/10**6
    mass=h[1:]*x
    axes[0,1].semilogx(x,np.cumsum(h)[1:]/h.sum(),label='Message-count CDF')
    axes[0,1].semilogx(x,np.cumsum(mass)/mass.sum(),label='Elapsed-time CDF (bin approximation)')
    axes[0,1].set(xlabel='Publication interval, ms',ylabel='Cumulative fraction',xlim=(.001,1e5));axes[0,1].legend()
    rh=np.load(root/'rates/publication_count_per_second_histogram.npy');axes[1,0].plot(np.arange(len(rh)),np.cumsum(rh)/rh.sum())
    axes[1,0].set(xlabel='Completed publications / wall second',ylabel='Fraction of seconds',xlim=(0,1000));axes[1,0].axvline(rates['quantiles']['0.5'],ls='--',color='red',label='Median '+str(rates['quantiles']['0.5']));axes[1,0].legend()
    day=pd.DataFrame(rates['days']);day=day[day.seconds>0]
    axes[1,1].plot(pd.to_datetime(day.date),day['median'],label='Daily median');axes[1,1].plot(pd.to_datetime(day.date),day['mean'],alpha=.6,label='Daily mean')
    axes[1,1].set(xlabel='UTC date',ylabel='Publications / wall second');axes[1,1].legend()
    for a in axes.flat:a.grid(alpha=.2)
    fig.suptitle('MBO is bursty: event-weighted cadence is not wall-clock frequency\nLondon–New York / known spot-reference seconds; source health is assessed separately')
    fig.tight_layout();fig.savefig(root/'cadence.png',dpi=150);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');a=p.parse_args();report(a.root)
