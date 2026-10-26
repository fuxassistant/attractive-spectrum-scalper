"""Build one crossing bank per trading day from canonical quote captures."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq
from spectrum.io import save_json, sha
from spectrum.market.quotes import QuoteColumns
from .bank import geometry_definition, source_identity
from .store import publish_bank

COLUMNS = ['available_ns', 'bid_nano', 'ask_nano', 'valid', 'epoch']
EPOCH_STRIDE = 1_000_000  # parts of one day never share a band history


def day_quotes(meta):
    """Quotes of one day: every quote part restricted to its [start_ns, end_ns) window."""
    parts = []
    for number, part in enumerate(meta['quote_parts']):
        path = Path(part['root']) / 'quotes.parquet'
        if sha(path) != part['quote_file']['sha256']:
            raise ValueError('Quote capture changed: ' + str(path))
        start, end = int(part['start_ns']), int(part['end_ns'])
        table = pq.read_table(path, columns=COLUMNS,
                              filters=[('available_ns', '>=', start), ('available_ns', '<', end)]).to_pandas()
        table['epoch'] = table.epoch + number * EPOCH_STRIDE
        parts.append(table)
    frame = pd.concat(parts, ignore_index=True)
    return QuoteColumns.from_frame(frame), max(int(p['end_ns']) for p in meta['quote_parts'])


def build_day(day_manifest, output):
    meta = json.loads(Path(day_manifest).read_text(encoding='utf-8'))
    quotes, end = day_quotes(meta)
    return publish_bank(quotes, Path(output) / meta['date'], source_id=source_identity(meta),
                        coverage_end_ns=end, meta=dict(date=meta['date'], symbol=meta['symbol'],
                                                       day_manifest_sha256=sha(day_manifest)))


def build_banks(days_root, output, workers=4):
    """Every */*/manifest.json under days_root (e.g. basket_multimonth_v1) becomes one bank."""
    days = sorted(Path(days_root).glob('*/*/manifest.json'))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs = {pool.submit(build_day, day, output): day for day in days}
        for job in as_completed(jobs):
            result = job.result()
            results.append(dict(date=result['meta']['date'], events=result['events'], quotes=result['quotes']))
            print(json.dumps(results[-1]), flush=True)
    results.sort(key=lambda r: r['date'])
    summary = dict(status='COMPLETE', days=len(results), events=sum(r['events'] for r in results),
                   geometry=geometry_definition(), source=str(days_root), entries=results)
    save_json(output / 'manifest.json', summary)
    return summary
