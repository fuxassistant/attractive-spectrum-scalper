"""Bind numerical, source, visual, and test evidence of the weekly comparison."""
import json
from pathlib import Path
import re
import subprocess
import sys
from datetime import datetime,timezone
from scripts.flow_context_coverage import save,sha
from scripts.vwap_week_source import REPORT


def main():
    read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
    def check(p,h):
        if sha(p)!=h:raise ValueError('Changed artifact '+str(p))
    s=read(REPORT/'summary.json');v=read(REPORT/'visual_receipt.json');src=read(REPORT/'source_receipt.json')
    assert s['status']=='COMPLETE' and src['status']=='PASS'
    check(REPORT/'source_receipt.json',s['source_receipt_sha256'])
    check('scripts/vwap_week_source.py',src['script_sha256'])
    check('contracts/vwap_bands_week_v1.json',s['contracts_sha256'])
    for p,h in s['code_sha256'].items():check(p,h)
    for r in s['files']:check(r['file'],r['sha256'])
    for r in src['captures']:check(r['file'],r['sha256'])
    for name,h in v['images'].items():check(REPORT/name,h)
    check('scripts/vwap_week_report.py',v['script_sha256'])
    for name,key in [('summary.json','summary_sha256'),('comparison_diagnostics.json','diagnostics_sha256'),('index.html','page_sha256')]:check(REPORT/name,v[key])
    audit=REPORT/'SOL61_VISUAL_AUDIT_RU.md'
    assert sha(REPORT/'visual_receipt.json') in audit.read_text(encoding='utf-8')
    for target in re.findall(r'(?:href|src)="([^"#]+)"',(REPORT/'index.html').read_text(encoding='utf-8')):
        assert (REPORT/target).is_file(),target
    old=subprocess.check_output(['git','diff','6997552','--','spectrum/cognition/bb_crossing.py','scripts/bb_crossing_build.py','reports/bb_crossing_bank_v1','contracts/bb_crossing_bank_v1.json'])
    assert not old,'Previous BB experiment changed'
    result=subprocess.run([sys.executable,'-m','pytest','tests/test_vwap_compare.py','tests/test_bb_crossing.py','-q',
        '--basetemp=.tmp/pytest_vwap_receipt'],capture_output=True,text=True)
    log=result.stdout+result.stderr
    if result.returncode or not re.search(r'12 passed',log):raise ValueError(log)
    save(REPORT/'completion_receipt.json',dict(status='COMPLETE_VERIFIED_DESCRIPTIVE_WEEK',completed_utc=datetime.now(timezone.utc).isoformat(),
        dates=s['dates'],tests=log.strip(),raw_member_sha256=src['source_member_sha256'],
        raw_trades=sum(r['rows'] for r in src['captures']),canonical_volume_checks=sum(r['canonical_volume_checks'] for r in src['captures']),
        max_volume_error=max(r['max_volume_error'] for r in src['captures']),direct_vwap_checks=s['independent_checks'],
        max_direct_price_error=s['max_direct_error'],future_prefix_checks=s['independent_checks'],
        previous_bb_experiment_unchanged=True,visual_images=len(v['images']),
        report_sha256={p.name:sha(p) for p in REPORT.iterdir() if p.suffix in ['.json','.html','.md','.png'] and p.name!='completion_receipt.json'},
        code_sha256={str(p):sha(p) for p in Path('scripts').glob('vwap_week*.py')},
        scope='Five-day geometry comparison; no model fitting or economic admission; crossings measured at one-second resolution'))
    print(json.dumps(dict(status='COMPLETE_VERIFIED_DESCRIPTIVE_WEEK',tests=12,visual_images=len(v['images']))),flush=True)


if __name__=='__main__':main()
