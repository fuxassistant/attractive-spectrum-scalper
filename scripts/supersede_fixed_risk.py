"""Preserve historical evidence while withdrawing the incorrect risk interpretation."""
from pathlib import Path
from datetime import datetime, timezone
import json
import hashlib


def main():
    base=Path(__file__).resolve().parents[1]/'reports'/'player_v3'
    reason='User clarified that loss MORE THAN 50% applies to a martingale SERIES relative to capital before that series, including accumulated profits; fixed initial100USD floor was incorrect.'
    for name in ('run_001','run_002','run_003','martingale_001','martingale_002'):
        root=base/name
        if not root.is_dir():continue
        receipt=dict(status='SUPERSEDED_WRONG_RISK_INTERPRETATION',reason=reason,
                     superseded_utc=datetime.now(timezone.utc).isoformat(),replacement_neural='series_001',
                     replacement_martingale='martingale_series_001',historical_data_retained=True)
        page=root/'index.html'
        if page.exists():
            backup=root/'index.fixed_risk_original.html'
            if not backup.exists():backup.write_bytes(page.read_bytes())
            receipt['original_html_sha256']=hashlib.sha256(backup.read_bytes()).hexdigest()
            banner='<div style="padding:20px;background:#722f34;color:white;font:16px system-ui"><b>АРХИВ: НЕВЕРНОЕ ПРАВИЛО РИСКА.</b> Здесь использован постоянный порог $100. Пользователь уточнил: потеря больше 50% относится к серии мартингейла и капиталу перед её началом. Экономические выводы этого прогона не отвечают уточнённой задаче. <a style="color:white" href="../series_001/index.html">Исправленный эксперимент</a></div>'
            page.write_text(backup.read_text(encoding='utf8').replace('<body>','<body>'+banner,1),encoding='utf8',newline='\n')
            receipt['annotated_html_sha256']=hashlib.sha256(page.read_bytes()).hexdigest()
        (root/'risk_superseded.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf8',newline='\n')
    latest=base/'LATEST.json'
    old=json.loads(latest.read_text()) if latest.exists() else {}
    payload=dict(status='REDO_IN_PROGRESS',reason=reason,prior_pointer=old,
                 replacement_neural='series_001',replacement_martingale='martingale_series_001')
    latest.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf8',newline='\n')


if __name__=='__main__':main()
