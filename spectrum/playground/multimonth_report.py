from pathlib import Path
import argparse,json
from spectrum.finalized import read_json
from .multimonth_evaluate import OUTPUT


def build(output=OUTPUT):
    root=Path(output);payload=read_json(root/'summary.json');payload['protocol']=read_json(root/'protocol.json')
    template=Path(__file__).with_name('multimonth_report.html').read_text(encoding='utf8')
    (root/'index.html').write_text(template.replace('__PAYLOAD__',json.dumps(payload,ensure_ascii=False,allow_nan=False).replace('</','<\\/')),encoding='utf8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default=OUTPUT);a=p.parse_args();build(a.output)
