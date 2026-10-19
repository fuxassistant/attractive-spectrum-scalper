"""Run a basket recipe without parameter selection; preserve input provenance."""
from pathlib import Path
import argparse
from spectrum.finalized import read_json
from spectrum.core.storage import sha
from spectrum.ingest import save_json
from .basket import BasketConfig
from .basket_search import evaluate_record,DEFAULT_OUTPUT
from .runner import DEFAULT_DATA


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',default=DEFAULT_DATA)
    p.add_argument('--recipe',default=DEFAULT_OUTPUT+'/frozen.json')
    p.add_argument('--split',choices=['train','validation','holdout','month'],default='month')
    p.add_argument('--output',required=True);a=p.parse_args()
    target=Path(a.output);target.mkdir(parents=True,exist_ok=False)
    recipe=read_json(a.recipe);c=BasketConfig(**recipe.get('config',recipe));manifest=read_json(Path(a.data)/'manifest.json')
    dates=[d['date'] for d in manifest['days']] if a.split=='month' else manifest['split'][a.split]
    save_json(target/'run_contract.json',dict(recipe=str(Path(a.recipe).resolve()),recipe_sha256=sha(a.recipe),
        split=a.split,data_sha256=sha(Path(a.data)/'manifest.json'),initial_capital=1500,
        kernel_sha256=sha(Path(__file__).with_name('basket.py')),selection='No selection in this command'))
    result=evaluate_record(a.data,dates,c,target);print(result['result'])
