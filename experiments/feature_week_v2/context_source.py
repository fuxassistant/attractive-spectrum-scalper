"""Extract the missing Thursday needed for Friday's complete CME session."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'feature_week_v1'))
import extract_context

if __name__=='__main__':
    extract_context.OUTPUT=extract_context.ROOT/'data/derived/feature_week_v2'
    print(extract_context.extract('2026-08-20'),flush=True)
