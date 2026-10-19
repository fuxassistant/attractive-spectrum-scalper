"""Read-only source review. Preserve bounded evidence; never deletes anything."""
from pathlib import Path
import hashlib,json,zipfile,subprocess
import pyarrow.parquet as pq

SOURCE=Path('D:/scalper-mini/poi-scalper')
OUTPUT=Path('reports/poi_research_review_20261018')


def selected(name,size):
    n=name.lower();ext=Path(n).suffix
    return ((ext in ('.md','.py','.txt','.toml','.yaml','.yml','.json') and size<=2*1024**2)
        or (ext=='.csv' and size<=512*1024)
        or (ext=='.png' and size<=512*1024 and any(x in n for x in ['report','analysis','audit','figure','atlas','error'])))


def main():
    inventory=json.loads((OUTPUT/'inventory.json').read_text())
    catalogs=json.loads((OUTPUT/'zip_catalog.json').read_text())
    entries=[];blobs=set();errors=[];total=0
    capsule=OUTPUT/'research_evidence.zip'
    def store(z,content,origin,archive=None,crc=None):
        nonlocal total
        digest=hashlib.sha256(content).hexdigest();key='blobs/'+digest
        if key not in blobs:z.writestr(key,content);blobs.add(key);total+=len(content)
        entries.append(dict(origin=origin,archive=archive,crc32=crc,bytes=len(content),sha256=digest,member=key))
    with zipfile.ZipFile(capsule,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for f in inventory:
            if selected(f['path'],f['bytes']):
                try:store(z,(SOURCE/f['path']).read_bytes(),f['path'])
                except Exception as e:errors.append(dict(path=f['path'],error=str(e)))
        for cat in catalogs:
            if 'entries' not in cat:continue
            try:
                with zipfile.ZipFile(SOURCE/cat['path']) as old:
                    for e in cat['entries']:
                        if selected(e['name'],e['bytes']):
                            store(z,old.read(e['name']),e['name'],cat['path'],e['crc'])
            except Exception as e:errors.append(dict(path=cat['path'],error=str(e)))
        git=subprocess.check_output(['git','-c','safe.directory='+SOURCE.as_posix(),'-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip()
        index=dict(version='POI_RESEARCH_EVIDENCE_CAPSULE_V1',source=str(SOURCE),head=git,entries=entries,
            errors=errors,policy='Text/code/JSON <=2MiB; CSV <=512KiB; selected report PNG <=512KiB. Large datasets/models and HTML viewers are omitted, not backed up.',
            unique_blobs=len(blobs),unique_bytes=total)
        z.writestr('index.json',json.dumps(index,ensure_ascii=False,indent=2))
    with zipfile.ZipFile(capsule) as z:
        assert z.testzip() is None
        for name in blobs:
            assert hashlib.sha256(z.read(name)).hexdigest()==name.split('/')[-1]
    (OUTPUT/'capsule_index.json').write_text(json.dumps(index,ensure_ascii=False,indent=2),encoding='utf8')
    receipt=dict(status='VERIFIED_SELECTED_EVIDENCE',capsule=str(capsule),bytes=capsule.stat().st_size,
        sha256=hashlib.sha256(capsule.read_bytes()).hexdigest(),source_references=len(entries),unique_blobs=len(blobs),
        unique_bytes=total,errors=errors,scope='Every stored blob SHA and capsule CRC verified. Source ZIP CRC checked only for selected members. Full large ZIP archives not rehashed.')
    (OUTPUT/'capsule_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf8')
    # Candidate A: old pre-fix binary payloads; all metadata stays in place.
    prefixes=['data/archive/price_flow_probe_pre_'+s+'/' for s in ['delivery_bootstrap_20261005','gap_flush_20261005','stock_mask_20261005','occupancy_20261005','queue_reactions_20261005']]
    prefixes+=['data/research/data_lab_v1/'+s+'/' for s in ['four_weeks_20261011','four_weeks_20261011_r2']]
    plan=[]
    for f in inventory:
        if any(f['path'].startswith(p) for p in prefixes) and Path(f['path']).suffix in ('.parquet','.npy','.npz','.bin','.arrow'):
            row=dict(f,decision='PROPOSE_DELETE_DERIVED_PAYLOAD_AFTER_APPROVAL',sha256=None)
            if f['path'].endswith('.parquet'):
                try:
                    m=pq.ParquetFile(SOURCE/f['path']);row.update(footer='READABLE',rows=m.metadata.num_rows)
                except Exception as e:row.update(footer='UNREADABLE',error=str(e))
            plan.append(row)
    (OUTPUT/'proposed_payload_removals.json').write_text(json.dumps(dict(status='PROPOSED_NOT_AUTHORIZED_OR_EXECUTED',
        files=plan,bytes=sum(p['bytes'] for p in plan),retain='All JSON receipts/inventory/checkpoints, code, source archives, and current canonical datasets remain untouched'),indent=2),encoding='utf8')
    print(json.dumps(receipt));print('PROPOSED PAYLOAD GiB',sum(p['bytes'] for p in plan)/1024**3)


if __name__=='__main__':main()
