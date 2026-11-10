"""Single authorized Validation-only timing-list preparation under guarded GO."""
import sys
from pathlib import Path
import json

if sys.stdin.readline().strip() != 'GO':
    raise RuntimeError('Guarded GO missing')
if '--guard-probe' in sys.argv:
    print('GUARDED_MODULE_PROBE_OK; no data imports or reads', flush=True)
    raise SystemExit(0)

from methods.m1.search_gbdt import ROOT,read,write,sha,require
from methods.m1.seal_gbdt import artifact_paths,file_inventory,freeze_files
from methods.m1.evaluation import timing_indices
from methods.m1.data import row_digest
import numpy as np

run = Path(sys.argv[1])
archive = ROOT/'results/m1_gbdt_search/20261101T115824-90de9307e77448aba412cc9ec08a5cbd'
manifest_sha = '00128ffea5088c939a2375502372be4ba6b95cc13bcdee4f68e6c1638d2b1d6d'
with freeze_files([*artifact_paths(archive).values(),archive/'MANIFEST.json']):
    require(sha(archive/'MANIFEST.json')==manifest_sha,'GBDT manifest changed')
    manifest=read(archive/'MANIFEST.json')
    require(file_inventory(archive)==manifest['files'],'GBDT original hash mismatch')
    require(sha(archive/'selection.json')=='859a883fe6ecc12246fb2f08c5ab3e2707530a6906f52c1c2863d9b12ba0d2c9','Selection changed')
    job=read(archive/'completed/G05_seed_42.json')
    prediction=archive/job['artifact_directory']/'validation.npz'
    require(sha(prediction)==job['files']['validation.npz']['sha256'],'Validation prediction changed')
    with np.load(prediction,allow_pickle=False) as saved:
        ids,y=saved['row_ids'],saved['y_true']
        require(len(ids)==len(y)==367036 and len(set(ids))==len(ids),'Invalid Validation population')
        require(row_digest(ids)==job['validation_row_ids_sha256'],'Row ID digest changed')
        chosen=timing_indices(ids,batch_size=512,seed=2026)
        document={'source_split':'val','population_row_ids_sha256':row_digest(ids),
                  'row_ids':ids[chosen].tolist(),'selection':'sha256_row_id_rank','seed':2026,
                  'purpose':'User-approved single common CPU timing list for P4/P5/P6/P7 reuse; no class balancing.',
                  'provenance':{'manifest_sha256':manifest_sha,'prediction_sha256':sha(prediction),
                                'test_accessed':False,'training_fits':0},
                  'benign_rows':int((y[chosen]==0).sum()),'attack_rows':int((y[chosen]==1).sum())}
    write(run/'timing_rows.json',document)
    require(file_inventory(archive)==manifest['files'],'Archive changed during ID preparation')
write(run/'worker_result.json',{'status':'timing_ids_prepared','row_count':512,
      'timing_file_sha256':sha(run/'timing_rows.json'),'test_accessed':False,'training_fits':0})
print('Validation timing IDs prepared; no Test read.',flush=True)
