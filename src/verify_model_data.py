"""Audit train-only statistics independently, then scan all exported model rows."""
import argparse,hashlib,json,sys
from pathlib import Path
parser=argparse.ArgumentParser()
parser.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[1])
parser.add_argument('--dependency-dir',type=Path)
args=parser.parse_args()
if args.dependency_dir:sys.path.insert(0,str(args.dependency_dir.resolve()))
import duckdb
import numpy as np
from preprocess import FlowPreprocessor,METADATA_COLUMNS
repo=args.repo.resolve()
f=json.loads((repo/'docs/feature_policy.json').read_text(encoding='utf8'))
results=[]
def lit(s):return "'"+str(s).replace("'","''")+"'"
def q(s):return '"'+s.replace('"','""')+'"'
def sha(p):
    with Path(p).open('rb') as s:return hashlib.file_digest(s,'sha256').hexdigest()
def connect():
    c=duckdb.connect();c.execute("SET memory_limit='768MB'");c.execute('SET threads=2');return c
statscon=connect()
for v in ['original','corrected']:
 source=repo/'data/clean'/f'{v}.parquet'
 sources=f['feature_sources'][v]
 for protocol in ['stratified','temporal']:
  d=repo/'data/split'/v/protocol
  m=json.loads((d/'preprocess_manifest.json').read_text(encoding='utf8'))
  index_manifest=json.loads((d/'manifest.json').read_text(encoding='utf8'))
  proc=FlowPreprocessor.load(d/'preprocessor.json');state=proc.state
  checks={'source_hash':sha(source)==m['input_sha256'],
          'state_hash':sha(d/'preprocessor.json')==m['preprocessor_sha256'],
          'module_hash':sha(repo/'src/preprocess.py')==m['preprocess_module_sha256'],
          'builder_hash':sha(repo/'src/build_model_data.py')==m['builder_sha256'],
          'index_manifest_hash':sha(d/'manifest.json')==m['index_manifest_sha256'],
          'fit_scope_train':state['fit_split']=='train' and state['fit_rows']==index_manifest['indices']['train']['rows']}
  if 'precision_guard' not in m:raise ValueError('Model-precision overlap guard has not been run')
  checks['precision_guard_script_hash']=sha(repo/'src/finalize_model_data.py')==m['precision_guard']['postprocessor_sha256']
  checks['precision_purged_hash']=sha(repo/m['precision_guard']['purged_path'])==m['precision_guard']['purged_sha256']
  train_index=d/'train_indices.parquet'
  statscon.execute(f'''CREATE OR REPLACE VIEW train AS SELECT s.*,i.y_binary,i.row_index FROM
      read_parquet({lit(source.as_posix())},file_row_number=true) s
      JOIN read_parquet({lit(train_index.as_posix())}) i ON s.file_row_number=i.row_index''')
  cols=[sources[c] for c in proc.numeric_columns]
  expr=','.join(z for c in cols for z in [f'avg({q(c)}::DOUBLE)',f'var_pop({q(c)}::DOUBLE)'])
  stats=np.asarray(statscon.execute('SELECT '+expr+' FROM train').fetchone(),dtype=np.float64).reshape(-1,2)
  checks['mean_from_train_only']=bool(np.allclose(stats[:,0],state['mean'],rtol=1e-9,atol=1e-6))
  checks['variance_from_train_only']=bool(np.allclose(stats[:,1],state['variance'],rtol=1e-8,atol=1e-7))
  expected_scale=np.sqrt(np.maximum(stats[:,1],0));expected_scale[expected_scale==0]=1
  checks['scale_from_train_only']=bool(np.allclose(expected_scale,state['scale'],rtol=1e-8,atol=1e-7))
  port=sources['Destination Port']
  port_counts=statscon.execute(f'SELECT {q(port)}::BIGINT,count(*) FROM train GROUP BY ALL ORDER BY count(*) DESC,{q(port)}::BIGINT').fetchall()
  checks['port_vocabulary_from_train_only']=[int(p) for p,n in port_counts[:64]]==state['port_vocabulary']
  checks['port_counts_from_train_only']={str(p):n for p,n in port_counts}==state['observed_port_counts']
  counts=dict(statscon.execute('SELECT y_binary,count(*) FROM train GROUP BY y_binary').fetchall())
  n=sum(counts.values())
  checks['class_weights_from_train_only']=all(state['class_counts'][str(c)]==counts[c] and np.isclose(state['class_weights'][str(c)],n/(2*counts[c]),rtol=0,atol=1e-14) for c in [0,1])
  indices=statscon.execute('SELECT row_index FROM train ORDER BY row_index').fetchnumpy()['row_index']
  checks['fit_row_digest']=hashlib.sha256(indices.astype('<i8',copy=False).tobytes()).hexdigest()==state['fit_row_indices_sha256']
  checks['provenance_train_index_hash']=sha(train_index)==state['provenance']['train_index_sha256']
  del indices
  outputs={}
  for split in ['train','val','test']:
   main=repo/m['output'][split]['main']['path'];abl=repo/m['output'][split]['no_destination_port']['path']
   idx=repo/m['output'][split]['effective_index']['path']
   local={'main_file_hash':sha(main)==m['output'][split]['main']['sha256'],
          'ablation_file_hash':sha(abl)==m['output'][split]['no_destination_port']['sha256'],
          'effective_index_hash':sha(idx)==m['output'][split]['effective_index']['sha256']}
   c1,c2,c3=connect(),connect(),connect()
   expected_main=proc.feature_names()+METADATA_COLUMNS
   expected_abl=proc.feature_names(False)+METADATA_COLUMNS
   desc1=c1.execute(f'DESCRIBE SELECT * FROM read_parquet({lit(main.as_posix())})').fetchall()
   desc2=c2.execute(f'DESCRIBE SELECT * FROM read_parquet({lit(abl.as_posix())})').fetchall()
   local['exact_feature_and_metadata_schema']=[r[0] for r in desc1]==expected_main and [r[0] for r in desc2]==expected_abl
   local['float32_features']=all(t=='FLOAT' for name,t,*_ in desc1[:len(proc.feature_names())])
   c1.execute(f'SELECT * FROM read_parquet({lit(main.as_posix())}) ORDER BY row_index')
   c2.execute(f'SELECT * FROM read_parquet({lit(abl.as_posix())}) ORDER BY row_index')
   c3.execute(f'SELECT * FROM read_parquet({lit(idx.as_posix())}) ORDER BY row_index')
   rows=0;finite=True;onehot=True;metadata=True;ablation=True
   while True:
    a,b,i=c1.fetch_df_chunk(8),c2.fetch_df_chunk(8),c3.fetch_df_chunk(8)
    if a.empty:
     if not b.empty or not i.empty:metadata=False
     break
    if len(a)!=len(b) or len(a)!=len(i):raise ValueError('Streaming row alignment mismatch')
    arr=a.loc[:,proc.feature_names()].to_numpy(dtype=np.float32)
    finite=finite and bool(np.isfinite(arr).all())
    ports=arr[:,len(proc.numeric_columns):]
    onehot=onehot and bool(((ports==0)|(ports==1)).all() and (ports.sum(axis=1)==1).all())
    ablation=ablation and np.array_equal(arr[:,:len(proc.numeric_columns)],b.loc[:,proc.numeric_columns].to_numpy(dtype=np.float32))
    metadata=metadata and a[METADATA_COLUMNS].equals(i[METADATA_COLUMNS]) and b[METADATA_COLUMNS].equals(i[METADATA_COLUMNS])
    rows+=len(a)
   local.update(all_feature_values_finite=finite,one_hot_valid=onehot,metadata_matches_indices=metadata,ablation_is_exact_numeric_subset=bool(ablation),rows_match_indices=rows==m['output'][split]['effective_index']['rows'])
   for c in [c1,c2,c3]:c.close()
   # Recompute values from source for a deterministic sample in each fold.
   sample=statscon.execute(f'SELECT row_index FROM read_parquet({lit(idx.as_posix())}) USING SAMPLE reservoir(1024 ROWS) REPEATABLE(42)').fetchnumpy()['row_index']
   import pandas as pd
   statscon.register('sample_ids',pd.DataFrame({'sample_row':sample}))
   selections=','.join(f's.{q(sources[x])} AS {q(x)}' for x in f['canonical_feature_names'])
   raw=statscon.execute(f'''SELECT {selections},s.file_row_number AS row_index FROM read_parquet({lit(source.as_posix())},file_row_number=true) s JOIN sample_ids a ON s.file_row_number=a.sample_row ORDER BY row_index''').fetchdf()
   saved=statscon.execute(f'SELECT x.* FROM read_parquet({lit(main.as_posix())}) x JOIN sample_ids a ON x.row_index=a.sample_row ORDER BY row_index').fetchdf()
   local['sample_transform_matches_source']=bool(np.array_equal(proc.transform(raw),saved[proc.feature_names()].to_numpy(dtype=np.float32)))
   outputs[split]={'rows':rows,'checks':local,'passed':all(local.values())}
   print(v,protocol,split,'PASS' if all(local.values()) else local,flush=True)
  # Check the actual float32 inputs, not just original raw-feature group IDs.
  union=' UNION ALL '.join(f'SELECT *,{lit(s)} AS split FROM read_parquet({lit((d/"no_destination_port"/f"{s}.parquet").as_posix())})' for s in ['train','val','test'])
  statscon.execute('CREATE OR REPLACE VIEW effective_model AS '+union)
  h='hash('+','.join(q(c) for c in proc.numeric_columns)+')'
  overlaps=statscon.execute(f'SELECT count(*) FROM (SELECT {h} AS h FROM effective_model GROUP BY h HAVING count(DISTINCT split)>1)').fetchone()[0]
  checks['no_actual_float32_input_overlap']=overlaps==0
  all_effective=' UNION ALL '.join(f'SELECT row_index FROM read_parquet({lit((repo/m["output"][s]["effective_index"]["path"]).as_posix())})' for s in ['train','val','test'])
  purged_path=repo/m['precision_guard']['purged_path']
  statscon.execute('CREATE OR REPLACE VIEW precision_accounted AS '+all_effective+f' UNION ALL SELECT row_index FROM read_parquet({lit(purged_path.as_posix())})')
  n_all,n_unique=statscon.execute('SELECT count(*),count(DISTINCT row_index) FROM precision_accounted').fetchone()
  base_union=' UNION ALL '.join(f'SELECT row_index FROM read_parquet({lit((d/f"{s}_indices.parquet").as_posix())})' for s in ['train','val','test'])
  statscon.execute('CREATE OR REPLACE VIEW base_kept_rows AS '+base_union)
  mismatch=statscon.execute('SELECT count(*) FROM ((SELECT row_index FROM precision_accounted EXCEPT SELECT row_index FROM base_kept_rows) UNION ALL (SELECT row_index FROM base_kept_rows EXCEPT SELECT row_index FROM precision_accounted))').fetchone()[0]
  checks['precision_cleanup_accounts_for_every_base_row']=n_all==n_unique==sum(index_manifest['indices'][s]['rows'] for s in ['train','val','test']) and mismatch==0
  checks['precision_cleanup_keeps_training_rows']=m['precision_guard']['removed_rows']['train']==0
  if protocol=='stratified':
   checks['effective_random_all_families_covered']=statscon.execute('SELECT count(*) FROM (SELECT split,attack_family FROM effective_model GROUP BY ALL)').fetchone()[0]==45
  r=dict(version=v,protocol=protocol,train_fit_checks=checks,output_checks=outputs,passed=all(checks.values()) and all(x['passed'] for x in outputs.values()))
  results.append(r)
output=dict(passed=all(r['passed'] for r in results),results=results)
(repo/'docs/preprocess_verification.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
if not output['passed']:
 print(json.dumps(output,ensure_ascii=False,indent=2));sys.exit(1)
print('All train-only and full-output verification checks passed',flush=True)
