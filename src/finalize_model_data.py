"""Purge held-out rows that collide after float32 preprocessing; never refit train."""
import argparse,hashlib,json,sys
from pathlib import Path
parser=argparse.ArgumentParser()
parser.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[1])
parser.add_argument('--work-dir',type=Path,required=True)
parser.add_argument('--dependency-dir',type=Path)
args=parser.parse_args()
if args.dependency_dir:sys.path.insert(0,str(args.dependency_dir.resolve()))
import duckdb
import pandas as pd
repo=args.repo.resolve();work=args.work_dir.resolve();work.mkdir(parents=True,exist_ok=True)
def lit(x):return "'"+str(x).replace("'","''")+"'"
def q(x):return '"'+x.replace('"','""')+'"'
def sha(p):
 with Path(p).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
f=json.loads((repo/'docs/feature_policy.json').read_text(encoding='utf8'))
meta=['row_index','row_id','group_id','label_raw','y_binary','attack_family','family_id','is_attempted','capture_date']
summary={}
for v in ['original','corrected']:
 for protocol in ['stratified','temporal']:
  d=repo/'data/split'/v/protocol
  m=json.loads((d/'preprocess_manifest.json').read_text(encoding='utf8'))
  if 'precision_guard' in m:raise ValueError('Already finalized; rebuild base model data before rerunning')
  state=json.loads((d/'preprocessor.json').read_text(encoding='utf8'))
  if sha(d/'preprocessor.json')!=m['preprocessor_sha256']:raise ValueError('State changed')
  db=duckdb.connect(str(work/f'{v}_{protocol}.duckdb'))
  db.execute("SET memory_limit='1GB'");db.execute('SET threads=1')
  db.execute('SET preserve_insertion_order=false')
  db.execute(f'SET temp_directory={lit((work/f"{v}_{protocol}_tmp").as_posix())}')
  union=' UNION ALL '.join(f'SELECT *,{i} AS split_code FROM read_parquet({lit((d/f"{s}.parquet").as_posix())})' for i,s in enumerate(['train','val','test']))
  db.execute('CREATE OR REPLACE VIEW model_rows AS '+union)
  serial='to_json(struct_pack('+','.join(f'f{i:02d}:=CASE WHEN {q(c)}=0 THEN 0.0::FLOAT ELSE {q(c)} END' for i,c in enumerate(m['ablation_features']))+'))::VARCHAR'
  db.execute(f'CREATE OR REPLACE TABLE model_keys AS SELECT row_index,split_code,{serial} AS canonical FROM model_rows')
  db.execute('ALTER TABLE model_keys ADD COLUMN model_group_id VARCHAR')
  db.execute('UPDATE model_keys SET model_group_id=sha256(canonical)')
  db.execute('CREATE OR REPLACE TABLE key_counts AS SELECT model_group_id,count(*) AS n,count(DISTINCT split_code) AS folds,min(split_code) AS first_code FROM model_keys GROUP BY model_group_id')
  collisions=db.execute('SELECT count(*) FROM (SELECT model_group_id FROM model_keys JOIN key_counts USING(model_group_id) WHERE n>1 GROUP BY model_group_id HAVING min(canonical)<>max(canonical))').fetchone()[0]
  if collisions:raise ValueError('Hash collision in transformed feature keys')
  db.execute('ALTER TABLE model_keys DROP COLUMN canonical')
  before=db.execute('SELECT count(*) FROM key_counts WHERE folds>1').fetchone()[0]
  index_union=' UNION ALL '.join(f'SELECT *,{i} AS split_code FROM read_parquet({lit((d/f"{s}_indices.parquet").as_posix())})' for i,s in enumerate(['train','val','test']))
  db.execute('CREATE OR REPLACE VIEW base_indices AS '+index_union)
  db.execute('CREATE OR REPLACE TABLE decisions AS SELECT b.*,k.model_group_id,b.split_code>c.first_code AS precision_purged FROM base_indices b JOIN model_keys k USING(row_index,split_code) JOIN key_counts c USING(model_group_id)')
  purge=d/'precision_purged_indices.parquet'
  selected=','.join(q(c) for c in meta)
  db.execute(f"COPY (SELECT {selected},model_group_id,CASE split_code WHEN 0 THEN 'train' WHEN 1 THEN 'val' ELSE 'test' END AS intended_split,'same_float32_75_features_in_earlier_split' AS reason FROM decisions WHERE precision_purged ORDER BY row_index) TO {lit(purge.as_posix())} (FORMAT PARQUET,COMPRESSION ZSTD)")
  removed={s:db.execute('SELECT count(*) FROM decisions WHERE precision_purged AND split_code=?',[i]).fetchone()[0] for i,s in enumerate(['train','val','test'])}
  effective={}
  for i,s in enumerate(['train','val','test']):
   ip=d/f'model_{s}_indices.parquet'
   db.execute(f'COPY (SELECT {selected},model_group_id FROM decisions WHERE NOT precision_purged AND split_code={i} ORDER BY row_index) TO {lit(ip.as_posix())} (FORMAT PARQUET,COMPRESSION ZSTD)')
   n=m['output'][s]['rows']-removed[s]
   effective[s]=dict(path=ip.relative_to(repo).as_posix(),rows=n,sha256=sha(ip))
   if removed[s]:
    for variant,basepath in [('main',d/f'{s}.parquet'),('no_destination_port',d/'no_destination_port'/f'{s}.parquet')]:
     stage=work/f'{v}_{protocol}_{s}_{variant}_final.parquet'
     db.execute(f'COPY (SELECT x.* FROM read_parquet({lit(basepath.as_posix())}) x JOIN read_parquet({lit(ip.as_posix())}) i USING(row_index) ORDER BY x.row_index) TO {lit(stage.as_posix())} (FORMAT PARQUET,COMPRESSION ZSTD)')
     stage.replace(basepath)
     m['output'][s][variant].update(sha256=sha(basepath),bytes=basepath.stat().st_size)
   m['output'][s]['base_rows_before_precision_guard']=m['output'][s]['rows']
   m['output'][s]['rows']=n
   m['output'][s]['effective_index']=effective[s]
   m['output'][s]['precision_purged_rows']=removed[s]
   # Recompute diagnostics on the retained subset, still using the original train vocabulary.
   old_diag=m['output'][s]['port_diagnostics'];m['output'][s]['base_port_diagnostics']=old_diag
   port=f['feature_sources'][v]['Destination Port']
   known=pd.DataFrame({'known_port':[int(x) for x in state['observed_port_counts']]})
   db.register('observed_ports',known)
   unknown=db.execute(f'''SELECT count(*) FROM read_parquet({lit((repo/'data/clean'/f'{v}.parquet').as_posix())},file_row_number=true) x
     JOIN read_parquet({lit(ip.as_posix())}) i ON x.file_row_number=i.row_index
     LEFT JOIN observed_ports p ON x.{q(port)}=p.known_port WHERE p.known_port IS NULL''').fetchone()[0]
   other=db.execute(f'SELECT sum("DestinationPort__OTHER")::BIGINT FROM read_parquet({lit((d/f"{s}.parquet").as_posix())})').fetchone()[0]
   m['output'][s]['port_diagnostics']=dict(other_rows=other,unseen_in_train_rows=unknown)
  after=db.execute('SELECT count(*) FROM (SELECT model_group_id FROM decisions WHERE NOT precision_purged GROUP BY model_group_id HAVING count(DISTINCT split_code)>1)').fetchone()[0]
  accounted=db.execute('SELECT count(*),count(DISTINCT row_index) FROM decisions').fetchone()
  base_total=sum(x['base_rows_before_precision_guard'] for x in m['output'].values())
  checks=dict(no_transformed_group_overlap=after==0,all_base_rows_accounted_for=accounted[0]==accounted[1]==base_total,train_unchanged=removed['train']==0,processor_unchanged=sha(d/'preprocessor.json')==m['preprocessor_sha256'],no_hash_collisions=collisions==0)
  if not all(checks.values()):raise ValueError(checks)
  distribution=[dict(split=['train','val','test'][code],precision_purged=purged,label_raw=label,y_binary=y,attack_family=family,count=n) for code,purged,label,y,family,n in db.execute('SELECT split_code,precision_purged,label_raw,y_binary,attack_family,count(*) FROM decisions GROUP BY ALL ORDER BY ALL').fetchall()]
  m['precision_guard']=dict(rule_version='1.0',features='75 standardized float32 numeric fields; signed zero normalized',priority=['train','val','test'],cross_split_groups_before=before,cross_split_groups_after=after,removed_rows=removed,indices=effective,purged_path=purge.relative_to(repo).as_posix(),purged_sha256=sha(purge),checks=checks,class_distribution=distribution,postprocessor_sha256=sha(Path(__file__)))
  m['status']='precision_guard_complete_pending_independent_verification'
  # Discard outdated certification from the initial pre-guard verification.
  m.pop('verification_report_sha256',None)
  dump(d/'preprocess_manifest.json',m)
  summary[f'{v}/{protocol}']=m
  dump(repo/'docs/preprocess_results.json',summary)
  print(v,protocol,'precision groups',before,'->',after,'removed',removed,flush=True)
  db.close()
dump(repo/'docs/preprocess_results.json',summary)
