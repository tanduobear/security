"""Independently verify exported index files against manifests and source rows."""
import argparse, hashlib, json, sys
from pathlib import Path
parser=argparse.ArgumentParser()
parser.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[1])
parser.add_argument('--dependency-dir',type=Path)
args=parser.parse_args()
if args.dependency_dir:sys.path.insert(0,str(args.dependency_dir.resolve()))
import duckdb
repo=args.repo.resolve()
con=duckdb.connect()
con.execute("SET memory_limit='1GB'")
con.execute('SET threads=4')
policy=json.loads((repo/'docs/label_split_policy.json').read_text(encoding='utf8'))
def lit(s):return "'"+str(s).replace("'","''")+"'"
def sha(p):
 with p.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
results=[]
for version in ['original','corrected']:
 source=repo/'data/clean'/f'{version}.parquet'
 con.execute(f'CREATE OR REPLACE VIEW source AS SELECT * FROM read_parquet({lit(source.as_posix())}, file_row_number=true)')
 date_expr="CASE Day WHEN 'Monday' THEN DATE '2017-07-03' WHEN 'Tuesday' THEN DATE '2017-07-04' WHEN 'Wednesday' THEN DATE '2017-07-05' WHEN 'Thursday' THEN DATE '2017-07-06' WHEN 'Friday' THEN DATE '2017-07-07' END" if version=='original' else 'try_cast(Timestamp AS TIMESTAMP)::DATE'
 mapping=[r for r in policy['label_mapping'] if r['version']==version]
 def expr(key):return 'CASE i.label_raw '+ ' '.join('WHEN '+lit(r['label_raw'])+' THEN '+(lit(r[key]) if isinstance(r[key],str) else str(r[key]).lower()) for r in mapping)+' END'
 for protocol in ['stratified','temporal']:
  d=repo/'data/split'/version/protocol
  manifest=json.loads((d/'manifest.json').read_text(encoding='utf8'))
  checks={'source_hash':sha(source)==manifest['input_sha256'],
          'feature_policy_hash':sha(repo/'docs/feature_policy.json')==manifest['feature_policy_sha256'],
          'label_policy_hash':sha(repo/'docs/label_split_policy.json')==manifest['label_policy_sha256'],
          'builder_script_hash':sha(repo/'src/build_split_indices.py')==manifest['script_sha256']}
  sql=[]
  for split in ['train','val','test']:
   p=d/f'{split}_indices.parquet'
   checks[split+'_file_hash']=sha(p)==manifest['indices'][split]['sha256']
   n=con.execute(f'SELECT count(*) FROM read_parquet({lit(p.as_posix())})').fetchone()[0]
   checks[split+'_row_count']=n==manifest['indices'][split]['rows']
   sql.append(f'SELECT *,{lit(split)} AS split FROM read_parquet({lit(p.as_posix())})')
  con.execute('CREATE OR REPLACE VIEW kept AS '+ ' UNION ALL '.join(sql))
  purged=d/'purged_indices.parquet'
  checks['purged_file_hash']=sha(purged)==manifest['purged_index_sha256']
  checks['purged_row_count']=con.execute(f'SELECT count(*) FROM read_parquet({lit(purged.as_posix())})').fetchone()[0]==manifest['purged_rows']
  con.execute(f"CREATE OR REPLACE VIEW accounted AS SELECT * FROM kept UNION ALL SELECT row_index,row_id,group_id,label_raw,y_binary,attack_family,family_id,is_attempted,capture_date,'purged' AS split FROM read_parquet({lit(purged.as_posix())})")
  n,unique,lo,hi=con.execute('SELECT count(*),count(DISTINCT row_index),min(row_index),max(row_index) FROM accounted').fetchone()
  checks['every_source_row_exactly_once']=n==unique==manifest['input_rows'] and lo==0 and hi==n-1
  checks['no_row_overlap']=con.execute('SELECT count(*)-count(DISTINCT row_index) FROM kept').fetchone()[0]==0
  checks['no_group_overlap']=con.execute('SELECT count(*) FROM (SELECT group_id FROM kept GROUP BY group_id HAVING count(DISTINCT split)>1)').fetchone()[0]==0
  checks['source_labels_and_days']=con.execute(f'''SELECT count(*) FROM accounted i JOIN source ON i.row_index=source.file_row_number
      WHERE i.label_raw IS DISTINCT FROM source.Label OR i.capture_date IS DISTINCT FROM {date_expr}''').fetchone()[0]==0
  conditions=[f'i.{c} IS DISTINCT FROM {expr(c)}' for c in ['y_binary','family_id','is_attempted']]
  conditions += [f'i.attack_family IS DISTINCT FROM {expr("attack_family")}',f'i.row_id IS DISTINCT FROM {lit(version+":")}||i.row_index::VARCHAR']
  checks['exported_label_mapping_and_row_ids']=con.execute('SELECT count(*) FROM accounted i WHERE '+' OR '.join(conditions)).fetchone()[0]==0
  if protocol=='temporal':
   checks['temporal_boundaries']=con.execute("SELECT count(*) FROM kept WHERE (split='train' AND capture_date>DATE '2017-07-05') OR (split='val' AND capture_date<>DATE '2017-07-06') OR (split='test' AND capture_date<>DATE '2017-07-07')").fetchone()[0]==0
   checks['purged_group_has_earlier_kept_group']=con.execute(f'''SELECT count(*) FROM read_parquet({lit(purged.as_posix())}) p WHERE NOT EXISTS(SELECT 1 FROM kept k WHERE k.group_id=p.group_id AND k.capture_date<p.capture_date)''').fetchone()[0]==0
  else:
   checks['all_families_in_all_splits']=con.execute('SELECT count(*) FROM (SELECT split,attack_family FROM kept GROUP BY ALL)').fetchone()[0]==3*len(policy['family_names'])
   checks['ratio_within_one_percentage_point']=all(abs(manifest['indices'][s]['rows']/n-r)<.01 for s,r in zip(['train','val','test'],[.7,.15,.15]))
  r=dict(version=version,protocol=protocol,checks=checks,passed=all(checks.values()))
  results.append(r)
  print(version,protocol,'PASS' if r['passed'] else json.dumps(checks),flush=True)
output=dict(passed=all(r['passed'] for r in results),results=results)
(repo/'docs/split_index_verification.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
if not output['passed']:sys.exit(1)
