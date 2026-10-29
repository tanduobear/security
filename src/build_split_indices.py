"""Build index-only P2 splits from read-only clean Parquet files."""
import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
parser.add_argument('--work-dir', type=Path, required=True)
parser.add_argument('--dependency-dir', type=Path)
args = parser.parse_args()
if args.dependency_dir:
    sys.path.insert(0, str(args.dependency_dir.resolve()))
import duckdb
import numpy as np
import pandas as pd

repo = args.repo.resolve()
work = args.work_dir.resolve()
work.mkdir(parents=True, exist_ok=True)
feature = json.loads((repo/'docs/feature_policy.json').read_text(encoding='utf8'))
policy = json.loads((repo/'docs/label_split_policy.json').read_text(encoding='utf8'))
ratios = [.70, .15, .15]
names = ['train', 'val', 'test']
seed = policy['seed']
families = policy['family_names']
reports = {}
def q(s): return '"'+s.replace('"','""')+'"'
def lit(s): return "'"+str(s).replace("'","''")+"'"
def digest(p):
    with p.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()
def dump(p, data):
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
def log(msg): print(msg, flush=True)

for version in ['original', 'corrected']:
    log(f'{version}: checking source and constructing exact feature groups')
    source = repo/'data/clean'/f'{version}.parquet'
    source_hash = digest(source)
    if source_hash != feature['source_sha256'][version]:
        raise ValueError('Source hash changed; review the policy before splitting')
    db_path = work/f'{version}_split_build.duckdb'
    con = duckdb.connect(str(db_path))
    con.execute("SET memory_limit='1GB'")
    con.execute('SET threads=4')
    con.execute(f'SET temp_directory={lit((work/(version+"_tmp")).as_posix())}')
    con.execute('SET preserve_insertion_order=false')
    con.execute(f'CREATE OR REPLACE VIEW source AS SELECT * FROM read_parquet({lit(source.as_posix())}, file_row_number=true)')
    schema = dict((r[0],r[1]) for r in con.execute('DESCRIBE source').fetchall())
    sources = feature['feature_sources'][version]
    selected = [c for c in feature['canonical_feature_names'] if c!='Destination Port']
    raw = [sources[c] for c in selected]
    # Integers must round-trip through the common float64 representation exactly.
    integers = [c for c in raw if 'INT' in schema[c]]
    if integers:
        bad = con.execute('SELECT '+','.join(f'count(*) FILTER (WHERE {q(c)}>9007199254740991 OR {q(c)}< -9007199254740991)' for c in integers)+' FROM source').fetchone()
        if any(bad): raise ValueError('Integer outside exact float64 range; use a lossless serializer')
    invalid = ' OR '.join(f'{q(c)} IS NULL OR NOT isfinite({q(c)})' for c in raw)
    if con.execute('SELECT count(*) FROM source WHERE '+invalid).fetchone()[0]:
        raise ValueError('Null/nonfinite selected features')
    # JSON object key order is explicit; normalize signed zero, no rounding or binning.
    serial = 'to_json(struct_pack('+','.join(f'f{i:02d} := CASE WHEN {q(c)}=0 THEN 0.0::DOUBLE ELSE {q(c)}::DOUBLE END' for i,c in enumerate(raw))+'))::VARCHAR'
    if version=='original':
        date_expr = "CASE Day WHEN 'Monday' THEN DATE '2017-07-03' WHEN 'Tuesday' THEN DATE '2017-07-04' WHEN 'Wednesday' THEN DATE '2017-07-05' WHEN 'Thursday' THEN DATE '2017-07-06' WHEN 'Friday' THEN DATE '2017-07-07' END"
    else:
        date_expr = 'try_cast(Timestamp AS TIMESTAMP)::DATE'
    label_rows = [r for r in policy['label_mapping'] if r['version']==version]
    def mapping_expr(key):
        return 'CASE Label '+ ' '.join('WHEN '+lit(r['label_raw'])+' THEN '+(lit(r[key]) if isinstance(r[key],str) else str(r[key]).lower()) for r in label_rows)+' END'
    con.execute(f'''CREATE OR REPLACE TABLE members AS
        SELECT file_row_number::BIGINT AS row_index, Label AS label_raw,
        {mapping_expr('family_id')}::UTINYINT AS family_id,
        {mapping_expr('y_binary')}::UTINYINT AS y_binary,
        {mapping_expr('is_attempted')}::BOOLEAN AS is_attempted,
        {date_expr} AS capture_date, {serial} AS canonical
        FROM source''')
    total = con.execute('SELECT count(*) FROM members').fetchone()[0]
    if total != policy['summaries'][version]['total']: raise ValueError('Row count changed')
    if con.execute("SELECT count(*) FROM members WHERE family_id IS NULL OR capture_date IS NULL OR capture_date NOT BETWEEN DATE '2017-07-03' AND DATE '2017-07-07'").fetchone()[0]:
        raise ValueError('Unmapped label or day')
    con.execute('ALTER TABLE members ADD COLUMN group_key VARCHAR')
    con.execute('UPDATE members SET group_key=sha256(canonical)')
    con.execute('CREATE OR REPLACE TABLE group_counts AS SELECT group_key,count(*) AS n FROM members GROUP BY group_key')
    log(f'{version}: verifying repeated-hash candidates by exact canonical comparison')
    collisions = con.execute('''SELECT count(*) FROM (
        SELECT m.group_key FROM members m JOIN group_counts g USING(group_key)
        WHERE g.n>1 GROUP BY m.group_key HAVING min(m.canonical)<>max(m.canonical))''').fetchone()[0]
    if collisions: raise ValueError('SHA256 collision: stop, do not merge unequal groups')
    con.execute('ALTER TABLE members DROP COLUMN canonical')
    con.execute('CREATE OR REPLACE TABLE group_families AS SELECT group_key,family_id,count(*) AS n FROM members GROUP BY ALL')
    con.execute('''CREATE OR REPLACE TABLE groups AS SELECT
        row_number() OVER(ORDER BY group_key)-1 AS group_index,group_key,sum(n)::BIGINT AS n,
        list(family_id ORDER BY family_id) AS family_ids,list(n ORDER BY family_id) AS family_counts
        FROM group_families GROUP BY group_key''')
    ngroups = con.execute('SELECT count(*) FROM groups').fetchone()[0]
    mixed = con.execute('SELECT count(*) FROM groups WHERE len(family_ids)>1').fetchone()[0]
    raw_conflicts = con.execute('SELECT count(*) FROM (SELECT group_key FROM members GROUP BY group_key HAVING count(DISTINCT label_raw)>1)').fetchone()[0]
    actual_totals = [0]*len(families)
    for fid,n in con.execute('SELECT family_id,count(*) FROM members GROUP BY family_id').fetchall(): actual_totals[fid]=n
    if any(x==0 for x in actual_totals): raise ValueError('Unexpected absent family')
    independent_groups = [0]*len(families)
    for fid,n in con.execute('SELECT family_id,count(*) FROM group_families GROUP BY family_id').fetchall(): independent_groups[fid]=n
    target = [[ratio*n for n in actual_totals] for ratio in ratios]
    counts = [[0]*len(families) for _ in names]
    sizes = [0]*3
    target_size = [total*r for r in ratios]
    assignment = np.full(ngroups,-1,dtype=np.int8)
    log(f'{version}: allocating {ngroups:,} groups; {mixed:,} mixed-family groups')
    cursor = con.execute(f'''SELECT group_index,n,family_ids,family_counts FROM groups
        ORDER BY n DESC,sha256({lit(str(seed)+':')}||group_key),group_key''')
    processed=0
    while True:
        batch = cursor.fetchmany(20000)
        if not batch: break
        for gi,n,ids,values in batch:
            best=0; best_cost=float('inf')
            for split in range(3):
                d=max(target_size[split],1)
                cost=(2*n*(sizes[split]-target_size[split])+n*n)/(d*d)
                for fid,amount in zip(ids,values):
                    d=max(target[split][fid],1)
                    cost+=(2*amount*(counts[split][fid]-target[split][fid])+amount*amount)/(d*d)
                if cost<best_cost:
                    best_cost=cost;best=split
            assignment[gi]=best
            sizes[best]+=n
            for fid,amount in zip(ids,values):counts[best][fid]+=amount
        processed+=len(batch)
        if processed%500000==0:log(f'{version}: allocated {processed:,}/{ngroups:,}')
    # Coverage repair: fixed family/fold order, smallest movable group then group hash.
    repairs=[]
    for fid in range(len(families)):
        if independent_groups[fid]<3: continue
        for dest in range(3):
            if counts[dest][fid]>0:continue
            candidates=con.execute('SELECT group_index,group_key,n,family_ids,family_counts FROM groups WHERE list_contains(family_ids,?) ORDER BY n,group_key',[fid]).fetchall()
            for gi,key,n,ids,values in candidates:
                src=int(assignment[gi])
                if src==dest:continue
                if any(independent_groups[c]>=3 and counts[src][c]-v<=0 for c,v in zip(ids,values)):continue
                assignment[gi]=dest;sizes[src]-=n;sizes[dest]+=n
                for c,v in zip(ids,values):counts[src][c]-=v;counts[dest][c]+=v
                repairs.append(dict(family=families[fid],group_key=key,source=names[src],destination=names[dest],rows=n))
                break
    if (assignment<0).any():raise ValueError('Unassigned group')
    failures=[dict(family=families[fid],missing=names[s]) for fid in range(len(families)) for s in range(3) if counts[s][fid]==0]
    if failures:raise ValueError('Coverage failure; revise documented constraints: '+json.dumps(failures))
    if max(abs(sizes[i]/total-ratios[i]) for i in range(3))>.01:raise ValueError('Ratio deviation exceeds 1pp')
    adf=pd.DataFrame({'group_index':np.arange(ngroups,dtype=np.int64),'split_code':assignment})
    con.register('assignment_frame',adf)
    con.execute('CREATE OR REPLACE TABLE assignments AS SELECT * FROM assignment_frame')
    con.unregister('assignment_frame')
    con.execute('''CREATE OR REPLACE TABLE allocated AS SELECT m.*,a.split_code
        FROM members m JOIN groups g USING(group_key) JOIN assignments a USING(group_index)''')
    outputs={}
    for protocol in ['stratified','temporal']:
        directory=repo/'data/split'/version/protocol
        directory.mkdir(parents=True,exist_ok=True)
        if protocol=='stratified':
            con.execute('CREATE OR REPLACE VIEW chosen AS SELECT *,false AS purged FROM allocated')
        else:
            con.execute("CREATE OR REPLACE TABLE time_allocated AS SELECT *,CASE WHEN capture_date<=DATE '2017-07-05' THEN 0 WHEN capture_date=DATE '2017-07-06' THEN 1 ELSE 2 END AS split_code FROM members")
            con.execute('CREATE OR REPLACE TABLE first_split AS SELECT group_key,min(split_code) AS first_code FROM time_allocated GROUP BY group_key')
            con.execute('CREATE OR REPLACE VIEW chosen AS SELECT t.*,t.split_code>f.first_code AS purged FROM time_allocated t JOIN first_split f USING(group_key)')
        indices={}; class_rows=[]; checks={}
        family_case='CASE family_id '+ ' '.join(f'WHEN {i} THEN {lit(x)}' for i,x in enumerate(families))+' END'
        fields=f"row_index,{lit(version+':')}||row_index::VARCHAR AS row_id,group_key AS group_id,label_raw,y_binary,{family_case} AS attack_family,family_id,is_attempted,capture_date"
        for i,split in enumerate(names):
            path=directory/f'{split}_indices.parquet'
            con.execute(f"COPY (SELECT {fields} FROM chosen WHERE NOT purged AND split_code={i} ORDER BY row_index) TO {lit(path.as_posix())} (FORMAT PARQUET, COMPRESSION ZSTD)")
            rows=con.execute('SELECT count(*) FROM chosen WHERE NOT purged AND split_code=?',[i]).fetchone()[0]
            indices[split]=dict(path=path.relative_to(repo).as_posix(),rows=rows,sha256=digest(path),bytes=path.stat().st_size)
        purged_path=directory/'purged_indices.parquet'
        con.execute(f"COPY (SELECT {fields},CASE split_code WHEN 0 THEN 'train' WHEN 1 THEN 'val' ELSE 'test' END AS intended_split,'same_75_features_in_earlier_split' AS reason FROM chosen WHERE purged ORDER BY row_index) TO {lit(purged_path.as_posix())} (FORMAT PARQUET, COMPRESSION ZSTD)")
        purged_count=con.execute('SELECT count(*) FROM chosen WHERE purged').fetchone()[0]
        overlap_rows=con.execute('SELECT count(*)-count(DISTINCT row_index) FROM chosen WHERE NOT purged').fetchone()[0]
        overlap_groups=con.execute('SELECT count(*) FROM (SELECT group_key FROM chosen WHERE NOT purged GROUP BY group_key HAVING count(DISTINCT split_code)>1)').fetchone()[0]
        kept=sum(x['rows'] for x in indices.values())
        checks=dict(row_overlap_count=overlap_rows,group_overlap_count=overlap_groups,all_source_rows_accounted_for=(kept+purged_count==total),exact_hash_collision_groups=collisions)
        if overlap_rows or overlap_groups or not checks['all_source_rows_accounted_for']:raise ValueError('Overlap/accounting failure')
        for code,purged,label,binval,fid,attempted,n in con.execute('SELECT split_code,purged,label_raw,y_binary,family_id,is_attempted,count(*) FROM chosen GROUP BY ALL ORDER BY ALL').fetchall():
            class_rows.append(dict(split=names[code],purged=purged,label_raw=label,y_binary=binval,attack_family=families[fid],is_attempted=attempted,count=n))
        manifest=dict(version=version,protocol=protocol,artifact_kind='indices_only_not_model_feature_data',created_at=datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds'),seed=seed,input_path=source.relative_to(repo).as_posix(),input_sha256=source_hash,input_rows=total,indices=indices,purged_rows=purged_count,purged_index_sha256=digest(purged_path),checks=checks,grouping_raw_fields=75,group_serializer='ordered JSON struct f00..f74 of exact float64 after integer-range validation and signed-zero normalization; SHA256 with exact min/max comparison for repeated hashes',feature_policy_sha256=digest(repo/'docs/feature_policy.json'),label_policy_sha256=digest(repo/'docs/label_split_policy.json'),script_sha256=digest(Path(__file__)),runtime=dict(python=platform.python_version(),duckdb=duckdb.__version__,numpy=np.__version__,pandas=pd.__version__),class_distribution=class_rows,random_coverage_repairs=repairs if protocol=='stratified' else [],family_independent_group_counts=dict(zip(families,independent_groups)),mixed_family_groups=mixed,multiple_raw_label_groups=raw_conflicts)
        dump(directory/'manifest.json',manifest)
        outputs[protocol]=manifest
        log(f'{version}/{protocol}: '+str({k:v['rows'] for k,v in indices.items()})+f'; purged={purged_count:,}; overlaps={overlap_rows}/{overlap_groups}')
    reports[version]=outputs
    con.close()
dump(repo/'docs/split_index_results.json',reports)
log('All index generation and internal checks passed')
