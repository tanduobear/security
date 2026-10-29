"""Fit train-only preprocessing and export main/port-ablation model Parquet data."""
import argparse, hashlib, json, platform, sys, uuid
from pathlib import Path
parser=argparse.ArgumentParser()
parser.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[1])
parser.add_argument('--work-dir',type=Path,required=True)
parser.add_argument('--dependency-dir',type=Path)
args=parser.parse_args()
if args.dependency_dir:sys.path.insert(0,str(args.dependency_dir.resolve()))
import duckdb
import numpy as np
import pandas as pd
import sklearn
from preprocess import FlowPreprocessor,METADATA_COLUMNS
repo=args.repo.resolve();work=args.work_dir.resolve();work.mkdir(parents=True,exist_ok=True)
feature=json.loads((repo/'docs/feature_policy.json').read_text(encoding='utf8'))
verification=json.loads((repo/'docs/split_index_verification.json').read_text(encoding='utf8'))
if not verification['passed']:raise ValueError('Index verification has not passed')
def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def lit(s):return "'"+str(s).replace("'","''")+"'"
def q(s):return '"'+s.replace('"','""')+'"'
def dump(path,data):Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def log(s):print(s,flush=True)
rawcon=duckdb.connect();rawcon.execute("SET memory_limit='1GB'");rawcon.execute('SET threads=1')
rawcon.execute('SET preserve_insertion_order=false')
rawcon.execute(f'SET temp_directory={lit((work/"raw_sort_tmp").as_posix())}')
outcon=duckdb.connect();outcon.execute("SET memory_limit='1GB'");outcon.execute('SET threads=2')
outcon.execute('SET preserve_insertion_order=false')
outcon.execute(f'SET temp_directory={lit((work/"sort_tmp").as_posix())}')
results={}
for version in ['original','corrected']:
    source=repo/'data/clean'/f'{version}.parquet'
    source_hash=sha(source)
    if source_hash!=feature['source_sha256'][version]:raise ValueError('Source changed')
    sources=feature['feature_sources'][version]
    canonical=feature['canonical_feature_names']
    numeric=[c for c in canonical if c!='Destination Port']
    for protocol in ['stratified','temporal']:
        directory=repo/'data/split'/version/protocol
        indices=json.loads((directory/'manifest.json').read_text(encoding='utf8'))
        if source_hash!=indices['input_sha256']:raise ValueError('Index source mismatch')
        for policy,field in [('feature_policy.json','feature_policy_sha256'),('label_split_policy.json','label_policy_sha256')]:
            if sha(repo/'docs'/policy)!=indices[field]:raise ValueError('Policy changed since splitting')
        for split in ['train','val','test']:
            if sha(directory/f'{split}_indices.parquet')!=indices['indices'][split]['sha256']:raise ValueError('Index changed')
        def batches(split):
            index=directory/f'{split}_indices.parquet'
            select=','.join(f's.{q(sources[c])} AS {q(c)}' for c in canonical)
            meta=','.join('i.'+q(c) for c in METADATA_COLUMNS)
            rawcon.execute(f'''SELECT {select},{meta} FROM read_parquet({lit(source.as_posix())},file_row_number=true) s
                JOIN read_parquet({lit(index.as_posix())}) i ON s.file_row_number=i.row_index ORDER BY i.row_index''')
            while True:
                frame=rawcon.fetch_df_chunk(8)
                if frame.empty:break
                yield frame
        log(f'{version}/{protocol}: fitting numeric scaler, top-64 port vocabulary and weights on train only')
        proc=FlowPreprocessor(numeric,port_top_k=64)
        proc.fit_train_batches(batches('train'),expected_rows=indices['indices']['train']['rows'],provenance=dict(
            version=version,protocol=protocol,input_sha256=source_hash,
            train_index_sha256=indices['indices']['train']['sha256'],index_manifest_sha256=sha(directory/'manifest.json'),
            feature_policy_sha256=sha(repo/'docs/feature_policy.json'),label_policy_sha256=sha(repo/'docs/label_split_policy.json')))
        state_path=directory/'preprocessor.json';proc.save(state_path)
        state_hash=sha(state_path)
        ablation=directory/'no_destination_port';ablation.mkdir(exist_ok=True)
        output={}
        for split in ['train','val','test']:
            parts=work/version/protocol/split/uuid.uuid4().hex
            parts.mkdir(parents=True)
            seen=0;diagnostics={'other_rows':0,'unseen_in_train_rows':0}
            for batch_index,frame in enumerate(batches(split)):
                x=proc.transform(frame)
                df=pd.DataFrame(x,columns=proc.feature_names(),index=frame.index)
                df=pd.concat([df,frame[METADATA_COLUMNS]],axis=1)
                outcon.register('transformed_batch',df)
                outcon.execute(f'COPY transformed_batch TO {lit((parts/f"part_{batch_index:05d}.parquet").as_posix())} (FORMAT PARQUET, COMPRESSION ZSTD)')
                outcon.unregister('transformed_batch')
                seen+=len(frame)
                for k,n in proc.port_diagnostics(frame).items():diagnostics[k]+=n
            if seen!=indices['indices'][split]['rows']:raise ValueError('Export row mismatch')
            main_path=directory/f'{split}.parquet';abl_path=ablation/f'{split}.parquet'
            pattern=(parts/'part_*.parquet').as_posix()
            outcon.execute(f'COPY (SELECT * FROM read_parquet({lit(pattern)}) ORDER BY row_index) TO {lit(main_path.as_posix())} (FORMAT PARQUET, COMPRESSION ZSTD)')
            cols=','.join(q(c) for c in proc.feature_names(False)+METADATA_COLUMNS)
            outcon.execute(f'COPY (SELECT {cols} FROM read_parquet({lit(main_path.as_posix())}) ORDER BY row_index) TO {lit(abl_path.as_posix())} (FORMAT PARQUET, COMPRESSION ZSTD)')
            # Transform must not mutate fitted parameters or learn from held-out folds.
            check_state=work/'post_transform_state.json';proc.save(check_state)
            if sha(check_state)!=state_hash:raise ValueError('Transform mutated fitted state')
            output[split]=dict(rows=seen,main=dict(path=main_path.relative_to(repo).as_posix(),sha256=sha(main_path),bytes=main_path.stat().st_size),no_destination_port=dict(path=abl_path.relative_to(repo).as_posix(),sha256=sha(abl_path),bytes=abl_path.stat().st_size),port_diagnostics=diagnostics)
            log(f'{version}/{protocol}/{split}: {seen:,} rows, main {len(proc.feature_names())} features / ablation {len(numeric)}; unseen ports {diagnostics["unseen_in_train_rows"]:,}')
        m=dict(version=version,protocol=protocol,status='model_data_generated_pending_independent_verification',input_sha256=source_hash,index_manifest_sha256=sha(directory/'manifest.json'),preprocessor_sha256=state_hash,preprocessor_path=state_path.relative_to(repo).as_posix(),fit_rows=proc.state['fit_rows'],fit_split='train',class_counts=proc.state['class_counts'],class_weights=proc.state['class_weights'],scale_pos_weight=proc.state['scale_pos_weight'],features=proc.feature_names(),ablation_features=proc.feature_names(False),metadata_columns=METADATA_COLUMNS,port_top_k=64,port_vocabulary=proc.state['port_vocabulary'],output=output,preprocess_module_sha256=sha(repo/'src/preprocess.py'),builder_sha256=sha(Path(__file__)),runtime=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,duckdb=duckdb.__version__,sklearn=sklearn.__version__))
        dump(directory/'preprocess_manifest.json',m)
        results[f'{version}/{protocol}']=m
        dump(repo/'docs/preprocess_results.json',results)
dump(repo/'docs/preprocess_results.json',results)
log('All four preprocessors and 24 main/ablation data files generated')
