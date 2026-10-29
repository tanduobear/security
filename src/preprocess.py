"""Shared, train-only flow preprocessing. Label/date/identity columns never enter X."""
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

METADATA_COLUMNS = ['row_index','row_id','group_id','label_raw','y_binary',
                    'attack_family','family_id','is_attempted','capture_date']

class FlowPreprocessor:
    def __init__(self, numeric_columns, port_top_k=64):
        self.numeric_columns=list(numeric_columns)
        if len(set(self.numeric_columns))!=len(self.numeric_columns):
            raise ValueError('Duplicate numeric feature names')
        forbidden=set(METADATA_COLUMNS)|{'Label','Day','Timestamp','id','Flow ID','Src IP','Dst IP','Attempted Category','Destination Port'}
        if forbidden.intersection(self.numeric_columns):
            raise ValueError('Metadata or port included in numeric features')
        if not isinstance(port_top_k,int) or port_top_k<1:
            raise ValueError('port_top_k must be a positive integer')
        self.port_top_k=port_top_k
        self.state=None

    def _numeric(self, frame):
        if not frame.columns.is_unique:raise ValueError('Duplicate input columns')
        x=frame.loc[:,self.numeric_columns].to_numpy(dtype=np.float64)
        if not np.isfinite(x).all():raise ValueError('Nonfinite numeric features')
        return x

    @staticmethod
    def _ports(frame):
        port=frame['Destination Port'].to_numpy(dtype=np.float64)
        if not (np.isfinite(port).all() and ((port>=0)&(port<=65535)&(port==np.floor(port))).all()):
            raise ValueError('Invalid destination port')
        return port.astype(np.int64)

    def fit_train_batches(self, batches, *, split='train', expected_rows=None, provenance=None):
        if split!='train':raise ValueError('Fitting is permitted only on the training fold')
        if self.state is not None:raise RuntimeError('Already fitted; create a new processor for each version/protocol')
        scaler=StandardScaler()
        ports=Counter();counts=np.zeros(2,dtype=np.int64);rows=0
        row_hash=hashlib.sha256();last_row=-1
        for frame in batches:
            x=self._numeric(frame)
            if not len(x):continue
            y=frame['y_binary'].to_numpy()
            if not np.isin(y,[0,1]).all():raise ValueError('Binary y must contain only 0 and 1')
            ids=frame['row_index'].to_numpy(dtype=np.int64)
            if ids[0]<=last_row or (np.diff(ids)<=0).any():raise ValueError('Training row indices must be strictly increasing')
            last_row=int(ids[-1]);row_hash.update(ids.astype('<i8',copy=False).tobytes())
            scaler.partial_fit(x)
            values,n=np.unique(self._ports(frame),return_counts=True)
            ports.update(dict(zip(values.tolist(),n.tolist())))
            counts+=np.bincount(y.astype(np.int64),minlength=2)
            rows+=len(frame)
        if rows==0 or (counts==0).any():raise ValueError('Training data must contain both binary classes')
        if expected_rows is not None and rows!=expected_rows:raise ValueError('Training row count mismatch')
        vocabulary=[p for p,n in sorted(ports.items(),key=lambda t:(-t[1],t[0]))[:self.port_top_k]]
        self.state=dict(format_version='1.0',fit_split='train',fit_rows=rows,
            fit_row_indices_sha256=row_hash.hexdigest(),numeric_columns=self.numeric_columns,
            mean=scaler.mean_.tolist(),variance=scaler.var_.tolist(),scale=scaler.scale_.tolist(),
            port_top_k=self.port_top_k,port_vocabulary=vocabulary,
            observed_port_counts={str(p):n for p,n in sorted(ports.items())},
            class_counts={str(i):int(counts[i]) for i in range(2)},
            class_weights={str(i):float(rows/(2*counts[i])) for i in range(2)},
            scale_pos_weight=float(counts[0]/counts[1]),
            provenance=provenance or {})
        if not (np.isfinite(self.state['mean']).all() and np.isfinite(self.state['scale']).all()):
            raise ValueError('Nonfinite training statistics')
        return self

    def feature_names(self, with_port=True):
        if self.state is None:raise RuntimeError('Not fitted')
        return self.numeric_columns + ([f'DestinationPort__{p}' for p in self.state['port_vocabulary']]+['DestinationPort__OTHER'] if with_port else [])

    def transform(self, frame, *, with_port=True):
        if self.state is None:raise RuntimeError('Not fitted')
        numeric=((self._numeric(frame)-np.asarray(self.state['mean']))/np.asarray(self.state['scale'])).astype(np.float32)
        if not np.isfinite(numeric).all():raise ValueError('Nonfinite float32 transformed features')
        if not with_port:return numeric
        vocab=self.state['port_vocabulary'];lookup={p:i for i,p in enumerate(vocab)}
        indices=np.fromiter((lookup.get(int(p),len(vocab)) for p in self._ports(frame)),dtype=np.int64,count=len(frame))
        encoded=np.zeros((len(frame),len(vocab)+1),dtype=np.float32)
        encoded[np.arange(len(frame)),indices]=1
        return np.concatenate([numeric,encoded],axis=1)

    def port_diagnostics(self, frame):
        ports=self._ports(frame)
        return {'other_rows':int((~np.isin(ports,self.state['port_vocabulary'])).sum()),
                'unseen_in_train_rows':int((~np.isin(ports,[int(x) for x in self.state['observed_port_counts']])).sum())}

    def save(self,path):
        if self.state is None:raise RuntimeError('Not fitted')
        Path(path).write_text(json.dumps(self.state,ensure_ascii=False,indent=2)+'\n',encoding='utf8')

    @classmethod
    def load(cls,path):
        state=json.loads(Path(path).read_text(encoding='utf8'))
        if state.get('format_version')!='1.0' or state.get('fit_split')!='train':raise ValueError('Unsupported or unsafe preprocessor state')
        obj=cls(state['numeric_columns'],state['port_top_k']);obj.state=state
        return obj

def iter_model_batches(path, processor, *, with_port=True, vectors_per_chunk=8):
    """Yield (float32 X, uint8 y, metadata); explicitly exclude metadata from X."""
    import duckdb
    con=duckdb.connect()
    escaped=str(Path(path).resolve()).replace("'","''")
    con.execute(f"SELECT * FROM read_parquet('{escaped}') ORDER BY row_index")
    try:
        while True:
            frame=con.fetch_df_chunk(vectors_per_chunk)
            if frame.empty:break
            yield (frame.loc[:,processor.feature_names(with_port)].to_numpy(dtype=np.float32),
                   frame['y_binary'].to_numpy(dtype=np.uint8),frame.loc[:,METADATA_COLUMNS].copy())
    finally:con.close()
