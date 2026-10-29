"""Load model data together with its matching processor and integrity manifest."""
import hashlib,json
from pathlib import Path
try:
    from .preprocess import FlowPreprocessor,iter_model_batches
except ImportError:
    from preprocess import FlowPreprocessor,iter_model_batches

def _sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

class ModelDataset:
    def __init__(self,path):
        self.path=Path(path).resolve()
        self.with_port=self.path.parent.name!='no_destination_port'
        self.directory=self.path.parent if self.with_port else self.path.parent.parent
        self.manifest=json.loads((self.directory/'preprocess_manifest.json').read_text(encoding='utf8'))
        if self.manifest.get('status')!='verified_train_only_and_model_data_complete':
            raise ValueError('Model data has not completed independent verification')
        if 'precision_guard' not in self.manifest:raise ValueError('Dataset has not passed model-precision cleanup')
        if self.manifest['precision_guard'].get('checks',{}).get('no_transformed_group_overlap') is not True:
            raise ValueError('Transformed feature overlap check has not passed')
        if self.manifest['version']!=self.directory.parent.name or self.manifest['protocol']!=self.directory.name:
            raise ValueError('Version/protocol directory mismatch')
        split=self.path.stem
        if split not in ['train','val','test']:raise ValueError('Expected train.parquet, val.parquet or test.parquet')
        variant='main' if self.with_port else 'no_destination_port'
        if _sha(self.path)!=self.manifest['output'][split][variant]['sha256']:raise ValueError('Model file hash mismatch')
        state=self.directory/'preprocessor.json'
        if _sha(state)!=self.manifest['preprocessor_sha256']:raise ValueError('Processor hash mismatch')
        self.processor=FlowPreprocessor.load(state)
        provenance=self.processor.state['provenance']
        if provenance.get('version')!=self.manifest['version'] or provenance.get('protocol')!=self.manifest['protocol']:
            raise ValueError('Wrong processor for this version/protocol')
        self.feature_names=self.processor.feature_names(self.with_port)
        expected=self.manifest['features' if self.with_port else 'ablation_features']
        if self.feature_names!=expected:raise ValueError('Feature schema mismatch')
        self.class_weights={int(k):v for k,v in self.processor.state['class_weights'].items()}
        self.scale_pos_weight=self.processor.state['scale_pos_weight']

    def __iter__(self):
        yield from iter_model_batches(self.path,self.processor,with_port=self.with_port)
