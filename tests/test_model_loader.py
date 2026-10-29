import hashlib,json,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import pandas as pd
from preprocess import FlowPreprocessor
from model_loader import ModelDataset

class VerifiedLoaderTests(unittest.TestCase):
    def setUp(self):
        # Keep test scratch files inside the explicitly writable workspace.
        self.scratch=tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.directory=Path(self.scratch.name)/'original'/'stratified'
        self.directory.mkdir(parents=True)
        self.path=self.directory/'train.parquet';self.path.write_bytes(b'test-hash-only')
        self.proc=FlowPreprocessor(['value'],1)
        frame=pd.DataFrame({'value':[1.,2.],'Destination Port':[80,80],'row_index':[0,1],'y_binary':[0,1]})
        self.proc.fit_train_batches([frame],provenance={'version':'original','protocol':'stratified'})
        self.proc.save(self.directory/'preprocessor.json')
        self.manifest=dict(version='original',protocol='stratified',status='verified_train_only_and_model_data_complete',precision_guard={'checks':{'no_transformed_group_overlap':True}},
            preprocessor_sha256=hashlib.sha256((self.directory/'preprocessor.json').read_bytes()).hexdigest(),
            output={'train':{'main':{'sha256':hashlib.sha256(self.path.read_bytes()).hexdigest()}}},
            features=self.proc.feature_names(),ablation_features=self.proc.feature_names(False))
        self.write_manifest()

    def tearDown(self):self.scratch.cleanup()
    def write_manifest(self):
        (self.directory/'preprocess_manifest.json').write_text(json.dumps(self.manifest),encoding='utf8')

    def test_matching_bundle_exposes_correct_schema_and_weights(self):
        bundle=ModelDataset(self.path)
        self.assertEqual(bundle.feature_names,self.proc.feature_names())
        self.assertEqual(bundle.class_weights,{0:1.,1:1.})

    def test_unverified_or_unguarded_data_rejected(self):
        self.manifest['status']='pending';self.write_manifest()
        with self.assertRaises(ValueError):ModelDataset(self.path)
        self.manifest['status']='verified_train_only_and_model_data_complete'
        self.manifest['precision_guard']['checks']['no_transformed_group_overlap']=False;self.write_manifest()
        with self.assertRaises(ValueError):ModelDataset(self.path)

    def test_corrupted_model_file_rejected(self):
        self.path.write_bytes(b'changed')
        with self.assertRaises(ValueError):ModelDataset(self.path)

    def test_wrong_processor_scope_rejected_even_with_matching_hash(self):
        self.proc.state['provenance']['protocol']='temporal';self.proc.save(self.directory/'preprocessor.json')
        self.manifest['preprocessor_sha256']=hashlib.sha256((self.directory/'preprocessor.json').read_bytes()).hexdigest();self.write_manifest()
        with self.assertRaises(ValueError):ModelDataset(self.path)

if __name__=='__main__':unittest.main()
