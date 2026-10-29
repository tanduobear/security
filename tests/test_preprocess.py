import copy
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
from preprocess import FlowPreprocessor

def frame(values,ports,y,ids):
    return pd.DataFrame({'value':values,'constant':5.,'Destination Port':ports,
                         'y_binary':y,'row_index':ids,'Label':['do-not-use']*len(values)})

class PreprocessIsolationTests(unittest.TestCase):
    def setUp(self):
        self.train=frame([1.,3.,5.],[80,443,80],[0,0,1],[0,2,4])
        self.obj=FlowPreprocessor(['value','constant'],port_top_k=1)
        self.obj.fit_train_batches([self.train.iloc[:2],self.train.iloc[2:]],expected_rows=3)

    def test_heldout_extreme_values_and_new_port_do_not_change_fit(self):
        before=copy.deepcopy(self.obj.state)
        heldout=frame([1000000.],[65535],[1],[8])
        x=self.obj.transform(heldout)
        self.assertEqual(before,self.obj.state)
        self.assertAlmostEqual(x[0,0],(1000000.-3)/np.sqrt(8/3),delta=.1)
        np.testing.assert_array_equal(x[0,1:],[0,0,1])
        self.assertEqual(self.obj.state['port_vocabulary'],[80])

    def test_train_only_weights_constant_column_and_ablation(self):
        self.assertEqual(self.obj.state['class_counts'],{'0':2,'1':1})
        self.assertEqual(self.obj.state['class_weights'],{'0':.75,'1':1.5})
        self.assertEqual(self.obj.state['scale'][1],1.)
        np.testing.assert_array_equal(self.obj.transform(self.train,with_port=False),self.obj.transform(self.train)[:,:2])

    def test_fit_on_heldout_and_refitting_are_rejected(self):
        with self.assertRaises(ValueError):FlowPreprocessor(['value']).fit_train_batches([self.train],split='val')
        with self.assertRaises(RuntimeError):self.obj.fit_train_batches([self.train])

    def test_duplicate_training_rows_and_bad_ports_rejected(self):
        with self.assertRaises(ValueError):FlowPreprocessor(['value']).fit_train_batches([self.train,self.train])
        bad=self.train.copy();bad.loc[0,'Destination Port']=-1
        with self.assertRaises(ValueError):self.obj.transform(bad)

    def test_metadata_cannot_become_a_numeric_feature(self):
        for name in ['Label','Timestamp','y_binary','row_index','Attempted Category']:
            with self.assertRaises(ValueError):FlowPreprocessor([name])

    def test_saved_state_round_trip_and_noport_needs_no_port_column(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'preprocessor.json';self.obj.save(path)
            restored=FlowPreprocessor.load(path)
            np.testing.assert_array_equal(restored.transform(self.train),self.obj.transform(self.train))
            np.testing.assert_array_equal(restored.transform(self.train.drop(columns='Destination Port'),with_port=False),self.obj.transform(self.train,with_port=False))

if __name__=='__main__':unittest.main()
