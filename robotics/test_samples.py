from pathlib import Path
import tempfile
import unittest
from robotics.samples import SampleJournal


class JournalTests(unittest.TestCase):
    def setUp(self):
        root=Path(__file__).resolve().parents[1]/'rl/.asset_cache'
        root.mkdir(exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=root)
        self.journal=SampleJournal(Path(self.temp.name)/'samples.sqlite')
        self.record=dict(sample_id='p1-t3-dbh1',plot_id='p1',tree_id='t3',
            observed_at='2026-11-06T08:00:00+08:00',kind='dbh',value=210.3,unit='mm',
            calibration_version='cal1',localization_source='local_map',quality='raw')

    def tearDown(self):self.temp.cleanup()

    def test_idempotency_and_conflict(self):
        self.assertTrue(self.journal.append(self.record))
        self.assertFalse(self.journal.append(self.record))
        with self.assertRaises(ValueError):self.journal.append(dict(self.record,value=999))
        self.assertEqual(self.journal.pending()[0]['record']['value'],210.3)

    def test_verified_receipt(self):
        self.journal.append(self.record)
        row=self.journal.pending()[0]
        with self.assertRaises(ValueError):self.journal.acknowledge(self.record['sample_id'],'bad','receipt')
        self.journal.acknowledge(self.record['sample_id'],row['sha256'],'receipt')
        self.assertEqual(self.journal.pending(),[])

    def test_rejects_missing_calibration_and_timezone(self):
        record=dict(self.record);del record['calibration_version']
        with self.assertRaises(ValueError):self.journal.append(record)
        with self.assertRaises(ValueError):self.journal.append(dict(self.record,observed_at='2026-11-06T08:00:00'))


if __name__=='__main__':unittest.main()
