import json
import tempfile
import unittest
from pathlib import Path
from openhear.live_report import summarise, percentile, report_file


class TestLiveReport(unittest.TestCase):
    def test_percentiles(self):
        self.assertEqual(percentile([100,200,300], .5), 200)
        self.assertEqual(percentile([], .95), None)
        self.assertEqual(percentile([100,200,300], .95), 290)

    def test_summary_preserves_directions_and_drop_rate(self):
        payload={'schema_version':'0.4-live-client','dropped':1,'errors':1,
                 'audit_records':[
            {'source':'en','target':'zh','status':'completed','tts_status':'generated',
             'pause_to_result_ms':350,'request_ms':230},
            {'source':'zh','target':'en','status':'failed','tts_status':'unavailable',
             'pause_to_result_ms':500,'request_ms':490}]}
        report=summarise(payload)
        self.assertEqual(report['total_audit_records'],2)
        self.assertAlmostEqual(report['estimated_drop_fraction'],1/3,places=6)
        self.assertEqual(report['by_direction']['en-zh']['voice_generated'],1)
        self.assertEqual(report['by_direction']['zh-en']['failed'],1)
        self.assertTrue(report['evidence_complete_for_all_turns'])

    def test_invalid_direction(self):
        with self.assertRaises(ValueError):summarise({'schema_version':'0.4-live-client',
            'audit_records':[{'source':'fr','target':'en'}]})

    def test_overflow_marks_incomplete(self):
        record=summarise({'schema_version':'0.4-live-client','audit_overflow':2,'audit_records':[]})
        self.assertFalse(record['evidence_complete_for_all_turns'])

    def test_bad_schema(self):
        with self.assertRaises(ValueError):summarise({'schema_version':'fixture','audit_records':[]})

    def test_no_overwrite_report(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);raw=root/'source.json';out=root/'summary.json'
            raw.write_text(json.dumps({'schema_version':'0.4-live-client','audit_records':[]}))
            filename,data=report_file(str(raw),str(out))
            self.assertEqual(Path(filename),out)
            self.assertEqual(data['total_audit_records'],0)
            with self.assertRaises(FileExistsError):report_file(str(raw),str(out))
