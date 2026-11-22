"""Dependency-light regression tests for association and CLI evidence preservation."""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from streaming.demo import IoUTracker, main


def det(box=(0, 0, 10, 10), cls=1):
    return {'box': list(box), 'class_id': cls, 'score': .9}


class TrackerTests(unittest.TestCase):
    def test_one_to_one_and_class_aware(self):
        tracker = IoUTracker()
        first = [det()]
        tracker.update(first, 0)
        second = [det(), det(), det(cls=2)]
        tracker.update(second, 1)
        ids = [x['track_id'] for x in second]
        self.assertEqual(ids[0], first[0]['track_id'])
        self.assertEqual(len(set(ids)), 3)

    def test_expired_track_does_not_reappear(self):
        tracker = IoUTracker()
        first, second = [det()], [det()]
        tracker.update(first, 0)
        tracker.update(second, 11)
        self.assertNotEqual(first[0]['track_id'], second[0]['track_id'])

    def test_disjoint_and_empty_boxes(self):
        self.assertEqual(IoUTracker.iou([0, 0, 0, 0], [0, 0, 0, 0]), 0)
        self.assertEqual(IoUTracker.iou([0, 0, 10, 10], [20, 20, 30, 30]), 0)


class ValidationTests(unittest.TestCase):
    def test_existing_evidence_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root/'checkpoint.pt'
            checkpoint.write_text('placeholder')
            evidence = root/'evidence'
            evidence.mkdir()
            marker = evidence/'summary.json'
            marker.write_text('preserve me')
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                main(['--model', 'yolo', '--weights', str(checkpoint), '--output-dir', str(evidence)])
            self.assertEqual(caught.exception.code, 2)
            self.assertEqual(marker.read_text(), 'preserve me')

    def test_invalid_warmup_rejected_before_dependencies_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp)/'checkpoint.pt'
            checkpoint.write_text('placeholder')
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                main(['--model', 'yolo', '--weights', str(checkpoint), '--output-dir', str(Path(tmp)/'out'),
                      '--warmup', '10', '--max-frames', '10'])
            self.assertEqual(caught.exception.code, 2)


if __name__ == '__main__':
    unittest.main()
