"""Behavioral checks for association and preflight validation; no model download."""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from streaming.demo import IoUTracker


def detection(class_id=1, score=0.9):
    return dict(box=[10, 10, 30, 30], score=score, class_id=class_id)


class TrackerTests(unittest.TestCase):
    def test_same_object_retains_identity_but_different_class_does_not(self):
        tracker = IoUTracker()
        first = [detection()]
        tracker.update(first, 0)
        second = [detection(), detection(class_id=2)]
        tracker.update(second, 1)
        self.assertEqual(first[0]['track_id'], second[0]['track_id'])
        self.assertNotEqual(second[0]['track_id'], second[1]['track_id'])

    def test_one_track_cannot_be_assigned_twice(self):
        tracker = IoUTracker()
        first = [detection()]
        tracker.update(first, 0)
        second = [detection(score=0.5), detection(score=0.95)]
        tracker.update(second, 1)
        self.assertEqual(second[1]['track_id'], first[0]['track_id'])
        self.assertNotEqual(second[0]['track_id'], second[1]['track_id'])

    def test_track_expires_after_gap(self):
        tracker = IoUTracker()
        first = [detection()]
        tracker.update(first, 0)
        second = [detection()]
        tracker.update(second, 11)
        self.assertNotEqual(first[0]['track_id'], second[0]['track_id'])

    def test_disjoint_and_degenerate_boxes_have_zero_iou(self):
        self.assertEqual(IoUTracker.iou([0, 0, 1, 1], [2, 2, 3, 3]), 0)
        self.assertEqual(IoUTracker.iou([0, 0, 0, 0], [0, 0, 0, 0]), 0)


class PreflightTests(unittest.TestCase):
    def run_cli(self, *args):
        return subprocess.run([sys.executable, '-m', 'streaming.demo', *args],
                              capture_output=True, text=True, timeout=30)

    def test_help_does_not_load_model(self):
        result = self.run_cli('--help')
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_missing_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self.run_cli('--model', 'yolo', '--weights', str(Path(tmp)/'absent.pt'),
                                  '--output-dir', str(Path(tmp)/'out'))
            self.assertEqual(result.returncode, 2)
            self.assertIn('Checkpoint does not exist', result.stderr)
            self.assertFalse((Path(tmp)/'out').exists())

    def test_invalid_warmup_and_nonempty_output_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            weights = Path(tmp)/'local.pt'
            weights.touch()
            base = ['--model', 'yolo', '--weights', str(weights), '--output-dir', tmp]
            invalid = self.run_cli(*base, '--warmup', '3', '--max-frames', '3')
            self.assertEqual(invalid.returncode, 2)
            self.assertIn('warmup < max-frames', invalid.stderr)
            occupied = self.run_cli(*base)
            self.assertEqual(occupied.returncode, 2)
            self.assertIn('preserve earlier evidence', occupied.stderr)


if __name__ == '__main__':
    unittest.main()
