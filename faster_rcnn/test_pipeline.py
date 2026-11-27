"""Regression checks that do not need images, GPU or downloaded weights."""
import json
import tempfile
import unittest
from pathlib import Path

from faster_rcnn.cli import get_train_parser, build_overrides
from faster_rcnn.configs.config_loader import load_config


class PipelineTests(unittest.TestCase):
    def test_defaults(self):
        args = get_train_parser().parse_args([])
        self.assertEqual(load_config(), load_config(cli_overrides=build_overrides(args)))

    def test_overrides(self):
        args = get_train_parser().parse_args([
            "--trainer", "debug_cpu", "--epochs", "3", "--lr", "0.01",
            "--no-save-best", "--augment", "--lr-milestones", "1", "2",
            "--set", "trainer.optimizer.momentum=0.8",
        ])
        cfg = load_config(cli_overrides=build_overrides(args))
        self.assertEqual(cfg["NUM_EPOCHS"], 3)
        self.assertEqual(cfg["LEARNING_RATE"], .01)
        self.assertEqual(cfg["MOMENTUM"], .8)
        self.assertFalse(cfg["SAVE_BEST"])
        self.assertTrue(cfg["USE_AUGMENT"])
        self.assertEqual(cfg["LR_MILESTONES"], [1, 2])


    def test_predict_parser(self):
        from faster_rcnn.predict import get_parser
        args = get_parser().parse_args(["--split", "test", "--conf", "0.01"])
        self.assertEqual(args.split, "test")
        self.assertEqual(args.conf_thresh, 0.01)
        with self.assertRaises(SystemExit):
            get_parser().parse_args(["--pred", "existing.json"])

if __name__ == "__main__":
    unittest.main()
