"""Real metric checks on synthetic captions, without torch, images or checkpoints."""

import json
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.evaluate import (cider_score, compute_metrics, load_evaluation_inputs,
                          main, normalize_caption, prepare_inputs, read_json)


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.predictions = {
            "dog.jpg": "a brown dog runs across the grass",
            "boat.jpg": "two people sail a boat on water",
            "child.jpg": "a little child plays with a ball",
        }
        self.references = {key: [text] * 5 for key, text in self.predictions.items()}

    def test_special_tokens_removed_before_punctuation(self):
        self.assertEqual(normalize_caption("<START>A DOG's, <UNK> toy!<END><PAD>"), "a dogs toy")
        self.assertEqual(normalize_caption("<unk> <pad>"), "")

    def test_perfect_beats_unrelated_predictions(self):
        perfect = compute_metrics(self.references, self.predictions)
        wrong = compute_metrics(self.references, {key: "purple airplanes fly above snowy mountains"
                                                  for key in self.predictions})
        self.assertEqual(set(perfect), {"BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "CIDEr", "ROUGE-L"})
        for metric in perfect:
            self.assertGreater(perfect[metric], wrong[metric], metric)
        self.assertAlmostEqual(perfect["BLEU-4"], 1, places=6)
        self.assertAlmostEqual(perfect["ROUGE-L"], 1)
        self.assertAlmostEqual(perfect["CIDEr"], cider_score(self.references, self.predictions))

    def test_normalization_and_order_do_not_change_scores(self):
        decorated = {key: f"<start>{text.upper()}!<unk><end><pad>"
                     for key, text in reversed(list(self.predictions.items()))}
        decorated_refs = {key: [f"<START>{text.upper()}!<END>"] * 5
                          for key, text in self.predictions.items()}
        self.assertEqual(compute_metrics(self.references, self.predictions),
                         compute_metrics(decorated_refs, decorated))

    def test_empty_predictions_are_counted(self):
        scores = compute_metrics(self.references, {key: "" for key in self.predictions})
        self.assertTrue(all(value == 0 for value in scores.values()))

    def test_bad_coverage_reference_count_and_types_fail(self):
        invalid = [({}, self.predictions),
                   (self.references, {"other.jpg": "caption"}),
                   ({key: [text] for key, text in self.predictions.items()}, self.predictions),
                   ({key: [""] * 5 for key in self.predictions}, self.predictions),
                   (self.references, {key: [text] for key, text in self.predictions.items()})]
        for refs, hyps in invalid:
            with self.subTest(refs=refs, hyps=hyps), self.assertRaises(ValueError):
                prepare_inputs(refs, hyps)

    def test_training_wrapper_keeps_partial_prediction_support(self):
        import ast
        tree = ast.parse((Path(__file__).resolve().parents[1] / "src/train.py").read_text(encoding="utf-8"))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "cider_score")
        namespace = {}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "train.cider_score", "exec"), namespace)
        subset = dict(list(self.predictions.items())[:2])
        refs = {key: self.references[key] for key in subset}
        self.assertEqual(namespace["cider_score"](self.references, subset), cider_score(refs, subset))
        single = dict(list(subset.items())[:1])
        self.assertEqual(namespace["cider_score"](self.references, single), 0.0)

    def test_duplicate_json_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.json"
            path.write_text('{"dog.jpg": "a dog", "dog.jpg": "another dog"}')
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                read_json(path)

    def test_test_guard_runs_before_reading_files(self):
        with self.assertRaisesRegex(ValueError, "reserved"):
            load_evaluation_inputs(Path("missing"), Path("missing"), "test")

    def test_single_image_cider_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "two images"):
            compute_metrics({"a": ["a brown dog runs"] * 5}, {"a": "a brown dog runs"})

    def test_cli_inputs_match_split_and_keep_all_five_references(self):
        import yaml
        from src.data.config import load_config

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            partition = root / "split.json"
            partition.write_text(json.dumps({"val": list(self.predictions), "test": ["test.jpg"],
                                             "train": [], "unused": []}))
            config = load_config()
            config["split_path"] = str(partition)
            config_path = root / "base.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            pred_path, refs_path = root / "pred.json", root / "refs.json"
            pred_path.write_text(json.dumps(self.predictions))
            refs_path.write_text(json.dumps(self.references))
            refs, predictions = load_evaluation_inputs(pred_path, config_path, "val",
                                                       references_path=refs_path)
            self.assertEqual(refs, self.references)
            self.assertEqual(predictions, self.predictions)
            output = root / "metrics.json"
            argv = ["evaluate", "--predictions", str(pred_path), "--references", str(refs_path),
                    "--config", str(config_path), "--model-name", "synthetic-test",
                    "--output", str(output)]
            with patch("sys.argv", argv), contextlib.redirect_stdout(io.StringIO()):
                main()
            result = read_json(output)
            self.assertEqual(result["n_images"], 3)
            self.assertEqual(result["metrics"], compute_metrics(self.references, self.predictions))
            before = output.read_bytes()
            with patch("sys.argv", argv), contextlib.redirect_stderr(io.StringIO()), \
                    self.assertRaises(SystemExit) as stopped:
                main()
            self.assertEqual(stopped.exception.code, 2)
            self.assertEqual(output.read_bytes(), before)
            pred_path.write_text(json.dumps({"test.jpg": "a test caption"}))
            with self.assertRaisesRegex(ValueError, "exactly all"):
                load_evaluation_inputs(pred_path, config_path, "val", references_path=refs_path)


if __name__ == "__main__":
    unittest.main()
