"""src/results_table.py: geçici, sahte bir results/ klasörüyle (gerçek sonuçlara dokunmadan)."""

import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path

from src.results_table import build, main

SHA = {"reference_sha256": "ref", "split_sha256": "split"}


def metrics(cider: float, base: float = 0.5) -> dict:
    return {"BLEU-1": base, "BLEU-2": base / 2, "BLEU-3": base / 4, "BLEU-4": base / 8,
            "CIDEr": cider, "ROUGE-L": 0.4}


def write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def make_run(root: Path, run: str, greedy: float, beam: float, *, epoch: int = 10, params: int = 1234567,
             summary_dir: str | None = None, n_images: int = 1000, split: str = "val",
             predict_cider: dict | None = None) -> None:
    for key, name, method, beam_size, cider in [("greedy", "val_metrics_greedy.json", "greedy", None, greedy),
                                                 ("beam3", "val_metrics_beam3.json", "beam", 3, beam)]:
        write(root / run / name, {"model": run, "split": split, "method": method, "beam_size": beam_size,
                                  "n_images": n_images, "references_per_image": 5,
                                  "metrics": metrics(cider), **SHA})
    predict_cider = predict_cider or {}
    write(root / run / "val_metrics.json", {
        key: {"checkpoint": f"checkpoints/{summary_dir or run}/best.pt", "checkpoint_epoch": epoch,
              "images": 1000, "unique_captions": count, "CIDEr": predict_cider.get(key, round(cider, 4)),
              "ms_per_image": ms, "device": "cuda"}
        for key, count, cider, ms in [("greedy", 100, greedy, 10.0), ("beam3", 90, beam, 30.0)]})
    write(root / (summary_dir or run) / "summary.json",
          {"parameters": params, "best_epoch": epoch, "checkpoint": f"checkpoints/{summary_dir or run}/best.pt"})


def make_nucleus(root: Path, ciders: list[float]) -> None:
    for seed, cider in zip([42, 43, 44], ciders):
        seed_dir = root / "model2_attention_lstm" / f"nucleus_p09_seed{seed}"
        write(seed_dir / "quality_metrics.json", {"split": "val", "method": "nucleus", "beam_size": None,
                                                  "n_images": 1000, "references_per_image": 5,
                                                  "metrics": metrics(cider), **SHA})
        write(seed_dir / "val_metrics.json", {"nucleus0.9": {
            "checkpoint_epoch": 10, "images": 1000, "unique_captions": 990, "unique_caption_ratio": 0.99,
            "CIDEr": round(cider, 4), "ms_per_image": 40.0, "device": "cuda"}})


class ResultsTableTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "results"
        self.root.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def full_fixture(self):
        make_run(self.root, "model1_lstm", 0.230555, 0.277755, params=5653004)
        make_run(self.root, "model1_lstm_frac25", 0.16418, 0.2178)
        make_run(self.root, "model1_lstm_frac50", 0.1943, 0.2086, epoch=12)
        make_run(self.root, "model2_attention_lstm", 0.4409, 0.46970, params=11159565,
                 summary_dir="model2_attention_lstm_full_20261104_01")
        make_nucleus(self.root, [0.30, 0.25, 0.20])

    def test_empty_results_dir_does_not_crash(self):
        md, rows, console = build(self.root)
        self.assertEqual(md.count("| henüz yok |"), 10 + 6 + 5 + 4 + 4)
        self.assertEqual([r[3] for r in rows[1:]], ["henüz yok"] * 10)
        self.assertTrue(any("model3_transformer/" in line for line in console))

    def test_numbers_come_from_json_with_four_decimals(self):
        self.full_fixture()
        md, rows, console = build(self.root)
        self.assertEqual([line for line in console if not line.startswith("eksik")], [])
        self.assertIn("| Model 1 — CNN-LSTM | greedy | 0.5000 | 0.2500 | 0.1250 | 0.0625 | — | 0.4000 | "
                      "0.2306 | 100 | 10.00 | 5,653,004 | 10 |", md)
        self.assertIn("| 0.4697 | 90 | 30.00 | 11,159,565 | 10 |", md)   # Model 2 summary başka klasörde
        self.assertIn("0.1642", md)
        self.assertIn("| Model 3 — Transformer | greedy | henüz yok |", md)
        self.assertIn("| Model 4 — BLIP (fine-tune edilmeden) | beam-3 | henüz yok |", md)
        self.assertIn("- `model2_attention_lstm_full_20261104_01/summary.json`", md)
        self.assertIn("**Uyarılar:**\n\n- Yok.", md)
        self.assertNotIn("model2_attention_lstm/summary.json", md)
        self.assertEqual(rows[0][:4], ["model", "run", "decoding", "status"])
        self.assertEqual(rows[1][rows[0].index("CIDEr")], "0.2306")
        self.assertEqual(rows[3][rows[0].index("parameters")], "11159565")
        self.assertEqual(rows[5][3], "henüz yok")

    def test_ablation_deltas_and_nucleus_mean_std(self):
        self.full_fixture()
        md, _, _ = build(self.root)
        self.assertIn("| %25 (`model1_lstm_frac25`) | greedy |", md)
        self.assertIn("| -0.0664 |", md)    # 0.16418 - 0.230555
        self.assertIn("| +0.0472 | 0.0900 |", md)   # Model 1 beam-3 vs greedy, oran 90/1000
        self.assertIn("| nucleus p=0.9 |", md)
        self.assertIn("0.2500 ± 0.0500", md)  # CIDEr ortalama ± ddof=1 std
        self.assertIn("| +0.2103 |", md)    # attention: 0.4409 - 0.230555
        self.assertIn("| model1_lstm_glove300 | greedy | henüz yok |", md)

    def test_embedding_runs_discovered_when_present(self):
        self.full_fixture()
        make_run(self.root, "model1_lstm_emb300", 0.24, 0.29)
        make_run(self.root, "model1_lstm_glove300", 0.26, 0.30)
        md, _, _ = build(self.root)
        self.assertIn("| model1_lstm_glove300 | greedy | 0.5000", md)
        self.assertIn("| +0.0200 |", md)
        self.assertNotIn("model1_lstm_glove300 | greedy | henüz yok", md)

    def test_consistency_warnings(self):
        self.full_fixture()
        make_run(self.root, "model1_lstm_frac25", 0.16418, 0.2178, n_images=999, split="test",
                 predict_cider={"greedy": 0.1700})
        md, _, console = build(self.root)
        warnings = md.split("**Uyarılar:**")[1]
        self.assertIn("n_images = 999", warnings)
        self.assertIn("split = 'test'", warnings)
        self.assertIn("CIDEr tutarsız", warnings)
        self.assertTrue(any("CIDEr tutarsız" in line for line in console))

    def test_config_diff_flags_extra_changes(self):
        self.full_fixture()
        configs = self.root.parent / "configs"
        configs.mkdir()
        (configs / "model1.yaml").write_text("name: a\nmodel: {type: lstm, embedding_dim: 256}\n"
                                             "optimizer: {lr: 0.001}\n", encoding="utf-8")
        (configs / "model2.yaml").write_text("name: b\nmodel: {type: attention_lstm, embedding_dim: 256}\n"
                                             "optimizer: {lr: 0.0005}\n", encoding="utf-8")
        md, _, _ = build(self.root)
        self.assertIn("farklı anahtarlar: `model.type`, `name`, `optimizer.lr`", md)
        self.assertIn("Attention ablation'ı tek değişkenli değil", md.split("**Uyarılar:**")[1])

    def test_main_writes_md_and_csv(self):
        self.full_fixture()
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["--results-dir", str(self.root)]), 0)
        self.assertTrue((self.root / "val_table.md").read_text(encoding="utf-8").startswith("# Birleşik"))
        with (self.root / "val_table.csv").open(encoding="utf-8", newline="") as handle:
            self.assertEqual(len(list(csv.reader(handle))), 11)


if __name__ == "__main__":
    unittest.main()
