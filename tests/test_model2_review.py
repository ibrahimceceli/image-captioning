"""Review-only regression checks in a temporary delivery copy; no ML/data access.

Run: python -m unittest discover -s tests -p test_model2_review.py -v
"""
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
STUDY = Path('results/model2_val_study_20261112')
EXPECTED = {'yanlış nesne':11, 'sayı':0, 'renk':10, 'eylem':11,
            'genel ifade':18, 'kelime tekrarı':4}
HUMAN_FIELDS = ('labels','reason','uncertain_labels','needs_user_review','human_reviewed',
                'review_status','uncertainty','human_labels','human_notes')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def hashes(folder):
    return {p.relative_to(folder).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
            for p in folder.rglob('*') if p.is_file()}


class Model2ReviewTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='model2-review-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.study = self.root/STUDY
        self.review = self.study/'review'
        shutil.copytree(ROOT/STUDY, self.study,
                        ignore=shutil.ignore_patterns('records_pending.json','captions_and_references.md'))
        (self.root/'report/tools').mkdir(parents=True)
        self.script = self.root/'report/tools/model2_val_study.py'
        shutil.copy2(ROOT/'report/tools/model2_val_study.py', self.script)
        shutil.copy2(ROOT/'report/model2_b_tamamlayici_20261112.md', self.root/'report')
        self.before = read(self.review/'review40.json')
        self.csv_before = self.csv_rows()
        self.png_hashes = {p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in (self.study/'attention').glob('*.png')}

    def csv_rows(self):
        with (self.review/'review40.csv').open(encoding='utf-8-sig', newline='') as handle:
            return list(csv.DictReader(handle))

    def run_stage(self, stage, success=True):
        result = subprocess.run([sys.executable,str(self.script),stage],cwd=self.root,
                                env={**os.environ,'PYTHONUTF8':'1'},capture_output=True,text=True,
                                encoding='utf-8',timeout=30)
        if success:
            self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)
        else:
            self.assertNotEqual(result.returncode,0)
        return result

    def assert_final(self):
        summary = read(self.review/'summary.json')
        self.assertEqual(summary['n'],40)
        self.assertEqual(summary['counts'],EXPECTED)
        self.assertEqual(summary['any_label'],28)
        self.assertEqual(summary['confirmed_label_assignments'],54)
        self.assertEqual(summary['uncertain_image_count'],6)
        self.assertEqual(summary['needs_user_review'],[])
        after = read(self.review/'review40.json')
        reviewed = [r for r in after['records'] if r.get('human_reviewed')]
        self.assertEqual(len(reviewed),18)
        for before,row in zip(self.before['records'],after['records']):
            for key in ('number','image_id','beam3','references',*HUMAN_FIELDS):
                self.assertEqual(before.get(key),row.get(key),f"Record {row['number']:02d}, {key}")
        csv_after = self.csv_rows()
        for old,row in zip(self.csv_before,csv_after):
            for key in old:
                self.assertEqual(old[key],row[key],f"Record {row['number']}, CSV {key}")
        self.assertEqual({c:sum(int(r[c]) for r in csv_after) for c in EXPECTED},EXPECTED)
        self.assertEqual(hashlib.sha256((self.review/'review40.csv').read_bytes()).hexdigest(),
                         summary['count_source_sha256'])
        document = (self.review/'review40.md').read_text(encoding='utf-8')
        self.assertNotIn('../../../data/',document)
        self.assertNotIn('![',document)
        self.assertEqual(len(re.findall(r'\*\*Görüntü kimliği:\*\*',document)),40)
        self.assertEqual(document.count('yerel Flickr8k kurulumu gerekir'),1)
        self.assertIn('18/18 kayıt tamamlandı',document)
        report_path = self.root/'report/model2_b_tamamlayici_20261112.md'
        report = report_path.read_text(encoding='utf-8')
        self.assertIn('18/18 kayıt tamamlandı',report)
        self.assertNotIn('incelenmeli',report)
        self.assertNotIn('sütunları sizin düzeltmeleriniz için boştur',report)
        links = re.findall(r'\[[^\]]*\]\(([^)]+)\)',report)
        self.assertEqual(sum(link.endswith('.png') for link in links),4)
        for link in links:
            self.assertTrue((report_path.parent/link).is_file(),link)
        self.assertEqual({p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (self.study/'attention').glob('*.png')},self.png_hashes)

    def test_preserve_all_decisions_and_repeat_without_drift(self):
        self.assertFalse((self.review/'records_pending.json').exists())
        self.run_stage('finish-review')
        self.run_stage('report')
        self.assert_final()
        before = hashes(self.root)
        self.run_stage('finish-review')
        self.run_stage('report')
        self.assertEqual(hashes(self.root),before)

    def test_recreate_missing_derived_files_from_canonical_decisions(self):
        for name in ['review40.csv','summary.json','review40.md']:
            (self.review/name).unlink()
        self.run_stage('finish-review')
        self.run_stage('report')
        self.assert_final()

    def test_conflicting_human_reason_stops_before_any_output_write(self):
        path = self.review/'annotations.json'
        annotations = read(path)
        annotations['annotations'][2]['reason'] = 'Conflicting decision for record 03'
        path.write_text(json.dumps(annotations,ensure_ascii=False),encoding='utf-8')
        before = hashes(self.root)
        result = self.run_stage('finish-review',success=False)
        self.assertIn('Record 03, reason',result.stderr)
        self.assertIn('no review output was written',result.stderr)
        self.assertEqual(hashes(self.root),before)


if __name__ == '__main__':
    unittest.main()
