import ast
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.blip import collate_training, generation_options, selected_data


class BlipTests(unittest.TestCase):
    def test_decoding_is_deterministic_and_method_specific(self):
        self.assertEqual(generation_options('greedy')['num_beams'], 1)
        self.assertEqual(generation_options('beam')['num_beams'], 3)
        self.assertEqual(generation_options('beam', 5)['num_beams'], 5)
        self.assertFalse(generation_options('beam')['do_sample'])
        self.assertEqual(generation_options('greedy')['max_new_tokens'], 33)
        with self.assertRaises(ValueError):
            generation_options('nucleus')

    def test_no_test_split_before_any_io(self):
        with self.assertRaisesRegex(ValueError, 'Only train/val'):
            selected_data('missing', 'missing', 'missing', 'test')

    def test_test_split_only_with_explicit_allow_test(self):
        # allow_test passes the split guard; the missing files then fail at I/O.
        with self.assertRaises(OSError):
            selected_data('missing', 'missing', 'missing', 'test', allow_test=True)
        with self.assertRaisesRegex(ValueError, 'Only train/val'):
            selected_data('missing', 'missing', 'missing', 'unused', allow_test=True)

    def test_partition_checks_and_five_references(self):
        partition = {'train': [f't{i}' for i in range(6000)], 'val': [f'v{i}' for i in range(1000)], 'test': []}
        refs = {key: ['A dog runs.'] * 5 for key in partition['val']}
        with patch('src.blip.read_json', return_value=partition), \
                patch('src.blip.load_captions', return_value=refs), patch.object(Path, 'is_file', return_value=True):
            actual = selected_data('captions', 'images', 'split', 'val')
            self.assertEqual(len(actual), 1000)
            self.assertEqual(actual['v0'], ['a dog runs'] * 5)
            refs['v0'] = ['a dog']
            with self.assertRaisesRegex(ValueError, 'Five references'):
                selected_data('captions', 'images', 'split', 'val')
            partition['test'] = ['v0']
            with self.assertRaisesRegex(ValueError, 'overlapping'):
                selected_data('captions', 'images', 'split', 'val')

    def test_padding_labels_ignore_only_masked_tokens(self):
        import numpy as np
        class FakeTensor:
            def __init__(self, values): self.values = np.array(values)
            def clone(self): return FakeTensor(self.values.copy())
            def __setitem__(self, key, value): self.values[key] = value
            def __eq__(self, other): return self.values == other
            def masked_fill(self, mask, value):
                self.values[mask] = value
                return self
        def processor(**kwargs):
            self.assertEqual(kwargs['max_length'], 34)
            self.assertTrue(kwargs['truncation'])
            return {'input_ids': FakeTensor([[101, 8, 102, 0], [101, 9, 3, 102]]),
                    'attention_mask': FakeTensor([[1, 1, 1, 0], [1, 1, 1, 1]])}
        batch = collate_training([('image1', 'text1'), ('image2', 'text2')], processor, 103)
        self.assertEqual(batch['labels'].values.tolist(), [[103, 8, 102, -100], [103, 9, 3, 102]])
        self.assertEqual(batch['input_ids'].values[0, -1], 0)


if __name__ == '__main__':
    unittest.main()
