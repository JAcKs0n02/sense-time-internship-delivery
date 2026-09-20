import os
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

class JudgeAccessTests(unittest.TestCase):
    def test_private_storage_and_no_overwrite(self):
        from setup_week8_judge_key import save_key, load_key
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'private' / 'key'
            save_key('test-not-a-real-key', path)
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            self.assertEqual(load_key(path), 'test-not-a-real-key')
            with self.assertRaises(FileExistsError):
                save_key('replacement', path)
            self.assertEqual(load_key(path), 'test-not-a-real-key')
            path.chmod(0o644)
            with self.assertRaisesRegex(ValueError, 'permissions'):
                load_key(path)

    def test_reject_invalid_or_linked_storage(self):
        from setup_week8_judge_key import save_key, load_key
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'key'
            for value in ('', 'a\nb', 'a b'):
                with self.assertRaises(ValueError):
                    save_key(value, path)
            target=Path(d)/'target';target.write_text('original')
            path.symlink_to(target)
            with self.assertRaises(FileExistsError):
                save_key('test-key', path)
            with self.assertRaises(ValueError):
                load_key(path)
            self.assertEqual(target.read_text(), 'original')

if __name__=='__main__':unittest.main()
