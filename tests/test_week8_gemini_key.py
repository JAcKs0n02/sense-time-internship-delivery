import contextlib,io,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import setup_week8_gemini_key as setup
class GeminiKeyTests(unittest.TestCase):
 def test_missing_check_and_no_noninteractive_input(self):
  with tempfile.TemporaryDirectory() as d,patch.object(setup,'KEY_PATH',Path(d)/'gemini.key'),patch.object(setup.sys.stdin,'isatty',return_value=False),patch.object(setup.getpass,'getpass') as prompt,contextlib.redirect_stdout(io.StringIO()):
   self.assertEqual(setup.main(['--check']),1);self.assertEqual(setup.main([]),1);prompt.assert_not_called()
 def test_save_check_and_no_overwrite(self):
  secret='fake-test-key'
  with tempfile.TemporaryDirectory() as d,patch.object(setup,'KEY_PATH',Path(d)/'gemini.key'),patch.object(setup.sys.stdin,'isatty',return_value=True),patch.object(setup.getpass,'getpass',return_value=secret) as prompt,contextlib.redirect_stdout(io.StringIO()) as log:
   self.assertEqual(setup.main([]),0);self.assertEqual(setup.main(['--check']),0);self.assertEqual(setup.main([]),0)
   self.assertEqual(prompt.call_count,1);self.assertEqual(setup.KEY_PATH.stat().st_mode & 0o777,0o600);self.assertNotIn(secret,log.getvalue())
if __name__=='__main__':unittest.main()
