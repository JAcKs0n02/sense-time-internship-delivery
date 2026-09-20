import signal,subprocess,sys,unittest
from pathlib import Path
from unittest.mock import Mock,patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from week8_trained_session import stop_group
class SessionTests(unittest.TestCase):
 def test_stop_group_term(self):
  child=Mock(pid=321)
  with patch('week8_trained_session.os.killpg') as kill:
   stop_group(child);kill.assert_called_once_with(321,signal.SIGTERM);child.wait.assert_called_once_with(timeout=20)
 def test_escalate_timeout(self):
  child=Mock(pid=321);child.wait.side_effect=[subprocess.TimeoutExpired('child',20),0]
  with patch('week8_trained_session.os.killpg') as kill:
   stop_group(child);self.assertEqual(kill.call_args_list[1].args,(321,signal.SIGKILL));self.assertEqual(child.wait.call_count,2)
 def test_gone_child(self):
  child=Mock(pid=321)
  with patch('week8_trained_session.os.killpg',side_effect=ProcessLookupError):stop_group(child)
  child.wait.assert_not_called()
if __name__=='__main__':unittest.main()
