"""Real socket checks for rapid deployment restart and occupied port rejection."""
from pathlib import Path
import socket
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import deploy

class DeploymentPortTests(unittest.TestCase):
    def test_closed_connection_does_not_block_restart(self):
        with socket.socket() as server, socket.socket() as client:
            server.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
            server.bind(('127.0.0.1',0))
            port=server.getsockname()[1]
            server.listen(1)
            client.connect(('127.0.0.1',port))
            conn,_=server.accept()
            # Server initiates close; its local port enters TIME_WAIT after ACK.
            conn.close()
            self.assertEqual(client.recv(1),b'')
        try:
            deploy.require_free_port(port)
        except OSError as exc:
            self.fail(f'Closed service must be restartable: {exc}')

    def test_listening_service_is_rejected(self):
        with socket.socket() as server:
            server.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
            server.bind(('127.0.0.1',0)); server.listen(1)
            with self.assertRaises(OSError):
                deploy.require_free_port(server.getsockname()[1])

if __name__=='__main__':unittest.main()
