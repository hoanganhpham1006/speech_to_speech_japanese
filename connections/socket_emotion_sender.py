import json
import logging
import socket
from baseHandler import BaseHandler

logger = logging.getLogger(__name__)

class SocketEmotionSender(BaseHandler):
    """
    Handles sending emotion data over a socket connection.
    """
    def setup(self, host="localhost", port=12348):
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.sock.connect((self.host, self.port))
            logger.info(f"Connected to {self.host}:{self.port}")
        except ConnectionRefusedError:
            logger.error(f"Failed to connect to {self.host}:{self.port}")
            self.sock = None

    def process(self, data):
        if self.sock is None:
            return
        try:
            json_data = json.dumps(data)
            self.sock.sendall(json_data.encode() + b'\n')
            yield None
        except Exception as e:
            logger.error(f"Error sending emotion data: {e}")
