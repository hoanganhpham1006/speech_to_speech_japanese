
import socket
import logging
from threading import Thread

logger = logging.getLogger(__name__)

class SocketTextSender:
    def __init__(self, stop_event, queue_in, host="0.0.0.0", port=12347):
        self.stop_event = stop_event
        self.queue_in = queue_in
        self.host = host
        self.port = port

    def run(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind((self.host, self.port))
        self.socket.listen(1)
        logger.info("TextSender waiting for connection...")
        conn, _ = self.socket.accept()
        logger.info("TextSender connected")

        try:
            while not self.stop_event.is_set():
                text_data = self.queue_in.get()
                if text_data is None:
                    break
                if isinstance(text_data, tuple):
                    text_data = f"[{text_data[0]}]: {text_data[1]}"  # e.g., "[user]: Hello"
                conn.sendall(text_data.encode('utf-8'))
                conn.sendall(b"\n")
        finally:
            conn.close()
            self.socket.close()
            logger.info("TextSender closed")