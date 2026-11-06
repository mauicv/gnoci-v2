import json
import socket
import threading

# How often the worker wakes up to check whether it should stop, in seconds.
POLL_TIMEOUT = 1.0


class Channel:
    """Streams the latest published value to one remote client at a time.

    publish() never blocks: it stores the value and wakes the worker thread,
    which sends it. If the worker falls behind, only the newest value is sent.
    Each message is a 4-byte big-endian length followed by that much JSON.
    """

    def __init__(
            self,
            host,
            port
        ):
        self.host = host
        self.port = port
        self._running = False
        self._thread = None
        # Created here, so publish() is safe before start().
        self._new_data_event = threading.Event()
        self._data = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            # Lets a restarted run bind straight away, without "address already in use".
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((self.host, self.port))
            s.listen()
            # Wake up from accept() regularly to check whether to stop.
            s.settimeout(POLL_TIMEOUT)
            print(f"Streaming on {self.host}:{self.port}")
            while self._running:
                try:
                    conn, addr = s.accept()
                except socket.timeout:
                    continue
                print(f"Stream client connected from {addr[0]}:{addr[1]}")
                with conn:
                    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    self._send_loop(conn)
                print(f"Stream client {addr[0]}:{addr[1]} disconnected")

    def _send_loop(self, conn):
        while self._running:
            if not self._new_data_event.wait(timeout=POLL_TIMEOUT):
                continue
            # Clear before reading, so a publish during the send isn't lost.
            self._new_data_event.clear()
            data = json.dumps(self._data).encode()
            try:
                conn.sendall(len(data).to_bytes(4, byteorder="big") + data)
            except OSError as e:
                print(f"Error sending data: {e}")
                return

    def publish(self, data):
        """Stores data (anything json.dumps accepts) for the worker to send."""
        self._data = data
        self._new_data_event.set()

    def stop(self):
        self._running = False
        if self._thread and self._thread is not threading.current_thread():
            self._thread.join()
