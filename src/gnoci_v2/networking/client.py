import json
import socket


class Client:
    """Reads the messages a Channel streams: a 4-byte length, then that much JSON."""

    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.s = None

    def connect(self):
        self.s = socket.create_connection((self.host, self.port))

    def receive(self):
        """Blocks until the next message arrives. Returns None if the stream closed."""
        header = self._recv_all(4)
        if header is None:
            return None
        body = self._recv_all(int.from_bytes(header, byteorder="big"))
        if body is None:
            return None
        return json.loads(body.decode())

    def _recv_all(self, n):
        """Returns exactly n bytes, or None if the connection closes first."""
        data = bytearray()
        while len(data) < n:
            packet = self.s.recv(n - len(data))
            if not packet:
                return None
            data.extend(packet)
        return data

    def close(self):
        if self.s is not None:
            self.s.close()
            self.s = None
