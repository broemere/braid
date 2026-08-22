"""Forward launch requests to the primary BRAID application process."""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Mapping

from PySide6.QtCore import QObject, Signal
from PySide6.QtNetwork import QLocalServer, QLocalSocket


log = logging.getLogger(__name__)
MAX_REQUEST_BYTES = 64 * 1024


def encode_request(request: Mapping[str, object]) -> bytes:
    """Serialize one application request using a newline-delimited message."""

    payload = json.dumps(dict(request), separators=(",", ":")).encode("utf-8")
    if len(payload) > MAX_REQUEST_BYTES:
        raise ValueError("Application request is too large.")
    return payload + b"\n"


def decode_request(payload: bytes) -> dict[str, object]:
    """Decode and minimally validate one application request."""

    if len(payload) > MAX_REQUEST_BYTES:
        raise ValueError("Application request is too large.")
    request = json.loads(payload.decode("utf-8"))
    if not isinstance(request, dict):
        raise ValueError("Application request must be a JSON object.")
    return request


class LocalApplicationRouter(QObject):
    """Own a local server or send a request to the process that owns it."""

    request_received = Signal(object)

    def __init__(self, server_name: str, parent=None):
        super().__init__(parent)
        self.server_name = server_name
        self._server = QLocalServer(self)
        self._server.setSocketOptions(QLocalServer.UserAccessOption)
        self._server.newConnection.connect(self._accept_connections)
        self._buffers: dict[QLocalSocket, bytearray] = {}

    @staticmethod
    def forward_request(
        server_name: str,
        request: Mapping[str, object],
        timeout_ms: int = 750,
    ) -> bool:
        """Return True after delivering a request to an existing process."""

        socket = QLocalSocket()
        socket.connectToServer(server_name)
        if not socket.waitForConnected(timeout_ms):
            return False

        payload = encode_request(request)
        if socket.write(payload) != len(payload):
            socket.abort()
            return False
        socket.flush()
        if socket.bytesToWrite() and not socket.waitForBytesWritten(timeout_ms):
            socket.abort()
            return False
        socket.disconnectFromServer()
        return True

    def listen(self, remove_stale_endpoint: bool = False) -> bool:
        """Try to become the primary process for future launch requests."""

        if remove_stale_endpoint and sys.platform != "win32":
            QLocalServer.removeServer(self.server_name)
        return self._server.listen(self.server_name)

    def close(self) -> None:
        """Stop accepting requests and release the local endpoint."""

        self._server.close()

    def _accept_connections(self) -> None:
        while self._server.hasPendingConnections():
            socket = self._server.nextPendingConnection()
            if socket is None:
                continue
            self._buffers[socket] = bytearray()
            socket.readyRead.connect(
                lambda connected_socket=socket: self._read_from(connected_socket)
            )
            socket.disconnected.connect(
                lambda connected_socket=socket: self._finish_connection(
                    connected_socket
                )
            )

    def _read_from(self, socket: QLocalSocket) -> None:
        buffer = self._buffers.get(socket)
        if buffer is None:
            return
        buffer.extend(bytes(socket.readAll()))
        if len(buffer) > MAX_REQUEST_BYTES + 1:
            log.warning("Discarding an oversized local application request.")
            socket.abort()
            self._buffers.pop(socket, None)
            return

        while b"\n" in buffer:
            line, remainder = buffer.split(b"\n", 1)
            buffer[:] = remainder
            if not line:
                continue
            try:
                request = decode_request(bytes(line))
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                log.warning("Ignoring an invalid local application request: %s", exc)
                continue
            self.request_received.emit(request)

    def _finish_connection(self, socket: QLocalSocket) -> None:
        self._read_from(socket)
        self._buffers.pop(socket, None)
        socket.deleteLater()
