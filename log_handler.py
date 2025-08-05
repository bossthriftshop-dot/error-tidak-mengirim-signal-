import logging
import socketio
import time

class WebServerHandler(logging.Handler):
    """
    A custom logging handler that sends log records to a web server via WebSockets.
    """
    def __init__(self, url):
        super().__init__()
        self.url = url
        self.sio = socketio.Client(reconnection_attempts=5, reconnection_delay=5)
        self.is_connected = False
        self.is_connecting = False # Re-entry guard

    def connect(self):
        """Establish connection to the Socket.IO server, with re-entry protection."""
        if self.is_connecting or self.is_connected:
            return

        self.is_connecting = True
        try:
            print(f"[LogHandler] Mencoba terhubung ke server log di {self.url}...")
            self.sio.connect(self.url, transports=['websocket'])
            self.is_connected = True
            print("[LogHandler] Berhasil terhubung ke server log.")
        except socketio.exceptions.ConnectionError as e:
            self.is_connected = False
            # Print directly to avoid re-triggering the handler
            print(f"[LogHandler Error] Gagal terhubung ke server log. Error: {e}")
        finally:
            self.is_connecting = False

    def emit(self, record):
        """
        Emits a log record by sending it to the web server via Socket.IO.
        """
        if not self.is_connected:
            self.connect()
            if not self.is_connected:
                # If still not connected, print to console and drop the log.
                print(f"[LogHandler Offline] {self.format(record)}")
                return

        try:
            log_entry = self.format(record)
            self.sio.emit('submit_log', {'message': log_entry})
        except Exception as e:
            print(f"[LogHandler Error] Gagal mengirim log via WebSocket. Error: {e}")
            self.is_connected = False

    def close(self):
        """Closes the Socket.IO connection."""
        if self.is_connected:
            self.sio.disconnect()
            self.is_connected = False
        super().close()
