import socket
import struct
import time
from typing import Tuple
import numpy as np


class DelsysEMGIMU:
    """
    Trigno SDK TCP client that mimics the pytrigno-based API.

    * EMG stream:    TCP port (default 50043)
    * IMU/AUX stream:TCP port (default 50044)
    * Commands:      TCP port (default 50040) — used for START/STOP

    Assumptions (tweak in __init__ if needed):
      - Each "frame" is one float32 sample per active channel (no header).
      - EMG frame width = number of EMG channels selected.
      - IMU frame width = imu_axes_per_sensor * number of IMU channels selected.
      - Data are little-endian float32.

    If your build prepends headers, set header_bytes to the number of bytes to skip per frame.
    """

    def __init__(self,
                 emg_channel_range: Tuple[int, int] = (0, 0),
                 imu_channel_range: Tuple[int, int] = (0, 0),
                 emg_samples_per_read: int = 1,
                 imu_samples_per_read: int = 1,
                 host: str = '127.0.0.1',
                 emg_units: str = 'mV',
                 # SDK ports (change if your TCU is configured differently)
                 command_port: int = 50040,
                 emg_port: int = 50043,
                 aux_port: int = 50044,
                 # IMU axes per sensor: 3 (acc), 6 (acc+gyro), or 9 (acc+gyro+mag)
                 imu_axes_per_sensor: int = 9,
                 # If SDK adds per-frame headers, set this to header size in bytes
                 header_bytes: int = 0,
                 connect_timeout: float = 5.0,
                 socket_timeout: float = 1.0):
        """
        EMG is typically 2000 Hz; IMU accel ~148.1 Hz (Avanti default). Actual rates depend on TCU config.
        """
        self.host = host
        self.emg_units = emg_units

        # Ranges are inclusive (like pytrigno): (start_idx, end_idx)
        self.emg_channel_range = emg_channel_range
        self.imu_channel_range = imu_channel_range

        # Samples per read (like pytrigno)
        self.emg_samples_per_read = int(emg_samples_per_read)
        self.imu_samples_per_read = int(imu_samples_per_read)

        # Networking
        self.command_port = command_port
        self.emg_port = emg_port
        self.aux_port = aux_port
        self.connect_timeout = connect_timeout
        self.socket_timeout = socket_timeout

        # Payload/format
        assert imu_axes_per_sensor in (3, 6, 9), "imu_axes_per_sensor must be 3, 6, or 9."
        self.imu_axes_per_sensor = imu_axes_per_sensor
        self.header_bytes = int(header_bytes)

        # Sockets & buffers
        self._cmd_sock: socket.socket | None = None
        self._emg_sock: socket.socket | None = None
        self._aux_sock: socket.socket | None = None

        self._emg_buf = bytearray()
        self._aux_buf = bytearray()

        # Derived sizes
        self._n_emg = self._range_len(self.emg_channel_range)
        self._n_imu = self._range_len(self.imu_channel_range)

        self._emg_frame_floats = self._n_emg
        self._imu_frame_floats = self._n_imu * self.imu_axes_per_sensor

        self._emg_frame_bytes = self.header_bytes + self._emg_frame_floats * 4
        self._imu_frame_bytes = self.header_bytes + self._imu_frame_floats * 4

        # Structs for unpacking just the float payload
        self._emg_unpack = struct.Struct("<" + "f" * self._emg_frame_floats) if self._emg_frame_floats > 0 else None
        self._imu_unpack = struct.Struct("<" + "f" * self._imu_frame_floats) if self._imu_frame_floats > 0 else None

        self.is_running = False

    # ---------------- Public API (kept from your class) ----------------

    def start(self):
        """Open sockets and start streaming."""
        if self.is_running:
            return
        self._open_sockets()
        # Tell SDK to start (redundant on some builds, harmless if already running)
        self._send_command("START")
        self.is_running = True

    def read_emg(self):
        """
        Returns a NumPy array shaped (n_emg_channels, emg_samples_per_read)
        """
        self._ensure_running()
        if self._n_emg == 0:
            # Match pytrigno behavior: zero channels => empty array
            return np.empty((0, self.emg_samples_per_read), dtype=np.float32)
        return self._read_block(sock=self._emg_sock,
                                buf=self._emg_buf,
                                frame_bytes=self._emg_frame_bytes,
                                unpack=self._emg_unpack,
                                frames=self.emg_samples_per_read,
                                out_shape=(self._n_emg, self.emg_samples_per_read))

    def read_imu(self):
        """
        Returns a NumPy array shaped (n_imu_axes*n_imu_channels, imu_samples_per_read)
        Axes are stacked per sensor: for each sensor -> [x,y,z,(gx,gy,gz,(mx,my,mz))]
        """
        self._ensure_running()
        if self._n_imu == 0:
            return np.empty((0, self.imu_samples_per_read), dtype=np.float32)
        return self._read_block(sock=self._aux_sock,
                                buf=self._aux_buf,
                                frame_bytes=self._imu_frame_bytes,
                                unpack=self._imu_unpack,
                                frames=self.imu_samples_per_read,
                                out_shape=(self._imu_frame_floats, self.imu_samples_per_read))

    def read(self):
        """
        Read both streams once.
        Returns: {"emg": np.ndarray, "accel": np.ndarray}
        (The 'accel' key matches your original signature; it may contain >3 axes if configured.)
        """
        self._ensure_running()
        emg = self.read_emg()
        imu = self.read_imu()
        return {"emg": emg, "IMU": imu}

    def stop(self):
        """Stop streaming (soft) but keep sockets open."""
        if not self.is_running:
            return
        try:
            self._send_command("STOP")
        except Exception:
            pass
        self.is_running = False

    def disconnect(self):
        """
        Stop (if running) and close sockets.
        """
        if self.is_running:
            self.stop()

        for s in (self._emg_sock, self._aux_sock, self._cmd_sock):
            try:
                if s:
                    s.close()
            except Exception:
                pass

        self._cmd_sock = self._emg_sock = self._aux_sock = None
        self._emg_buf.clear()
        self._aux_buf.clear()

    # Channel management (kept for API parity)

    def set_emg_channel_range(self, channel_range: Tuple[int, int]):
        if self.is_running:
            raise RuntimeError("Cannot change EMG channel range while running.")
        self.emg_channel_range = channel_range
        self._recompute_emg_sizes()

    def set_imu_channel_range(self, channel_range: Tuple[int, int]):
        if self.is_running:
            raise RuntimeError("Cannot change IMU channel range while running.")
        self.imu_channel_range = channel_range
        self._recompute_imu_sizes()

    # ---------------- Internals ----------------

    @staticmethod
    def _range_len(r: Tuple[int, int]) -> int:
        a, b = r
        if b < a:
            return 0
        return (b - a + 1)

    def _recompute_emg_sizes(self):
        self._n_emg = self._range_len(self.emg_channel_range)
        self._emg_frame_floats = self._n_emg
        self._emg_frame_bytes = self.header_bytes + self._emg_frame_floats * 4
        self._emg_unpack = struct.Struct("<" + "f" * self._emg_frame_floats) if self._emg_frame_floats > 0 else None

    def _recompute_imu_sizes(self):
        self._n_imu = self._range_len(self.imu_channel_range)
        self._imu_frame_floats = self._n_imu * self.imu_axes_per_sensor
        self._imu_frame_bytes = self.header_bytes + self._imu_frame_floats * 4
        self._imu_unpack = struct.Struct("<" + "f" * self._imu_frame_floats) if self._imu_frame_floats > 0 else None

    def _open_sockets(self):
        # Command
        self._cmd_sock = socket.create_connection((self.host, self.command_port), timeout=self.connect_timeout)
        self._cmd_sock.settimeout(self.socket_timeout)
        # Data sockets
        self._emg_sock = socket.create_connection((self.host, self.emg_port), timeout=self.connect_timeout)
        self._emg_sock.settimeout(self.socket_timeout)
        self._aux_sock = socket.create_connection((self.host, self.aux_port), timeout=self.connect_timeout)
        self._aux_sock.settimeout(self.socket_timeout)

    def _send_command(self, cmd: str):
        if not self._cmd_sock:
            raise RuntimeError("Command socket not connected.")
        payload = cmd.encode("ascii") + b"\r\n"
        self._cmd_sock.sendall(payload)
        # Some SDK builds reply; ignore if they don't
        try:
            self._cmd_sock.settimeout(0.2)
            _ = self._cmd_sock.recv(512)
        except socket.timeout:
            pass
        finally:
            self._cmd_sock.settimeout(self.socket_timeout)

    def _ensure_running(self):
        if not self.is_running:
            raise RuntimeError("Device not started. Call start() before read().")

    def _read_block(self,
                    sock: socket.socket | None,
                    buf: bytearray,
                    frame_bytes: int,
                    unpack: struct.Struct | None,
                    frames: int,
                    out_shape: Tuple[int, int]) -> np.ndarray:
        """Read `frames` frames from `sock` and return as array shaped out_shape."""
        if sock is None:
            raise RuntimeError("Data socket not connected.")
        if unpack is None:
            # zero channels
            return np.empty(out_shape, dtype=np.float32)

        needed_frames = frames
        cols = []
        # Collect 'frames' columns; each column is one frame of floats
        while needed_frames > 0:
            # Ensure buffer has a full frame
            while len(buf) < frame_bytes:
                try:
                    chunk = sock.recv(4096)
                    if not chunk:
                        # No data; brief sleep to avoid spin
                        time.sleep(0.001)
                        continue
                    buf.extend(chunk)
                except socket.timeout:
                    # keep waiting; caller controls frequency
                    continue

            # Slice one frame
            if self.header_bytes:
                # drop header
                del buf[:self.header_bytes]
            frame = bytes(buf[:frame_bytes - self.header_bytes])
            del buf[:frame_bytes - self.header_bytes]

            values = unpack.unpack(frame)  # tuple of float32
            cols.append(np.asarray(values, dtype=np.float32))
            needed_frames -= 1

        # Stack as (channels, samples_per_read)
        data = np.column_stack(cols)
        # Safety: ensure final shape
        if data.shape != out_shape:
            # reshape if contiguous; this should not happen if sizes are correct
            data = data.reshape(out_shape)
        return data


if __name__ == "__main__":
    HOST = 'localhost'
    PORT_Cont = 50040
    PORT_EMG = 50043
    
    # create a TCP socket
    command_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    emg_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Connect sockets to ports
    command_socket.connect((HOST, PORT_Cont))
    print(f"Connected to {HOST}:{PORT_Cont} (command)")
    emg_socket.connect((HOST, PORT_EMG))
    print(f"Connected to {HOST}:{PORT_EMG} (EMG)")

    command_socket.sendall(b'START\r\n')
    print("Sent START command")

    data = emg_socket.recv(4096)
    print(f"Received {len(data)} bytes of EMG data")
    print(data)

    command_socket.sendall(b'STOP\r\n')
    print("Sent STOP command")
    
    command_socket.close()
    emg_socket.close()
    print("Sockets closed")
