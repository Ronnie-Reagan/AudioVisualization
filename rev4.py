#!/usr/bin/env python3
"""
Audio visualizer - Rev 3.1 (Unlocked GUI framerate, newest-data-only)

Changes:
- Queue maxsize=1 with replace-on-full semantics: oldest pending block is dropped.
- AudioProcessor computes spectrum and stores latest snapshot (buffer, peak, spectrum, seq).
- GUI uses an unlocked QTimer (interval=0) to poll AudioProcessor.get_snapshot() and only redraws when seq advances.
- GUI and sound engine are decoupled: GUI refresh independent of incoming blocks.
- Thread-safe snapshot access via a lock.
"""

import sys
import os
import subprocess
import importlib
import venv
import queue
import time
import threading
from typing import Optional, Tuple

USE_VENV = False
try:
    from PyQt5 import QtWidgets
    app = QtWidgets.QApplication(sys.argv)
    reply = QtWidgets.QMessageBox.question(
        None,
        "Virtual Environment",
        "Do you want to create/use a virtual environment for this session?",
        QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
        QtWidgets.QMessageBox.Yes
    )
    USE_VENV = reply == QtWidgets.QMessageBox.Yes
    app.quit()
except Exception:
    resp = input("Do you want to create/use a virtual environment for this session? (Y/n): ").strip().lower()
    USE_VENV = (resp == "y" or resp == "")

# ----------------------------
# Virtual environment setup (unchanged)
# ----------------------------
ENV_DIR = "venv_audio_vis"
if USE_VENV:
    if not os.path.exists(ENV_DIR):
        print(f"[INFO] Creating virtual environment at {ENV_DIR}...")
        venv.create(ENV_DIR, with_pip=True)

    activate_script = (
        os.path.join(ENV_DIR, "Scripts", "activate_this.py")
        if os.name == "nt"
        else os.path.join(ENV_DIR, "bin", "activate_this.py")
    )

    if not hasattr(sys, 'real_prefix') and os.path.exists(activate_script):
        with open(activate_script) as f:
            code = compile(f.read(), activate_script, 'exec')
            exec(code, dict(__file__=activate_script))
        print(f"[INFO] Activated virtual environment: {ENV_DIR}")

    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "--no-user", "pip"])

# ----------------------------
# Import or install dependencies
# ----------------------------
REQUIRED_PACKAGES = ["PyQt5", "pyqtgraph", "sounddevice", "scipy", "numpy", "soundfile"]
modules = {}
for pkg in REQUIRED_PACKAGES:
    try:
        modules[pkg] = importlib.import_module(pkg)
        print(f"[INFO] Module '{pkg}' found.")
    except ImportError:
        if USE_VENV:
            print(f"[INFO] Installing missing package '{pkg}' in venv...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-user", pkg])
            modules[pkg] = importlib.import_module(pkg)
        else:
            raise ImportError(f"Module '{pkg}' not found and virtual environment usage declined.")

QtWidgets = modules['PyQt5'].QtWidgets
QtCore = modules['PyQt5'].QtCore
QtGui = modules['PyQt5'].QtGui
pg = modules['pyqtgraph']
sd = modules['sounddevice']
np = modules['numpy']
sf = modules['soundfile']

from sounddevice import InputStream
from scipy.fft import rfft, rfftfreq
from scipy import signal
from numpy import ndarray

# -------------------------
# Constants / defaults
# -------------------------
DEFAULT_SAMPLE_RATE = 48000
DEFAULT_CHUNK = 1024
DEFAULT_WINDOW_SECONDS = 0.1
TIMER_MS = 10  # legacy fallback; we now use unlocked GUI timer (interval=0)
SPECTRUM_MAX_FREQ = 10000
QSETTINGS_ORG = "DonLabs"
QSETTINGS_APP = "AudioVisualizer"

# -------------------------
# Audio Processor (Threaded)
# -------------------------
class AudioProcessor(QtCore.QThread):
    """
    Processes incoming audio blocks in a dedicated thread.
    Key design:
      - Uses a queue with maxsize=1 and replace-on-full semantics (so only newest pending block is kept).
      - Computes the FFT in the audio thread and stores a snapshot (rolling buffer, peak, spectrum, seq).
      - Exposes get_snapshot() for GUI to pull the latest snapshot under lock.
    """

    # Lightweight signal that indicates a new snapshot is available (no heavy payload).
    snapshot_available = QtCore.pyqtSignal()

    def __init__(self, sample_rate, chunk, parent=None, window_seconds=DEFAULT_WINDOW_SECONDS):
        super().__init__(parent)
        self.sample_rate = int(sample_rate)
        self.chunk = int(chunk)
        self.window_seconds = float(window_seconds)

        # Queue of pending blocks — maxsize=1 ensures we don't accumulate old blocks.
        self.audio_queue = queue.Queue(maxsize=1)

        self.running = False

        # Rolling buffer holds the latest window of audio (tail is newest)
        self.rolling_buffer = np.zeros(int(self.sample_rate * self.window_seconds), dtype=np.float32)
        self.peak = 1e-6

        # Latest spectrum and sequence number for snapshot exchange
        self.latest_spectrum = np.zeros(self.chunk // 2 + 1, dtype=np.float32)
        self._snapshot_seq = 0

        # Lock to protect rolling_buffer, peak, latest_spectrum, and seq
        self.buffer_lock = threading.Lock()

    def run(self):
        """Main loop: consume the single-slot queue, update rolling buffer, compute spectrum, update snapshot."""
        self.running = True
        while self.running:
            try:
                block = self.audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            # Ensure we have a 1-D float32 block with correct length
            try:
                block = np.asarray(block, dtype=np.float32).flatten()
                if block.size != self.chunk:
                    # If block is larger/smaller, pad/trim to chunk
                    if block.size > self.chunk:
                        block = block[-self.chunk:]
                    else:
                        block = np.pad(block, (0, self.chunk - block.size), mode='constant')
            except Exception:
                continue

            # Thread-safe update of rolling buffer and peak
            with self.buffer_lock:
                L = block.size
                buf_len = self.rolling_buffer.size
                if L >= buf_len:
                    self.rolling_buffer[:] = block[-buf_len:]
                else:
                    # slide left and append tail
                    self.rolling_buffer[:-L] = self.rolling_buffer[L:]
                    self.rolling_buffer[-L:] = block

                # Update peak with gentle decay
                current_peak = float(np.max(np.abs(self.rolling_buffer) + 1e-12))
                self.peak = max(current_peak, self.peak * 0.995)

            # Compute spectrum on the new block (outside lock for speed). This uses the raw block to reflect immediate freq content.
            yf = np.abs(rfft(block))
            yf_norm = (yf / (yf.max() + 1e-12)).astype(np.float32)

            # Store spectrum and bump sequence under lock (fast)
            with self.buffer_lock:
                # Resize latest_spectrum if chunk changed (unlikely at runtime, but safe)
                if self.latest_spectrum.shape[0] != yf_norm.shape[0]:
                    self.latest_spectrum = np.zeros_like(yf_norm)
                self.latest_spectrum[:] = yf_norm
                self._snapshot_seq += 1
                seq = self._snapshot_seq

            # Emit a lightweight signal (no big payload) so GUI can wake early if desired
            try:
                self.snapshot_available.emit()
            except Exception:
                # be resilient to signal errors
                pass

        # clean exit
        self.running = False

    def stop(self):
        """Signal thread to exit and wait for completion."""
        self.running = False
        self.wait(500)

    def push_block(self, block: ndarray):
        """Push incoming block into the single-slot queue.
        If queue is full, drop the currently waiting block (replace with newest).
        This enforces newest-data-only behavior.
        """
        try:
            self.audio_queue.put_nowait(block)
        except queue.Full:
            # Replace: remove oldest pending, then put newest
            try:
                _ = self.audio_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.audio_queue.put_nowait(block)
            except queue.Full:
                # if still full (race), just drop silently
                pass

    def update_buffer_size(self, new_size: int):
        """Thread-safe buffer resize: preserve tail of old buffer."""
        new_size = int(new_size)
        with self.buffer_lock:
            new_buffer = np.zeros(new_size, dtype=np.float32)
            copy_len = min(new_size, len(self.rolling_buffer))
            if copy_len > 0:
                new_buffer[-copy_len:] = self.rolling_buffer[-copy_len:]
            self.rolling_buffer = new_buffer

    def get_snapshot(self) -> Tuple[np.ndarray, float, np.ndarray, int]:
        """Return copies of (rolling_buffer, peak, latest_spectrum, seq).
        This is safe to call from GUI thread — copying under lock.
        """
        with self.buffer_lock:
            buf_copy = self.rolling_buffer.copy()
            peak_copy = float(self.peak)
            spec_copy = self.latest_spectrum.copy()
            seq_copy = int(self._snapshot_seq)
        return buf_copy, peak_copy, spec_copy, seq_copy

# -------------------------
# Audio engine (unchanged except minor safety)
# -------------------------
class AudioEngine(QtCore.QObject):
    """Manages sounddevice stream and device selection. Emits incoming audio blocks."""
    audio_block = QtCore.pyqtSignal(np.ndarray)

    def __init__(self, samplerate=DEFAULT_SAMPLE_RATE, chunk=DEFAULT_CHUNK, device: Optional[int] = None):
        super().__init__()
        self.samplerate = int(samplerate)
        self.chunk = int(chunk)
        self._device = device
        self._stream: Optional[InputStream] = None
        self._running = False

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, idx):
        self._device = idx

    def start(self):
        """Start the input stream. If device is None, pick first input device."""
        if self._running:
            return

        if self._device is None:
            try:
                self._device = self._pick_default_input()
            except RuntimeError as e:
                QtWidgets.QMessageBox.critical(None, "Device Error", f"Could not find input device:\n{e}")
                return

        try:
            self._stream = sd.InputStream(
                device=self._device,
                channels=1,
                samplerate=self.samplerate,
                blocksize=self.chunk,
                callback=self._sd_callback
            )
            self._stream.start()
            self._running = True
        except Exception as e:
            QtWidgets.QMessageBox.critical(None, "Audio Start Error", f"Could not start audio stream:\n{e}")
            self._running = False

    def stop(self):
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        self._running = False

    def restart(self):
        self.stop()
        QtCore.QThread.msleep(50)
        self.start()

    def _sd_callback(self, indata, frames, time_info, status):
        if status:
            print("InputStream status:", status, file=sys.stderr)
        try:
            block = indata[:, 0].copy()
            # Emit to UI -> connected handler should choose to push to processor
            self.audio_block.emit(block)
        except Exception as e:
            print("Callback exception:", e, file=sys.stderr)

    def _pick_default_input(self):
        devices = sd.query_devices()
        for idx, dev in enumerate(devices):
            if dev['max_input_channels'] > 0:
                return idx
        raise RuntimeError("No input device found")

    @staticmethod
    def list_devices():
        return sd.query_devices()

    @staticmethod
    def find_loopbacks():
        """Return indices of devices that look like loopback devices (heuristic)."""
        devs = sd.query_devices()
        hits = []
        for i, d in enumerate(devs):
            name = d.get('name', '').lower()
            if "loopback" in name or "loop back" in name or "stereo mix" in name:
                hits.append(i)
        return hits

# -------------------------
# Settings Dialog (unchanged)
# -------------------------
class SettingsDialog(QtWidgets.QDialog):
    settingsChanged = QtCore.pyqtSignal(dict)
    def __init__(self, parent=None, initial_settings=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setModal(True)
        self.setMinimumWidth(420)

        self._devices = sd.query_devices()
        self._hostapis = sd.query_hostapis()

        defaults = {
            "device_index": None,
            "use_loopback": False,
            "display_mode": "Last N seconds",
            "window_seconds": DEFAULT_WINDOW_SECONDS,
            "autoscale": True,
            "fixed_scale": 0.05,
            "high_amp_threshold_pct": 70,
            "color_low": "#FFFF00",
            "color_high": "#FF0000",
            "spectrum_smooth": True,
        }
        if initial_settings:
            defaults.update(initial_settings)
        self.settings = defaults
        self._build_ui()
        self.load_settings_to_ui()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout()
        form = QtWidgets.QFormLayout()
        self.device_combo = QtWidgets.QComboBox()
        for idx, d in enumerate(self._devices):
            chan = d.get('max_input_channels', 0)
            label = f"{idx}: {d.get('name')} ({chan} ch)"
            self.device_combo.addItem(label, userData=idx)
        form.addRow("Input device:", self.device_combo)

        self.loopback_check = QtWidgets.QCheckBox("Prefer loopback devices (if available)")
        form.addRow("", self.loopback_check)

        self.display_mode_combo = QtWidgets.QComboBox()
        self.display_mode_combo.addItems(["Full Graph", "Last N seconds"])
        form.addRow("Display mode:", self.display_mode_combo)

        self.window_spin = QtWidgets.QDoubleSpinBox()
        self.window_spin.setRange(0.01, 30.0)
        self.window_spin.setSingleStep(0.05)
        self.window_spin.setDecimals(2)
        form.addRow("Window (seconds):", self.window_spin)

        self.autoscale_check = QtWidgets.QCheckBox("Autoscale amplitude (dynamic zoom)")
        self.fixed_scale_spin = QtWidgets.QDoubleSpinBox()
        self.fixed_scale_spin.setRange(1e-6, 1.0)
        self.fixed_scale_spin.setDecimals(6)
        self.fixed_scale_spin.setSingleStep(0.001)
        form.addRow(self.autoscale_check, self.fixed_scale_spin)

        self.threshold_spin = QtWidgets.QSpinBox()
        self.threshold_spin.setRange(1, 200)
        form.addRow("High amplitude threshold (% of peak):", self.threshold_spin)

        self.color_low_btn = QtWidgets.QPushButton("Choose low-amplitude color")
        self.color_high_btn = QtWidgets.QPushButton("Choose high-amplitude color")
        self.color_low_lbl = QtWidgets.QLabel()
        self.color_high_lbl = QtWidgets.QLabel()
        color_row = QtWidgets.QHBoxLayout()
        color_row.addWidget(self.color_low_btn)
        color_row.addWidget(self.color_low_lbl)
        color_row.addWidget(self.color_high_btn)
        color_row.addWidget(self.color_high_lbl)
        form.addRow("Wave colors:", color_row)

        self.spectrum_smooth_check = QtWidgets.QCheckBox("Smooth spectrum")
        form.addRow("", self.spectrum_smooth_check)

        layout.addLayout(form)
        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok |
            QtWidgets.QDialogButtonBox.Cancel |
            QtWidgets.QDialogButtonBox.Apply
        )
        btns.accepted.connect(self._on_ok)
        btns.rejected.connect(self.reject)
        btns.button(QtWidgets.QDialogButtonBox.Apply).clicked.connect(self._on_apply)
        layout.addWidget(btns)
        self.setLayout(layout)
        self.color_low_btn.clicked.connect(lambda: self._pick_color("low"))
        self.color_high_btn.clicked.connect(lambda: self._pick_color("high"))

    def load_settings_to_ui(self):
        idx = self.settings.get("device_index")
        if idx is None and self.settings.get("use_loopback"):
            loops = AudioEngine.find_loopbacks()
            if loops:
                idx = loops[0]
        if idx is not None and 0 <= idx < len(self._devices):
            pos = next((i for i in range(self.device_combo.count())
                       if self.device_combo.itemData(i) == idx), 0)
            self.device_combo.setCurrentIndex(pos)
        self.loopback_check.setChecked(self.settings.get("use_loopback", False))
        self.display_mode_combo.setCurrentText(self.settings.get("display_mode", "Last N seconds"))
        self.window_spin.setValue(self.settings.get("window_seconds", DEFAULT_WINDOW_SECONDS))
        self.autoscale_check.setChecked(self.settings.get("autoscale", True))
        self.fixed_scale_spin.setValue(self.settings.get("fixed_scale", 0.05))
        self.threshold_spin.setValue(self.settings.get("high_amp_threshold_pct", 70))
        self.spectrum_smooth_check.setChecked(self.settings.get("spectrum_smooth", True))
        self._update_color_labels()

    def _update_color_labels(self):
        low = self.settings.get("color_low", "#FFFF00")
        high = self.settings.get("color_high", "#FF0000")
        self.color_low_lbl.setText(low)
        self.color_high_lbl.setText(high)
        self.color_low_lbl.setStyleSheet(f"background:{low}; padding:4px;")
        self.color_high_lbl.setStyleSheet(f"background:{high}; padding:4px;")

    def _pick_color(self, which):
        cur = self.settings.get("color_low" if which == "low" else "color_high")
        color = QtWidgets.QColorDialog.getColor(QtGui.QColor(cur), self, "Pick color")
        if color.isValid():
            hexc = color.name()
            if which == "low":
                self.settings["color_low"] = hexc
            else:
                self.settings["color_high"] = hexc
            self._update_color_labels()

    def _gather_settings(self):
        idx = self.device_combo.currentData()
        s = {
            "device_index": int(idx) if idx is not None else None,
            "use_loopback": bool(self.loopback_check.isChecked()),
            "display_mode": str(self.display_mode_combo.currentText()),
            "window_seconds": float(self.window_spin.value()),
            "autoscale": bool(self.autoscale_check.isChecked()),
            "fixed_scale": float(self.fixed_scale_spin.value()),
            "high_amp_threshold_pct": int(self.threshold_spin.value()),
            "color_low": self.settings.get("color_low"),
            "color_high": self.settings.get("color_high"),
            "spectrum_smooth": bool(self.spectrum_smooth_check.isChecked()),
        }
        return s

    def _on_apply(self):
        s = self._gather_settings()
        self.settingsChanged.emit(s)

    def _on_ok(self):
        self._on_apply()
        self.accept()

# -------------------------
# Main Visualizer widget
# -------------------------
class AudioVisualizer(QtWidgets.QMainWindow):
    """
    Visualizer updated to poll AudioProcessor snapshots at an unlocked framerate (interval=0).
    Only redraws when snapshot sequence number changes, guaranteeing newest-data-only updates.
    """
    def __init__(self):
        super().__init__()

        # --- Initialize color defaults before building UI ---
        self.color_low = pg.mkColor("#00ff00")
        self.color_high = pg.mkColor("#ff0000")
        self.wave_color = pg.mkColor("#ffff00")

        self.setWindowTitle("Audio Visualizer — Rev 3.1")
        self.setMinimumSize(900, 600)

        # Load persistent settings
        self.qsettings = QtCore.QSettings(QSETTINGS_ORG, QSETTINGS_APP)
        self._load_persistent_settings()

        # Audio engine
        self.engine = AudioEngine(
            samplerate=self.settings["samplerate"],
            chunk=self.settings["chunk"],
            device=self.settings.get("device_index")
        )
        # Connect live incoming audio to on_audio_block handler
        self.engine.audio_block.connect(self.on_audio_block)

        # Internal state
        self.window_seconds = max(0.05, float(self.settings.get("window_seconds", DEFAULT_WINDOW_SECONDS)))
        self.sample_rate = int(self.settings["samplerate"])
        self.chunk = int(self.settings["chunk"])
        self.buffer_size = int(self.sample_rate * self.window_seconds)
        self.rolling_buffer = np.zeros(self.buffer_size, dtype=np.float32)

        # File playback vars
        self.audio_file_data = None
        self.audio_file_ptr = 0
        self.is_playing_file = False
        self.file_playback_timer = QtCore.QTimer()
        self.file_playback_timer.timeout.connect(self._advance_file_playback)

        # Audio Processor
        self.processor = AudioProcessor(self.sample_rate, self.chunk, window_seconds=self.window_seconds)
        # GUI will poll get_snapshot() — also optionally listen to snapshot_available for faster wake
        self.processor.snapshot_available.connect(self._on_processor_snapshot_available)
        self.processor.start()

        # Plotting setup
        pg.setConfigOptions(antialias=True)
        self._build_ui()

        # Visualization state
        self.paused = False
        self.single_step = False
        self.peak = 1e-6
        self.autoscale = self.settings.get("autoscale", True)
        self.fixed_scale = self.settings.get("fixed_scale", 0.05)
        self.color_low = self.settings.get("color_low", "#FFFF00")
        self.color_high = self.settings.get("color_high", "#FF0000")
        self.threshold_pct = int(self.settings.get("high_amp_threshold_pct", 70))
        self.display_mode = self.settings.get("display_mode", "Last N seconds")
        self.spectrum_smooth = self.settings.get("spectrum_smooth", True)

        # FFT x-axis
        self.x_freq = rfftfreq(self.chunk, 1.0 / self.sample_rate)
        self.p_freq.setXRange(0, min(SPECTRUM_MAX_FREQ, np.max(self.x_freq)))
        self.p_freq.setYRange(0, 1.05, padding=0)

        # Start audio
        self.engine.start()

        # GUI timer: unlocked framerate (interval=0) — polls processor for newest snapshot
        self._last_snapshot_seq = -1
        self.gui_timer = QtCore.QTimer()
        # Use the most precise timer type available to get responsive updates
        self.gui_timer.setTimerType(QtCore.Qt.PreciseTimer)
        self.gui_timer.timeout.connect(self._on_gui_tick)
        # interval 0 => run as frequently as event loop allows, decoupled from audio blocks
        self.gui_timer.start(0)

        # Restore geometry
        geom = self.qsettings.value("geometry")
        if geom:
            self.restoreGeometry(geom)

    def _load_persistent_settings(self):
        self.settings = {
            "samplerate": DEFAULT_SAMPLE_RATE,
            "chunk": DEFAULT_CHUNK,
            "window_seconds": DEFAULT_WINDOW_SECONDS,
            "device_index": None,
            "autoscale": True,
            "fixed_scale": 0.05,
            "color_low": "#FFFF00",
            "color_high": "#FF0000",
            "high_amp_threshold_pct": 70,
            "display_mode": "Last N seconds",
            "spectrum_smooth": True,
        }
        for k in list(self.settings.keys()):
            v = self.qsettings.value(k)
            if v is not None:
                try:
                    if isinstance(self.settings[k], bool):
                        self.settings[k] = (str(v).lower() in ("1", "true", "yes"))
                    elif isinstance(self.settings[k], float):
                        self.settings[k] = float(v)
                    elif isinstance(self.settings[k], int):
                        self.settings[k] = int(v)
                    else:
                        self.settings[k] = v
                except Exception:
                    self.settings[k] = v

    def _save_persistent_settings(self):
        for k, v in self.settings.items():
            self.qsettings.setValue(k, v)
        self.qsettings.setValue("geometry", self.saveGeometry())

    def _build_ui(self):
        central = QtWidgets.QWidget()
        main_layout = QtWidgets.QVBoxLayout()
        central.setLayout(main_layout)
        self.setCentralWidget(central)

        toolbar = QtWidgets.QToolBar()
        self.addToolBar(toolbar)

        btn_load_file = QtWidgets.QAction("Load Audio File", self)
        btn_load_file.triggered.connect(self.load_audio_file)
        toolbar.addAction(btn_load_file)

        btn_settings = QtWidgets.QAction("Settings", self)
        btn_settings.triggered.connect(self.open_settings)
        toolbar.addAction(btn_settings)

        btn_pause = QtWidgets.QAction("Pause", self)
        btn_pause.setCheckable(True)
        btn_pause.triggered.connect(self.toggle_pause)
        toolbar.addAction(btn_pause)
        self._toolbar_pause_action = btn_pause

        btn_step = QtWidgets.QAction("Step", self)
        btn_step.triggered.connect(self.step_once)
        toolbar.addAction(btn_step)

        btn_reset = QtWidgets.QAction("Reset Zoom", self)
        btn_reset.triggered.connect(self.reset_zoom)
        toolbar.addAction(btn_reset)

        self.graphics = pg.GraphicsLayoutWidget()
        main_layout.addWidget(self.graphics)

        self.p_wave = self.graphics.addPlot(row=0, col=0, title="Waveform")
        vb = self.p_wave.getViewBox()
        vb.setMouseEnabled(x=True, y=True)
        self.p_wave.showGrid(True, True)
        self.p_wave.setLabel('bottom', 'Time', 's')
        self.p_wave.setLabel('left', 'Amplitude')
        self.curve_wave = self.p_wave.plot(pen=pg.mkPen(self.color_low, width=1.5))

        self.graphics.nextRow()
        self.p_freq = self.graphics.addPlot(row=1, col=0, title="Spectrum")
        self.p_freq.showGrid(True, True)
        self.p_freq.setLabel('bottom', 'Frequency', 'Hz')
        self.p_freq.setLabel('left', 'Magnitude (normalized)')
        self.curve_freq = self.p_freq.plot(pen=pg.mkPen("#00ffff", width=1.2))

        self.status = self.statusBar()
        self.lbl_peak = QtWidgets.QLabel("Peak: 0.000")
        self.status.addPermanentWidget(self.lbl_peak)

    # -------------------------
    # Incoming audio (from engine)
    # -------------------------
    @QtCore.pyqtSlot(ndarray)
    def on_audio_block(self, block: ndarray):
        """Push incoming block into processor. Respect pause/single-step logic."""
        if self.paused and not self.single_step:
            return

        # Always push newest block into processor queue (processor replaces older pending blocks)
        self.processor.push_block(block)

        # Single-step handling
        if self.single_step:
            # After pushing one block, we want to pause until user steps again
            self.single_step = False
            self.paused = True
            if hasattr(self, "_toolbar_pause_action"):
                self._toolbar_pause_action.setChecked(True)

    # -------------------------
    # GUI / Polling
    # -------------------------
    def _on_processor_snapshot_available(self):
        """Optional: this signal is fired when processor updates snapshot.
        We don't rely solely on it — the main polling timer handles updates — but we can use it to wake the GUI.
        """
        # Wake GUI timer if it's not running (it is running with interval=0). No-op to keep safe.
        return

    def _on_gui_tick(self):
        """Called extremely frequently (interval=0). Poll the processor for a snapshot and update only if seq advanced."""
        buf, peak, spectrum, seq = self.processor.get_snapshot()
        if seq == self._last_snapshot_seq:
            # no new data — skip
            return
        self._last_snapshot_seq = seq

        # Respect pause: if paused (and not single-step), do not update visuals (still keep newest data in processor)
        if self.paused and not self.single_step:
            return

        # Update local variables used by GUI
        self.rolling_buffer = buf
        self.peak = peak

        # Update waveform and spectrum visuals
        self._update_plots_from_snapshot(buf, spectrum)

        # If single_step was requested, pause after drawing this frame
        if self.single_step:
            self.single_step = False
            self.paused = True
            if hasattr(self, "_toolbar_pause_action"):
                self._toolbar_pause_action.setChecked(True)

    def _update_plots_from_snapshot(self, latest_buffer, latest_spectrum):
        """Perform plotting — lightweight plotting only when new snapshot arrived."""
        # Time axis (tail is newest)
        t = np.arange(-latest_buffer.size, 0) / self.sample_rate
        self.curve_wave.setData(t, latest_buffer)

        # Dynamic scaling
        if self.autoscale:
            margin = 1.15
            y_max = max(self.peak * margin, 1e-8)
            self.p_wave.setYRange(-y_max, y_max, padding=0.02)
        else:
            fs = float(self.fixed_scale)
            self.p_wave.setYRange(-fs, fs, padding=0.0)

        # Color based on most recent buffer's peak (compute on tail of rolling buffer)
        latest_tail = latest_buffer[-self.chunk:] if latest_buffer.size >= self.chunk else latest_buffer
        latest_abs_peak = float(np.max(np.abs(latest_tail) + 1e-12))
        pct = 0.0
        if self.peak > 0:
            pct = (latest_abs_peak / self.peak) * 100.0

        if pct >= self.threshold_pct:
            pen = pg.mkPen(self.color_high, width=1.8)
        else:
            pen = pg.mkPen(self.color_low, width=1.2)
        self.curve_wave.setPen(pen)

        # Spectrum smoothing (the spectrum passed in is already normalized by processor)
        spectrum_out = latest_spectrum
        if self.spectrum_smooth:
            kernel = np.ones(3) / 3.0
            spectrum_out = np.convolve(spectrum_out, kernel, mode='same')
        # Update spectrum plot
        self.curve_freq.setData(self.x_freq, spectrum_out)

    def _on_timer(self):
        """Legacy: kept for compatibility but not used for main drawing"""
        self._update_status()

    def _update_status(self):
        mode = "FILE" if self.is_playing_file else "LIVE"
        status_text = f"Peak: {self.peak:.6f} | Mode: {mode}"
        if self.paused:
            status_text += " | PAUSED"
        self.lbl_peak.setText(status_text)

    # -------------------------
    # File playback handling (mostly unchanged)
    # -------------------------
    def load_audio_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select audio file", "", "Audio Files (*.wav *.flac *.aiff *.ogg *.mp3)"
        )
        if not path:
            return

        self.engine.stop()
        self.is_playing_file = True

        try:
            data, sr = sf.read(path, always_2d=True)
            data = data[:, 0].astype(np.float32)

            if sr != self.sample_rate:
                print(f"[INFO] Resampling from {sr} Hz to {self.sample_rate} Hz...")
                num_samples = int(len(data) * self.sample_rate / sr)
                data = signal.resample(data, num_samples)
                sr = self.sample_rate

            self.audio_file_data = data
            self.audio_file_ptr = 0

            sd.stop()
            sd.play(data, samplerate=sr)

            interval_ms = int((self.chunk / self.sample_rate) * 1000)
            # Keep feeding visualization at block intervals separately from sd.play
            self.file_playback_timer.start(interval_ms)

            print(f"[INFO] Loaded {len(data)/sr:.2f}s audio file at {sr} Hz")

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "File Load Error", f"Could not load audio file:\n{e}")
            self.is_playing_file = False
            self.engine.start()

    def _advance_file_playback(self):
        if self.audio_file_data is None:
            self.file_playback_timer.stop()
            return

        if self.audio_file_ptr >= len(self.audio_file_data):
            self.file_playback_timer.stop()
            self.is_playing_file = False
            print("[INFO] File playback complete")
            return

        end_ptr = min(self.audio_file_ptr + self.chunk, len(self.audio_file_data))
        chunk = self.audio_file_data[self.audio_file_ptr:end_ptr]
        if len(chunk) < self.chunk:
            chunk = np.pad(chunk, (0, self.chunk - len(chunk)), mode='constant')
        self.audio_file_ptr = end_ptr

        if not self.paused or self.single_step:
            self.processor.push_block(chunk)
            if self.single_step:
                self.single_step = False
                self.paused = True

    # -------------------------
    # Controls
    # -------------------------
    def toggle_pause(self, checked=None):
        if checked is None:
            checked = not self.paused
        self.paused = bool(checked)
        if hasattr(self, "_toolbar_pause_action"):
            self._toolbar_pause_action.setChecked(self.paused)
        if self.paused:
            self.status.showMessage("PAUSED", 2000)
        else:
            self.status.clearMessage()

    def step_once(self):
        self.single_step = True

    def reset_zoom(self):
        self.p_wave.enableAutoRange()
        self.p_freq.enableAutoRange()

    # -------------------------
    # Settings dialog integration
    # -------------------------
    def open_settings(self):
        dlg = SettingsDialog(self, initial_settings=self.settings)
        dlg.settingsChanged.connect(self.apply_settings)
        dlg.exec_()

    def apply_settings(self, s: dict):
        old_dev = self.engine.device
        new_dev = s.get("device_index", old_dev)

        if s.get("use_loopback", False):
            loops = AudioEngine.find_loopbacks()
            if loops:
                new_dev = loops[0]

        if new_dev is not None and new_dev != old_dev:
            self.engine.device = new_dev
            self.settings["device_index"] = new_dev
            self._save_device_to_qsettings(new_dev)
            self.engine.restart()

        new_window = max(0.001, float(s.get("window_seconds", self.window_seconds)))
        if abs(new_window - self.window_seconds) > 0.001:
            self.window_seconds = new_window
            self.settings["window_seconds"] = self.window_seconds

            new_size = int(self.sample_rate * self.window_seconds)
            new_buffer = np.zeros(new_size, dtype=np.float32)
            copy_len = min(new_size, len(self.rolling_buffer))
            if copy_len > 0:
                new_buffer[-copy_len:] = self.rolling_buffer[-copy_len:]
            self.rolling_buffer = new_buffer
            self.buffer_size = new_size
            # Update processor buffer size
            self.processor.window_seconds = self.window_seconds
            self.processor.update_buffer_size(new_size)

        self.autoscale = bool(s.get("autoscale", self.autoscale))
        self.fixed_scale = float(s.get("fixed_scale", self.fixed_scale))
        self.settings["autoscale"] = self.autoscale
        self.settings["fixed_scale"] = self.fixed_scale

        self.color_low = s.get("color_low", self.color_low)
        self.color_high = s.get("color_high", self.color_high)
        self.threshold_pct = int(s.get("high_amp_threshold_pct", self.threshold_pct))
        self.settings["color_low"] = self.color_low
        self.settings["color_high"] = self.color_high
        self.settings["high_amp_threshold_pct"] = self.threshold_pct

        self.display_mode = s.get("display_mode", self.display_mode)
        self.settings["display_mode"] = self.display_mode

        self.spectrum_smooth = bool(s.get("spectrum_smooth", self.spectrum_smooth))
        self.settings["spectrum_smooth"] = self.spectrum_smooth

        self._save_persistent_settings()

    def _save_device_to_qsettings(self, devindex):
        self.settings["device_index"] = devindex
        self.qsettings.setValue("device_index", int(devindex))

    def closeEvent(self, event):
        self._save_persistent_settings()
        self.file_playback_timer.stop()
        try:
            self.processor.stop()
            self.engine.stop()
        except Exception:
            pass
        super().closeEvent(event)

# -------------------------
# Entry point
# -------------------------
def main():
    app = QtWidgets.QApplication(sys.argv)
    pg.setConfigOption('background', 'k')
    pg.setConfigOption('foreground', 'w')
    win = AudioVisualizer()
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
