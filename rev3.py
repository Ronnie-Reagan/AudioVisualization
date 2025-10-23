"""
Audio visualizer - Rev 3.0 (FIXED)

All critical and high-priority bugs resolved:
- File playback visualization now works
- Thread-safe buffer access with locks
- Sample rate resampling implemented
- Redundant FFT removed
- Pause functionality working
- Window resize syncs with processor
- Improved error handling

Run:
python rev3.py
"""

import sys
import os
import subprocess
import importlib
import venv
import queue
import time
import threading
from typing import Optional

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
    # If PyQt5 not installed yet, fallback to console prompt
    resp = input("Do you want to create/use a virtual environment for this session? (Y/n): ").strip().lower()
    USE_VENV = (resp == "y" or resp == "")

# ----------------------------
# Virtual environment setup
# ----------------------------
ENV_DIR = "venv_audio_vis"
if USE_VENV:
    if not os.path.exists(ENV_DIR):
        print(f"[INFO] Creating virtual environment at {ENV_DIR}...")
        venv.create(ENV_DIR, with_pip=True)

    # Determine activation script path
    activate_script = (
        os.path.join(ENV_DIR, "Scripts", "activate_this.py")
        if os.name == "nt"
        else os.path.join(ENV_DIR, "bin", "activate_this.py")
    )

    # Activate virtual environment
    if not hasattr(sys, 'real_prefix') and os.path.exists(activate_script):
        with open(activate_script) as f:
            code = compile(f.read(), activate_script, 'exec')
            exec(code, dict(__file__=activate_script))
        print(f"[INFO] Activated virtual environment: {ENV_DIR}")

    # Upgrade pip inside the venv to avoid install errors
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

# ----------------------------
# Assign module names for convenience
# ----------------------------
QtWidgets = modules['PyQt5'].QtWidgets
QtCore = modules['PyQt5'].QtCore
QtGui = modules['PyQt5'].QtGui
pg = modules['pyqtgraph']
sd = modules['sounddevice']
np = modules['numpy']
sf = modules['soundfile']

if sd and np:
    from sounddevice import InputStream
    from scipy.fft import rfft, rfftfreq
    from scipy import signal
    from numpy import ndarray
else:
    print("Unable to find / load modules in full; good luck")

# -------------------------
# Constants / defaults
# -------------------------
DEFAULT_SAMPLE_RATE = 48000
DEFAULT_CHUNK = 1024
DEFAULT_WINDOW_SECONDS = 0.1
TIMER_MS = 10  # GUI update interval
SPECTRUM_MAX_FREQ = 10000
QSETTINGS_ORG = "DonLabs"
QSETTINGS_APP = "AudioVisualizer"

# -------------------------
# Audio Processor (Threaded)
# -------------------------
class AudioProcessor(QtCore.QThread):
    """
    FIXED: Added thread-safe buffer access and emits latest_block for color calculation
    """
    # Signals to update GUI safely
    waveform_ready = QtCore.pyqtSignal(np.ndarray, float, np.ndarray)  # buffer, peak, latest_block
    spectrum_ready = QtCore.pyqtSignal(np.ndarray)  # spectrum data

    def __init__(self, sample_rate, chunk, parent=None, window_seconds=DEFAULT_WINDOW_SECONDS):
        super().__init__(parent)
        self.sample_rate = sample_rate
        self.chunk = chunk
        self.window_seconds = window_seconds
        self.audio_queue = queue.Queue(maxsize=1024)
        self.running = False
        self.rolling_buffer = np.zeros(int(sample_rate * self.window_seconds), dtype=np.float32)
        self.peak = 1e-6

        # FIXED: Add thread lock for buffer safety
        self.buffer_lock = threading.Lock()

    def run(self):
        """Main processing loop."""
        self.running = True
        while self.running:
            try:
                block = self.audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            # FIXED: Thread-safe buffer update
            L = block.size
            buf_len = self.rolling_buffer.size

            with self.buffer_lock:
                if L >= buf_len:
                    self.rolling_buffer[:] = block[-buf_len:]
                else:
                    self.rolling_buffer[:-L] = self.rolling_buffer[L:]
                    self.rolling_buffer[-L:] = block

                # Update peak with gentle decay
                current_peak = float(np.max(np.abs(self.rolling_buffer) + 1e-12))
                self.peak = max(current_peak, self.peak * 0.995)

                # Make copies for signal emission
                buffer_copy = self.rolling_buffer.copy()
                peak_copy = self.peak
                block_copy = block.copy()

            # Compute spectrum (outside lock)
            yf = np.abs(rfft(block))
            yf_norm = yf / (yf.max() + 1e-12)

            # FIXED: Emit latest_block for color calculation
            self.waveform_ready.emit(buffer_copy, peak_copy, block_copy)
            self.spectrum_ready.emit(yf_norm)

    def stop(self):
        """Signal thread to exit and wait for completion."""
        self.running = False
        self.wait()

    def push_block(self, block: ndarray):
        """Non-blocking push into queue."""
        try:
            self.audio_queue.put_nowait(block)
        except queue.Full:
            # Drop block if queue is full to avoid blocking GUI
            queue.Empty()
            self.audio_queue.put_nowait(block)
            pass

    # FIXED: Add method to update buffer size
    def update_buffer_size(self, new_size: int):
        """Thread-safe buffer resize"""
        with self.buffer_lock:
            new_buffer = np.zeros(new_size, dtype=np.float32)
            copy_len = min(new_size, len(self.rolling_buffer))
            new_buffer[-copy_len:] = self.rolling_buffer[-copy_len:]
            self.rolling_buffer = new_buffer

# -------------------------
# Audio engine
# -------------------------
class AudioEngine(QtCore.QObject):
    """Manages sounddevice stream and device selection. Emits incoming audio blocks."""
    audio_block = QtCore.pyqtSignal(np.ndarray)

    def __init__(self, samplerate=DEFAULT_SAMPLE_RATE, chunk=DEFAULT_CHUNK, device: Optional[int] = None):
        super().__init__()
        self.samplerate = samplerate
        self.chunk = chunk
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

        # FIXED: Better error handling for device selection
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
# Settings Dialog
# -------------------------
class SettingsDialog(QtWidgets.QDialog):
    """Dialog to edit visualizer settings. Emits settingsChanged(dict)."""
    settingsChanged = QtCore.pyqtSignal(dict)

    def __init__(self, parent=None, initial_settings=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setModal(True)
        self.setMinimumWidth(420)

        self._devices = sd.query_devices()
        self._hostapis = sd.query_hostapis()

        # Default settings
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

        # Device combo
        self.device_combo = QtWidgets.QComboBox()
        for idx, d in enumerate(self._devices):
            chan = d.get('max_input_channels', 0)
            label = f"{idx}: {d.get('name')} ({chan} ch)"
            self.device_combo.addItem(label, userData=idx)
        form.addRow("Input device:", self.device_combo)

        # Loopback checkbox
        self.loopback_check = QtWidgets.QCheckBox("Prefer loopback devices (if available)")
        form.addRow("", self.loopback_check)

        # Display mode
        self.display_mode_combo = QtWidgets.QComboBox()
        self.display_mode_combo.addItems(["Full Graph", "Last N seconds"])
        form.addRow("Display mode:", self.display_mode_combo)

        # Window seconds spinner
        self.window_spin = QtWidgets.QDoubleSpinBox()
        self.window_spin.setRange(0.01, 30.0)
        self.window_spin.setSingleStep(0.05)
        self.window_spin.setDecimals(2)
        form.addRow("Window (seconds):", self.window_spin)

        # Autoscale / fixed scale
        self.autoscale_check = QtWidgets.QCheckBox("Autoscale amplitude (dynamic zoom)")
        self.fixed_scale_spin = QtWidgets.QDoubleSpinBox()
        self.fixed_scale_spin.setRange(1e-6, 1.0)
        self.fixed_scale_spin.setDecimals(6)
        self.fixed_scale_spin.setSingleStep(0.001)
        form.addRow(self.autoscale_check, self.fixed_scale_spin)

        # High amplitude threshold (percent)
        self.threshold_spin = QtWidgets.QSpinBox()
        self.threshold_spin.setRange(1, 200)
        form.addRow("High amplitude threshold (% of peak):", self.threshold_spin)

        # Color pickers
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

        # Spectrum smoothing
        self.spectrum_smooth_check = QtWidgets.QCheckBox("Smooth spectrum")
        form.addRow("", self.spectrum_smooth_check)

        layout.addLayout(form)

        # Buttons
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

        # Connections
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
    FIXED: Multiple improvements
    - File playback now feeds visualization via timer
    - Pause functionality working
    - Sample rate resampling
    - Redundant FFT removed
    - Thread-safe buffer access
    """
    def __init__(self):
        super().__init__()


        # --- Initialize color defaults before building UI ---
        self.color_low = pg.mkColor("#00ff00")   # low amplitude (greenish)
        self.color_high = pg.mkColor("#ff0000")  # high amplitude (reddish)
        self.wave_color = pg.mkColor("#ffff00")  # base waveform color (yellow)

        self.setWindowTitle("Audio Visualizer — Rev 3.0 FIXED")
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
        self.engine.audio_block.connect(self.on_audio_block)

        # Internal state
        self.window_seconds = max(0.05, float(self.settings.get("window_seconds", DEFAULT_WINDOW_SECONDS)))
        self.sample_rate = self.settings["samplerate"]
        self.chunk = self.settings["chunk"]
        self.buffer_size = int(self.sample_rate * self.window_seconds)
        self.rolling_buffer = np.zeros(self.buffer_size, dtype=np.float32)

        # FIXED: File playback variables
        self.audio_file_data = None
        self.audio_file_ptr = 0
        self.is_playing_file = False
        self.file_playback_timer = QtCore.QTimer()
        self.file_playback_timer.timeout.connect(self._advance_file_playback)

        # Audio Processor
        self.processor = AudioProcessor(self.sample_rate, self.chunk, window_seconds=self.window_seconds)
        self.processor.waveform_ready.connect(self._update_waveform)
        self.processor.spectrum_ready.connect(self._update_spectrum)
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
        self.p_freq.setYRange(0, 1.05, padding=0)  # FIXED: Set explicit Y range for normalized spectrum

        # Start audio
        self.engine.start()

        # Timer for UI updates
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._on_timer)
        self.timer.start(TIMER_MS)

        # Restore geometry
        geom = self.qsettings.value("geometry")
        if geom:
            self.restoreGeometry(geom)

    def _load_persistent_settings(self):
        # Defaults
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

        # Override from QSettings if present
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
        # Central widget
        central = QtWidgets.QWidget()
        main_layout = QtWidgets.QVBoxLayout()
        central.setLayout(main_layout)
        self.setCentralWidget(central)

        # Toolbar
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

        # Main plot area
        self.graphics = pg.GraphicsLayoutWidget()
        main_layout.addWidget(self.graphics)

        # Waveform plot
        self.p_wave = self.graphics.addPlot(row=0, col=0, title="Waveform")
        vb = self.p_wave.getViewBox()
        vb.setMouseEnabled(x=True, y=True)
        self.p_wave.showGrid(True, True)
        self.p_wave.setLabel('bottom', 'Time', 's')
        self.p_wave.setLabel('left', 'Amplitude')
        self.curve_wave = self.p_wave.plot(pen=pg.mkPen(self.color_low, width=1.5))

        # Frequency plot
        self.graphics.nextRow()
        self.p_freq = self.graphics.addPlot(row=1, col=0, title="Spectrum")
        self.p_freq.showGrid(True, True)
        self.p_freq.setLabel('bottom', 'Frequency', 'Hz')
        self.p_freq.setLabel('left', 'Magnitude (normalized)')
        self.curve_freq = self.p_freq.plot(pen=pg.mkPen("#00ffff", width=1.2))

        # Status bar
        self.status = self.statusBar()
        self.lbl_peak = QtWidgets.QLabel("Peak: 0.000")
        self.status.addPermanentWidget(self.lbl_peak)

    # -------------------------
    # Signal handlers
    # -------------------------
    @QtCore.pyqtSlot(np.ndarray, float, np.ndarray)
    def _update_waveform(self, buffer: ndarray, peak: float, latest_block: ndarray):
        """FIXED: Now receives latest_block for threshold color calculation"""
        self.rolling_buffer = buffer
        self.peak = peak
        self._update_plots(latest_block)

    @QtCore.pyqtSlot(np.ndarray)
    def _update_spectrum(self, spectrum: ndarray):
        """FIXED: Apply smoothing here if enabled, use provided spectrum"""
        if self.spectrum_smooth:
            kernel = np.ones(3) / 3.0
            spectrum = np.convolve(spectrum, kernel, mode='same')

        # Update spectrum plot
        self.curve_freq.setData(self.x_freq, spectrum)

    @QtCore.pyqtSlot(ndarray)
    def on_audio_block(self, block: ndarray):
        """FIXED: Respect pause state"""
        if self.paused and not self.single_step:
            return  # Don't process when paused

        self.processor.push_block(block)

        # Reset single step flag
        if self.single_step:
            self.single_step = False
            self.paused = True
            if hasattr(self, "_toolbar_pause_action"):
                self._toolbar_pause_action.setChecked(True)

    def _on_timer(self):
        """Timer for periodic UI updates"""
        self._update_status()

    def _update_plots(self, latest_block):
        """FIXED: Time axis calculation, removed redundant FFT"""
        # FIXED: Use accurate time axis calculation
        t = np.arange(-self.rolling_buffer.size, 0) / self.sample_rate
        self.curve_wave.setData(t, self.rolling_buffer)

        # Dynamic scaling
        if self.autoscale:
            margin = 1.15  # FIXED: Increased margin to avoid clipping
            y_max = max(self.peak * margin, 1e-8)
            self.p_wave.setYRange(-y_max, y_max, padding=0.02)
        else:
            fs = float(self.fixed_scale)
            self.p_wave.setYRange(-fs, fs, padding=0.0)

        # Color based on latest peak % threshold
        latest_abs_peak = float(np.max(np.abs(latest_block) + 1e-12))
        pct = 0.0
        if self.peak > 0:
            pct = (latest_abs_peak / self.peak) * 100.0

        if pct >= self.threshold_pct:
            pen = pg.mkPen(self.color_high, width=1.8)
        else:
            pen = pg.mkPen(self.color_low, width=1.2)

        self.curve_wave.setPen(pen)

        # FIXED: Removed redundant FFT calculation - now done in processor only

    def _update_status(self):
        """Update status bar"""
        mode = "FILE" if self.is_playing_file else "LIVE"
        status_text = f"Peak: {self.peak:.6f} | Mode: {mode}"
        if self.paused:
            status_text += " | PAUSED"
        self.lbl_peak.setText(status_text)

    # -------------------------
    # File playback handling
    # -------------------------
    def load_audio_file(self):
        """FIXED: Complete file playback with visualization"""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select audio file", "", "Audio Files (*.wav *.flac *.aiff *.ogg *.mp3)"
        )
        if not path:
            return

        # Stop live input temporarily
        self.engine.stop()
        self.is_playing_file = True

        try:
            data, sr = sf.read(path, always_2d=True)
            data = data[:, 0].astype(np.float32)

            # FIXED: Handle sample rate mismatch with resampling
            if sr != self.sample_rate:
                print(f"[INFO] Resampling from {sr} Hz to {self.sample_rate} Hz...")
                num_samples = int(len(data) * self.sample_rate / sr)
                data = signal.resample(data, num_samples)
                sr = self.sample_rate

            # FIXED: Don't change window_seconds - keep visualization consistent
            self.audio_file_data = data
            self.audio_file_ptr = 0

            # Stop any previous playback
            sd.stop()

            # Play audio in background
            sd.play(data, samplerate=sr)

            # FIXED: Start feeding visualization
            interval_ms = int((self.chunk / self.sample_rate) * 1000)
            self.file_playback_timer.start(interval_ms)

            print(f"[INFO] Loaded {len(data)/sr:.2f}s audio file at {sr} Hz")

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "File Load Error", f"Could not load audio file:\n{e}")
            self.is_playing_file = False
            self.engine.start()  # Resume live input

    def _advance_file_playback(self):
        """FIXED: Feed chunks from loaded audio file to visualization"""
        if self.audio_file_data is None:
            self.file_playback_timer.stop()
            return

        # Check if we reached end of file
        if self.audio_file_ptr >= len(self.audio_file_data):
            self.file_playback_timer.stop()
            self.is_playing_file = False
            print("[INFO] File playback complete")
            # Optionally restart live input
            # self.engine.start()
            return

        # Extract next chunk
        end_ptr = min(self.audio_file_ptr + self.chunk, len(self.audio_file_data))
        chunk = self.audio_file_data[self.audio_file_ptr:end_ptr]

        # Pad if needed
        if len(chunk) < self.chunk:
            chunk = np.pad(chunk, (0, self.chunk - len(chunk)), mode='constant')

        self.audio_file_ptr = end_ptr

        # Feed to processor (unless paused)
        if not self.paused or self.single_step:
            self.processor.push_block(chunk)
            if self.single_step:
                self.single_step = False
                self.paused = True

    # -------------------------
    # Controls
    # -------------------------
    def toggle_pause(self, checked=None):
        """FIXED: Fully implemented pause functionality"""
        if checked is None:
            checked = not self.paused
        self.paused = bool(checked)

        # Update UI feedback
        if hasattr(self, "_toolbar_pause_action"):
            self._toolbar_pause_action.setChecked(self.paused)

        # Update status
        if self.paused:
            self.status.showMessage("PAUSED", 2000)
        else:
            self.status.clearMessage()

    def step_once(self):
        """Single-step: process one block then pause"""
        self.single_step = True

    def reset_zoom(self):
        """Reset plot zoom to auto-range"""
        self.p_wave.enableAutoRange()
        self.p_freq.enableAutoRange()

    # -------------------------
    # Settings dialog integration
    # -------------------------
    def open_settings(self):
        """Open settings dialog"""
        dlg = SettingsDialog(self, initial_settings=self.settings)
        dlg.settingsChanged.connect(self.apply_settings)
        dlg.exec_()

    def apply_settings(self, s: dict):
        """FIXED: Properly sync settings with processor"""
        # Handle device change
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

        # FIXED: Handle window_seconds change and sync with processor
        new_window = max(0.001, float(s.get("window_seconds", self.window_seconds)))
        if abs(new_window - self.window_seconds) > 0.001:
            self.window_seconds = new_window
            self.settings["window_seconds"] = self.window_seconds

            # Update local buffer
            new_size = int(self.sample_rate * self.window_seconds)
            new_buffer = np.zeros(new_size, dtype=np.float32)

            # Copy tail of old buffer
            copy_len = min(new_size, len(self.rolling_buffer))
            new_buffer[-copy_len:] = self.rolling_buffer[-copy_len:]

            self.rolling_buffer = new_buffer
            self.buffer_size = new_size

            # FIXED: Update processor's buffer too
            self.processor.window_seconds = self.window_seconds
            self.processor.update_buffer_size(new_size)

        # Other settings
        self.autoscale = bool(s.get("autoscale", self.autoscale))
        self.fixed_scale = float(s.get("fixed_scale", self.fixed_scale))
        self.settings["autoscale"] = self.autoscale
        self.settings["fixed_scale"] = self.fixed_scale

        # Colors and threshold
        self.color_low = s.get("color_low", self.color_low)
        self.color_high = s.get("color_high", self.color_high)
        self.threshold_pct = int(s.get("high_amp_threshold_pct", self.threshold_pct))
        self.settings["color_low"] = self.color_low
        self.settings["color_high"] = self.color_high
        self.settings["high_amp_threshold_pct"] = self.threshold_pct

        # Display mode
        self.display_mode = s.get("display_mode", self.display_mode)
        self.settings["display_mode"] = self.display_mode

        # Spectrum smoothing
        self.spectrum_smooth = bool(s.get("spectrum_smooth", self.spectrum_smooth))
        self.settings["spectrum_smooth"] = self.spectrum_smooth

        # Save settings
        self._save_persistent_settings()

    def _save_device_to_qsettings(self, devindex):
        """Save device index to persistent settings"""
        self.settings["device_index"] = devindex
        self.qsettings.setValue("device_index", int(devindex))

    def closeEvent(self, event):
        """Clean shutdown"""
        self._save_persistent_settings()

        # Stop timers
        self.file_playback_timer.stop()
        self.timer.stop()

        # FIXED: Clean up audio file data to prevent memory leaks
        self.audio_file_data = None

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

    # Better default style for pyqtgraph
    pg.setConfigOption('background', 'k')
    pg.setConfigOption('foreground', 'w')

    win = AudioVisualizer()
    win.show()

    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
