"""
Audio visualizer - Rev 2.0

Run:
    python rev2.py
"""

import sys
import os
import subprocess
import importlib
import venv
import queue
import time
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
    from numpy import ndarray
else:
    print("Unable to find / load modules in full; good luck")
# -------------------------
# Constants / defaults
# -------------------------
DEFAULT_SAMPLE_RATE = 48000
DEFAULT_CHUNK = 1024
DEFAULT_WINDOW_SECONDS = 0.1
TIMER_MS = 30  # GUI update interval
SPECTRUM_MAX_FREQ = 20000
QSETTINGS_ORG = "DonLabs"
QSETTINGS_APP = "AudioVisualizer"

# -------------------------
# Audio Processor (Threaded)
# -------------------------
class AudioProcessor(QtCore.QThread):
    # Signals to update GUI safely
    waveform_ready = QtCore.pyqtSignal(np.ndarray, float)  # rolling buffer, peak
    spectrum_ready = QtCore.pyqtSignal(np.ndarray)         # spectrum data

    def __init__(self, sample_rate, chunk, parent=None, window_seconds=DEFAULT_WINDOW_SECONDS):
        super().__init__(parent)
        self.sample_rate = sample_rate
        self.chunk = chunk
        self.window_seconds = window_seconds

        self.audio_queue = queue.Queue(maxsize=1024)
        self.running = False

        self.rolling_buffer = np.zeros(int(sample_rate * self.window_seconds), dtype=np.float32)
        self.peak = 1e-6

    def run(self):
        """Main processing loop."""
        self.running = True
        while self.running:
            try:
                block = self.audio_queue.get(timeout=0.1)  # wait for new audio block
            except queue.Empty:
                continue

            # Update rolling buffer
            L = block.size
            buf_len = self.rolling_buffer.size
            if L >= buf_len:
                self.rolling_buffer[:] = block[-buf_len:]
            else:
                self.rolling_buffer[:-L] = self.rolling_buffer[L:]
                self.rolling_buffer[-L:] = block

            # Update peak with gentle decay
            current_peak = float(np.max(np.abs(self.rolling_buffer) + 1e-12))
            self.peak = max(current_peak, self.peak * 0.995)

            # Compute spectrum
            yf = np.abs(rfft(block))
            yf_norm = yf / (yf.max() + 1e-12)

            # Emit signals to GUI
            self.waveform_ready.emit(self.rolling_buffer.copy(), self.peak)
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
            pass


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
        if self._device is None:
            self._device = self._pick_default_input()
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
            # Print or emit status if desired
            print("InputStream status:", status, file=sys.stderr)
        # indata is shape (frames, channels)
        # send the first channel as 1D numpy array
        try:
            block = indata[:, 0].copy()
            # emit to Qt main thread safely
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
            "color_low": "#FFFF00",   # yellow
            "color_high": "#FF0000",  # red
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

        # Loopback checkbox (for convenience)
        self.loopback_check = QtWidgets.QCheckBox("Prefer loopback devices (if available)")
        form.addRow("", self.loopback_check)

        # Display mode
        self.display_mode_combo = QtWidgets.QComboBox()
        self.display_mode_combo.addItems(["Full Graph", "Last N seconds"])
        form.addRow("Display mode:", self.display_mode_combo)

        # Window seconds spinner
        self.window_spin = QtWidgets.QDoubleSpinBox()
        self.window_spin.setRange(0.05, 30.0)
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
        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel | QtWidgets.QDialogButtonBox.Apply)
        btns.accepted.connect(self._on_ok)
        btns.rejected.connect(self.reject)
        btns.button(QtWidgets.QDialogButtonBox.Apply).clicked.connect(self._on_apply)
        layout.addWidget(btns)

        self.setLayout(layout)

        # connections
        self.color_low_btn.clicked.connect(lambda: self._pick_color("low"))
        self.color_high_btn.clicked.connect(lambda: self._pick_color("high"))

    def load_settings_to_ui(self):
        # Device
        idx = self.settings.get("device_index")
        if idx is None:
            # try to pick loopback if requested and exists
            if self.settings.get("use_loopback"):
                loops = AudioEngine.find_loopbacks()
                if loops:
                    idx = loops[0]
        if idx is not None and 0 <= idx < len(self._devices):
            pos = next((i for i in range(self.device_combo.count()) if self.device_combo.itemData(i) == idx), 0)
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
    def __init__(self):
        super().__init__()

        # --- Initialize color defaults before building UI ---
        self.color_low = pg.mkColor("#00ff00")   # low amplitude (greenish)
        self.color_high = pg.mkColor("#ff0000")  # high amplitude (reddish)
        self.wave_color = pg.mkColor("#ffff00")  # base waveform color (yellow)

        self.setWindowTitle("Audio Visualizer — Refined")
        self.setMinimumSize(900, 600)

        # Load persistent settings
        self.qsettings = QtCore.QSettings(QSETTINGS_ORG, QSETTINGS_APP)
        self._load_persistent_settings()

        # Audio engine
        self.engine = AudioEngine(samplerate=self.settings["samplerate"], chunk=self.settings["chunk"], device=self.settings.get("device_index"))
        self.engine.audio_block.connect(self.on_audio_block)

        self.audio_file_data = None
        self.audio_file_ptr = 0
        self.file_playback_timer = QtCore.QTimer()
        self.file_playback_timer.timeout.connect(self._advance_file_playback)
        self.is_playing_file = False

        # Internal buffers
        self.audio_q = queue.Queue(maxsize=1024)
        self.window_seconds = max(0.05, float(self.settings.get("window_seconds", DEFAULT_WINDOW_SECONDS)))
        self.sample_rate = self.settings["samplerate"]
        self.chunk = self.settings["chunk"]
        self.buffer_size = int(self.sample_rate * self.window_seconds)
        self.rolling_buffer = np.zeros(self.buffer_size, dtype=np.float32)

        # Audio Processor
        self.processor = AudioProcessor(self.sample_rate, self.chunk)
        self.processor.waveform_ready.connect(self._update_waveform)
        self.processor.spectrum_ready.connect(self._update_spectrum)
        self.processor.start()

        # plotting items
        pg.setConfigOptions(antialias=True)
        self._build_ui()

        # visualization state
        self.paused = False
        self.single_step = False
        self.peak = 1e-6
        self.peak_age = 0
        self.peak_decay_epochs = 20
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

        # start audio
        self.engine.start()

        # timer for UI updates
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._on_timer)
        self.timer.start(TIMER_MS)

        # restore geometry
        geom = self.qsettings.value("geometry")
        if geom:
            self.restoreGeometry(geom)

    def _load_persistent_settings(self):
        # defaults
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
        # override from QSettings if present
        for k in list(self.settings.keys()):
            v = self.qsettings.value(k)
            if v is not None:
                # attempt type conversions
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
        # store geometry too
        self.qsettings.setValue("geometry", self.saveGeometry())

    def _build_ui(self):
        # Central widget with pyqtgraph layout
        central = QtWidgets.QWidget()
        main_layout = QtWidgets.QVBoxLayout()
        central.setLayout(main_layout)
        self.setCentralWidget(central)

        # Toolbar
        toolbar = QtWidgets.QToolBar()
        self.addToolBar(toolbar)
        
        # load gaymer files
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

        # main plot area
        self.graphics = pg.GraphicsLayoutWidget()
        main_layout.addWidget(self.graphics)

        # waveform plot
        self.p_wave = self.graphics.addPlot(row=0, col=0, title="Waveform")
        vb = self.p_wave.getViewBox()
        vb.setMouseEnabled(x=True, y=True)
        self.p_wave.showGrid(True, True)
        self.curve_wave = self.p_wave.plot(pen=pg.mkPen(self.color_low, width=1.5))
        self.fill_wave = pg.FillBetweenItem(self.curve_wave, pg.PlotCurveItem(np.zeros(1)), brush=(50, 50, 50, 60))
        # (Don't add the fill item directly here; we'll manage visuals via curve pen.)
        self.p_wave.addItem(self.curve_wave)

        # frequency plot
        self.graphics.nextRow()
        self.p_freq = self.graphics.addPlot(row=1, col=0, title="Spectrum")
        self.p_freq.showGrid(True, True)
        self.curve_freq = self.p_freq.plot(pen=pg.mkPen("#00ffff", width=1.2))

        # status bar
        self.status = self.statusBar()
        self.lbl_peak = QtWidgets.QLabel("Peak: 0.000")
        self.status.addPermanentWidget(self.lbl_peak)

    @QtCore.pyqtSlot(np.ndarray, float)
    def _update_waveform(self, buffer: ndarray, peak: float):
        self.rolling_buffer = buffer
        self.peak = peak
        self._update_plots(buffer)

    @QtCore.pyqtSlot(np.ndarray)
    def _update_spectrum(self, spectrum: ndarray):
        # Update spectrum plot
        self.curve_freq.setData(self.x_freq, spectrum)

    # -------------------------
    # Audio & UI update handling
    # -------------------------
    @QtCore.pyqtSlot(ndarray)
    def on_audio_block(self, block: ndarray):
        self.processor.push_block(block)

    def _on_timer(self):
        # nothing heavy, just update GUI labels
        self._update_status()

    def _push_to_buffer(self, y):
        # ensure buffer size possibly changed if window_seconds changed
        desired_len = int(self.sample_rate * self.window_seconds)
        if desired_len != self.rolling_buffer.size:
            newbuf = np.zeros(desired_len, dtype=np.float32)
            # copy tail of old buffer into end of new buffer
            copy_len = min(desired_len, self.rolling_buffer.size)
            newbuf[-copy_len:] = self.rolling_buffer[-copy_len:]
            self.rolling_buffer = newbuf
            self.buffer_size = desired_len

        L = y.size
        if L >= self.rolling_buffer.size:
            # keep last portion only
            self.rolling_buffer[:] = y[-self.rolling_buffer.size:]
        else:
            self.rolling_buffer[:-L] = self.rolling_buffer[L:]
            self.rolling_buffer[-L:] = y

    def _update_peak_and_scale(self, y):
        current_peak = float(np.max(np.abs(self.rolling_buffer) + 1e-12))
        if current_peak > self.peak:
            self.peak = current_peak
            self.peak_age = 0
        else:
            self.peak_age += 1
            # gentle decay
            if self.peak_age > self.peak_decay_epochs:
                self.peak *= 0.995
                # avoid going below tiny floor
                self.peak = max(self.peak, 1e-8)

    def _update_plots(self, latest_block):
        # waveform
        t = np.linspace(-self.window_seconds, 0, self.rolling_buffer.size)
        self.curve_wave.setData(t, self.rolling_buffer)

        # dynamic scaling
        if self.autoscale:
            margin = 1.1
            y_max = max(self.peak * margin, 1e-8)
            self.p_wave.setYRange(-y_max, y_max, padding=0.0)
        else:
            fs = float(self.fixed_scale)
            self.p_wave.setYRange(-fs, fs, padding=0.0)

        # color based on latest peak % threshold
        latest_abs_peak = float(np.max(np.abs(latest_block) + 1e-12))
        # compute percent of global peak (avoid divide by zero)
        pct = 0.0
        if self.peak > 0:
            pct = (latest_abs_peak / self.peak) * 100.0
        if pct >= self.threshold_pct:
            pen = pg.mkPen(self.color_high, width=1.8)
        else:
            pen = pg.mkPen(self.color_low, width=1.2)
        self.curve_wave.setPen(pen)

        # spectrum
        yf = np.abs(rfft(latest_block))
        if self.spectrum_smooth:
            # simple smoothing by small moving average
            kernel = np.ones(3) / 3.0
            yf = np.convolve(yf, kernel, mode='same')
        # normalize
        norm = yf.max() if yf.max() > 0 else 1.0
        yf_norm = yf / (norm + 1e-12)
        self.curve_freq.setData(self.x_freq, yf_norm)

        self._update_status()

    def _update_status(self):
        self.lbl_peak.setText(f"Peak: {self.peak:.6f}")

    # -------------------------
    # Controls
    # -------------------------

    def load_audio_file(self):
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
            # Take first channel only
            data = data[:, 0].astype(np.float32)

            # FIXED: Don't change window_seconds, keep visualization window consistent
            # self.window_seconds stays as configured in settings

            # Handle sample rate mismatch
            if sr != self.sample_rate:
                # Resample audio to match current sample rate
                from scipy import signal
                num_samples = int(len(data) * self.sample_rate / sr)
                data = signal.resample(data, num_samples)
                sr = self.sample_rate

            self.audio_file_data = data
            self.audio_file_ptr = 0

            # Stop any previous playback
            sd.stop()

            # Play audio in background
            sd.play(data, samplerate=sr)

            # FIXED: Start feeding visualization
            # Calculate update interval to match chunk size
            interval_ms = int((self.chunk / self.sample_rate) * 1000)
            self.file_playback_timer.start(interval_ms)

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "File Load Error", f"Could not load audio file:\n{e}")
            self.is_playing_file = False
            self.engine.start()  # Resume live input

    def _advance_file_playback(self):
        """Feed chunks from loaded audio file to visualization"""
        if self.audio_file_data is None:
            self.file_playback_timer.stop()
            return
    
        # Check if we reached end of file
        if self.audio_file_ptr >= len(self.audio_file_data):
            self.file_playback_timer.stop()
            self.is_playing_file = False
            # Optionally restart live input
            self.engine.start()
            return
    
        # Extract next chunk
        end_ptr = min(self.audio_file_ptr + self.chunk, len(self.audio_file_data))
        chunk = self.audio_file_data[self.audio_file_ptr:end_ptr]
    
        # Pad if needed
        if len(chunk) < self.chunk:
            chunk = np.pad(chunk, (0, self.chunk - len(chunk)), mode='constant')
    
        self.audio_file_ptr = end_ptr
    
        # Feed to processor
        self.processor.push_block(chunk)
        
    def toggle_pause(self, checked=None):
        if checked is None:
            checked = not self.paused
        self.paused = bool(checked)
        # update toolbar check state if present
        if hasattr(self, "_toolbar_pause_action"):
            self._toolbar_pause_action.setChecked(self.paused)

    def step_once(self):
        # single-step: allow processing one audio block then pause again
        self.single_step = True
        self.paused = False

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
        # apply incoming settings dict immediately
        # stop/restart audio if device change
        old_dev = self.engine.device
        new_dev = s.get("device_index", old_dev)
        if s.get("use_loopback", False):
            loops = AudioEngine.find_loopbacks()
            if loops:
                new_dev = loops[0]

        if new_dev is not None and new_dev != old_dev:
            # change device
            self.engine.device = new_dev
            self.settings["device_index"] = new_dev
            self._save_device_to_qsettings(new_dev)
            self.engine.restart()

        # window seconds
        self.window_seconds = max(0.001, float(s.get("window_seconds", self.window_seconds)))
        self.settings["window_seconds"] = self.window_seconds

        # autoscale / fixed scale
        self.autoscale = bool(s.get("autoscale", self.autoscale))
        self.fixed_scale = float(s.get("fixed_scale", self.fixed_scale))
        self.settings["autoscale"] = self.autoscale
        self.settings["fixed_scale"] = self.fixed_scale

        # colors and threshold
        self.color_low = s.get("color_low", self.color_low)
        self.color_high = s.get("color_high", self.color_high)
        self.threshold_pct = int(s.get("high_amp_threshold_pct", self.threshold_pct))
        self.settings["color_low"] = self.color_low
        self.settings["color_high"] = self.color_high
        self.settings["high_amp_threshold_pct"] = self.threshold_pct

        # display mode
        self.display_mode = s.get("display_mode", self.display_mode)
        self.settings["display_mode"] = self.display_mode

        # spectrum smoothing
        self.spectrum_smooth = bool(s.get("spectrum_smooth", self.spectrum_smooth))
        self.settings["spectrum_smooth"] = self.spectrum_smooth

        # update visuals immediately
        self.curve_wave.setPen(pg.mkPen(self.color_low, width=1.5))
        self._save_persistent_settings()

    def _save_device_to_qsettings(self, devindex):
        self.settings["device_index"] = devindex
        self.qsettings.setValue("device_index", int(devindex))

    def closeEvent(self, event):
        # Save settings & stop audio
        self._save_persistent_settings()
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
    # Better default style for pyqtgraph text readability
    pg.setConfigOption('background', 'k')
    pg.setConfigOption('foreground', 'w')

    win = AudioVisualizer()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
