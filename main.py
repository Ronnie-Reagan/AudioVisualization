import sys
import numpy as np
import sounddevice as sd
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
from scipy.fft import rfft, rfftfreq
import queue

SAMPLE_RATE = 48000
CHUNK = 1024
DISPLAY_DURATION = 0.1  # last 100ms of audio
DISPLAY_FREQ_MAX = 20000

BUFFER_SIZE = int(SAMPLE_RATE * DISPLAY_DURATION)
rolling_buffer = np.zeros(BUFFER_SIZE)
audio_q = queue.Queue()

# pick the first input device that actually works
def find_input_device():
    devices = sd.query_devices()
    for idx, dev in enumerate(devices):
        if dev['max_input_channels'] > 0:
            print(f"Using input device #{idx}: {dev['name']}")
            return idx
    raise RuntimeError("no input device found")

INPUT_DEVICE = find_input_device()
sd.default.device = INPUT_DEVICE

# just dump incoming audio into the queue
def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    audio_q.put(indata[:, 0].copy())

class AudioVisualizer(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

        self.stream = sd.InputStream(
            channels=1,
            samplerate=SAMPLE_RATE,
            blocksize=CHUNK,
            callback=audio_callback
        )
        self.stream.start()

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(10)

    def init_ui(self):
        self.win = pg.GraphicsLayoutWidget()
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.win)
        self.setLayout(layout)

        # waveform
        self.p_wave = self.win.addPlot(title="waveform")
        self.p_wave.showGrid(x=True, y=True)
        self.curve_wave = self.p_wave.plot(pen='y')

        # spectrum
        self.win.nextRow()
        self.p_freq = self.win.addPlot(title="spectrum")
        self.p_freq.setXRange(0, DISPLAY_FREQ_MAX)
        self.p_freq.showGrid(x=True, y=True)
        self.curve_freq = self.p_freq.plot(pen='c')

        self.x_time = np.linspace(-DISPLAY_DURATION, 0, BUFFER_SIZE)
        self.x_freq = rfftfreq(CHUNK, 1 / SAMPLE_RATE)

    def update_plot(self):
        global rolling_buffer
        if not audio_q.empty():
            y = audio_q.get()

            # scroll buffer
            rolling_buffer[:-len(y)] = rolling_buffer[len(y):]
            rolling_buffer[-len(y):] = y

            # auto zoom y-axis
            peak = max(np.max(np.abs(rolling_buffer)), 1e-6)
            self.p_wave.setYRange(-peak*1.1, peak*1.1)
            self.curve_wave.setData(self.x_time, rolling_buffer)

            yf = np.abs(rfft(y))
            yf /= np.max(yf) + 1e-10
            self.curve_freq.setData(self.x_freq, yf)

    def closeEvent(self, event):
        # stop audio on close
        self.stream.stop()
        self.stream.close()
        event.accept()

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    vis = AudioVisualizer()
    vis.show()
    sys.exit(app.exec_())
