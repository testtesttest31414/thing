#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
B200mini-i Control Panel

A multi-purpose tool for the Ettus Research B200mini-i SDR that provides:
- A real-time Spectrum Analyzer
- An IQ Data Sampler
- A basic Signal Generator
- A GPIO Controller

Dependencies:
- uhd
- PyQt5
- pyqtgraph
- numpy

Installation:
1. Install UHD driver and the Python API. Follow the instructions from Ettus Research.
   Ensure you can find the device by running `uhd_find_devices`.
2. Install Python dependencies:
   pip install PyQt5 pyqtgraph numpy
"""

import sys
import numpy as np
import uhd
import threading
import time

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QLabel, QLineEdit, QPushButton, QComboBox, QFormLayout,
    QMessageBox, QCheckBox, QProgressBar
)
from PyQt5.QtCore import QTimer, pyqtSignal, QObject

import pyqtgraph as pg

# --- Constants ---
DEFAULT_SAMPLE_RATE = 2e6
DEFAULT_CENTER_FREQ = 100e6
DEFAULT_GAIN = 50

# --- Main Application Window ---
class B200miniControlPanel(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ettus B200mini-i Control Panel")
        self.setGeometry(100, 100, 1200, 800)

        self.usrp = None
        self.init_usrp()

        if not self.usrp:
            # Show a message box and exit if USRP is not found
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("USRP Device Not Found")
            msg.setInformativeText("Could not find a connected Ettus B200mini-i. Please ensure it's connected and the UHD drivers are installed correctly.")
            msg.setWindowTitle("Error")
            msg.exec_()
            sys.exit(1) # Exit the application

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # Create Tab Widget
        self.tabs = QTabWidget()
        self.layout.addWidget(self.tabs)

        # Add tabs
        self.spec_analyzer_tab = SpectrumAnalyzerTab(self.usrp)
        self.tabs.addTab(self.spec_analyzer_tab, "Spectrum Analyzer")

        self.iq_sampler_tab = IQSamplerTab(self.usrp)
        self.tabs.addTab(self.iq_sampler_tab, "IQ Sampler")

        self.signal_gen_tab = SignalGeneratorTab(self.usrp)
        self.tabs.addTab(self.signal_gen_tab, "Signal Generator")

        self.gpio_tab = GPIOControllerTab(self.usrp)
        self.tabs.addTab(self.gpio_tab, "GPIO Controller")
        
        self.current_tab_idx = 0
        self.tabs.currentChanged.connect(self.on_tab_changed)


    def init_usrp(self):
        """Initialize the USRP device."""
        try:
            # Find B200mini-i device. You can be more specific with args if needed.
            # e.g., uhd.usrp.MultiUSRP("type=b200")
            self.usrp = uhd.usrp.MultiUSRP()
            print("Found USRP device.")
        except RuntimeError as e:
            print(f"Error finding USRP device: {e}")
            self.usrp = None

    def on_tab_changed(self, index):
        """Handle tab changes to stop activity on the previous tab."""
        # Stop spectrum analyzer if switching away from it
        if self.current_tab_idx == 0 and self.spec_analyzer_tab.is_running:
            self.spec_analyzer_tab.stop_spectrum()
        
        # Stop signal generator if switching away
        if self.current_tab_idx == 2 and self.signal_gen_tab.is_transmitting:
            self.signal_gen_tab.stop_tx()
            
        self.current_tab_idx = index

    def closeEvent(self, event):
        """Ensure all threads and streams are stopped on exit."""
        print("Closing application...")
        self.spec_analyzer_tab.stop_spectrum()
        self.iq_sampler_tab.stop_sampling()
        self.signal_gen_tab.stop_tx()
        event.accept()

# --- Spectrum Analyzer Tab ---
class SpectrumAnalyzerTab(QWidget):
    def __init__(self, usrp):
        super().__init__()
        self.usrp = usrp
        self.is_running = False
        self.rx_streamer = None
        self.rx_thread = None

        # --- UI Elements ---
        self.layout = QVBoxLayout(self)

        # Plotting widget
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.getPlotItem().setLabel('bottom', 'Frequency (MHz)')
        self.plot_widget.getPlotItem().setLabel('left', 'Power (dBFS)')
        self.plot_widget.getPlotItem().showGrid(x=True, y=True)
        self.plot_curve = self.plot_widget.plot(pen='y')
        self.layout.addWidget(self.plot_widget)

        # Controls
        controls_layout = QFormLayout()
        self.freq_input = QLineEdit(str(DEFAULT_CENTER_FREQ / 1e6))
        self.rate_input = QLineEdit(str(DEFAULT_SAMPLE_RATE / 1e6))
        self.gain_input = QLineEdit(str(DEFAULT_GAIN))
        self.fft_size_input = QComboBox()
        self.fft_size_input.addItems(["1024", "2048", "4096", "8192"])
        self.fft_size_input.setCurrentText("2048")
        
        controls_layout.addRow("Center Frequency (MHz):", self.freq_input)
        controls_layout.addRow("Sample Rate (MS/s):", self.rate_input)
        controls_layout.addRow("Gain (dB):", self.gain_input)
        controls_layout.addRow("FFT Size:", self.fft_size_input)
        self.layout.addLayout(controls_layout)

        # Buttons
        button_layout = QHBoxLayout()
        self.start_button = QPushButton("Start Spectrum")
        self.start_button.clicked.connect(self.start_spectrum)
        self.stop_button = QPushButton("Stop Spectrum")
        self.stop_button.clicked.connect(self.stop_spectrum)
        self.stop_button.setEnabled(False)
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        self.layout.addLayout(button_layout)
        
        # Timer for plot updates
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_plot)
        self.latest_fft_data = None

    def start_spectrum(self):
        if self.is_running:
            return
        
        try:
            # Get settings from UI
            center_freq = float(self.freq_input.text()) * 1e6
            sample_rate = float(self.rate_input.text()) * 1e6
            gain = float(self.gain_input.text())
            self.fft_size = int(self.fft_size_input.currentText())

            # Configure USRP
            self.usrp.set_rx_rate(sample_rate, 0)
            self.usrp.set_rx_freq(uhd.types.TuneRequest(center_freq), 0)
            self.usrp.set_rx_gain(gain, 0)

            # Setup streamer
            st_args = uhd.usrp.StreamArgs("fc32", "sc16")
            st_args.channels = [0]
            self.rx_streamer = self.usrp.get_rx_stream(st_args)
            
            # Create recv buffer
            self.recv_buffer = np.zeros((1, self.fft_size), dtype=np.complex64)

            # Start thread for receiving samples
            self.is_running = True
            self.rx_thread = threading.Thread(target=self.rx_loop)
            self.rx_thread.daemon = True
            self.rx_thread.start()

            # Update UI
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.update_timer.start(50) # 20 FPS update rate

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start spectrum analyzer: {e}")
            self.is_running = False

    def stop_spectrum(self):
        if not self.is_running:
            return
            
        self.is_running = False
        self.update_timer.stop()
        if self.rx_thread:
            self.rx_thread.join(timeout=1) # Wait for thread to finish
        
        self.rx_streamer = None # This should stop the stream
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        print("Spectrum stopped.")

    def rx_loop(self):
        """Receiving loop that runs in a separate thread."""
        stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
        stream_cmd.stream_now = True
        self.rx_streamer.issue_stream_cmd(stream_cmd)
        
        metadata = uhd.types.RXMetadata()

        while self.is_running:
            try:
                num_samps = self.rx_streamer.recv(self.recv_buffer, metadata)
                if metadata.error_code != uhd.types.RXMetadataErrorCode.none:
                    print(f"Receiver error: {metadata.strerror()}")
                    continue
                
                if num_samps > 0:
                    # Process samples
                    samples = self.recv_buffer[0]
                    # Use a window function for better FFT results
                    win = np.hanning(len(samples))
                    fft_result = np.fft.fft(samples * win)
                    fft_result = np.fft.fftshift(fft_result)
                    # Convert to dBFS (decibels relative to full scale)
                    psd = 20 * np.log10(np.abs(fft_result) / self.fft_size)
                    self.latest_fft_data = psd

            except Exception as e:
                if self.is_running: # Don't print error if we are stopping
                    print(f"Error in RX loop: {e}")
                break

        # Stop stream
        stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
        self.rx_streamer.issue_stream_cmd(stream_cmd)
        print("RX loop finished.")

    def update_plot(self):
        """Updates the plot with the latest FFT data."""
        if self.latest_fft_data is not None:
            sample_rate = float(self.rate_input.text()) * 1e6
            center_freq = float(self.freq_input.text()) * 1e6
            
            # Generate frequency axis
            freq_axis = np.fft.fftshift(np.fft.fftfreq(self.fft_size, 1.0/sample_rate))
            freq_axis_mhz = (freq_axis + center_freq) / 1e6
            
            self.plot_curve.setData(x=freq_axis_mhz, y=self.latest_fft_data)
            self.plot_widget.setXRange(min(freq_axis_mhz), max(freq_axis_mhz))
            self.plot_widget.setYRange(-120, 0) # Reasonable dBFS range

# --- IQ Sampler Tab ---
class IQSamplerTab(QWidget):
    # Signal to update the progress bar from the worker thread
    progress_signal = pyqtSignal(int)

    def __init__(self, usrp):
        super().__init__()
        self.usrp = usrp
        self.is_sampling = False
        self.sampling_thread = None

        self.layout = QVBoxLayout(self)
        form_layout = QFormLayout()

        self.freq_input = QLineEdit(str(DEFAULT_CENTER_FREQ / 1e6))
        self.rate_input = QLineEdit(str(DEFAULT_SAMPLE_RATE / 1e6))
        self.gain_input = QLineEdit(str(DEFAULT_GAIN))
        self.num_samps_input = QLineEdit("1000000")
        self.filename_input = QLineEdit("iq_samples.dat")

        form_layout.addRow("Center Frequency (MHz):", self.freq_input)
        form_layout.addRow("Sample Rate (MS/s):", self.rate_input)
        form_layout.addRow("Gain (dB):", self.gain_input)
        form_layout.addRow("Number of Samples:", self.num_samps_input)
        form_layout.addRow("Filename:", self.filename_input)
        
        self.layout.addLayout(form_layout)

        self.start_button = QPushButton("Start Sampling")
        self.start_button.clicked.connect(self.start_sampling)
        self.layout.addWidget(self.start_button)
        
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 100)
        self.layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("Status: Idle")
        self.layout.addWidget(self.status_label)

        # Connect the signal
        self.progress_signal.connect(self.update_progress)

    def start_sampling(self):
        if self.is_sampling:
            return
        
        self.is_sampling = True
        self.start_button.setEnabled(False)
        self.status_label.setText("Status: Starting...")
        self.progress_bar.setValue(0)
        
        self.sampling_thread = threading.Thread(target=self.sampling_worker)
        self.sampling_thread.daemon = True
        self.sampling_thread.start()

    def stop_sampling(self):
        """Can be called to forcefully stop sampling."""
        self.is_sampling = False

    def sampling_worker(self):
        """Worker thread to handle the sampling process."""
        try:
            center_freq = float(self.freq_input.text()) * 1e6
            sample_rate = float(self.rate_input.text()) * 1e6
            gain = float(self.gain_input.text())
            num_samps = int(self.num_samps_input.text())
            filename = self.filename_input.text()

            # Configure USRP
            self.usrp.set_rx_rate(sample_rate, 0)
            self.usrp.set_rx_freq(uhd.types.TuneRequest(center_freq), 0)
            self.usrp.set_rx_gain(gain, 0)
            time.sleep(1) # Allow settings to settle

            # Setup streamer
            st_args = uhd.usrp.StreamArgs("fc32", "sc16")
            st_args.channels = [0]
            rx_streamer = self.usrp.get_rx_stream(st_args)

            # Allocate buffer and start stream
            samps = np.zeros(num_samps, dtype=np.complex64)
            recv_buffer = np.zeros(10000, dtype=np.complex64)
            
            stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.num_samps_and_done)
            stream_cmd.num_samps = num_samps
            stream_cmd.stream_now = True
            rx_streamer.issue_stream_cmd(stream_cmd)

            metadata = uhd.types.RXMetadata()
            total_samps_recd = 0
            
            self.status_label.setText("Status: Sampling...")

            while total_samps_recd < num_samps and self.is_sampling:
                samps_recd = rx_streamer.recv(recv_buffer, metadata)
                if metadata.error_code != uhd.types.RXMetadataErrorCode.none:
                    raise RuntimeError(metadata.strerror())
                
                if samps_recd > 0:
                    end_idx = total_samps_recd + samps_recd
                    samps[total_samps_recd:end_idx] = recv_buffer[:samps_recd]
                    total_samps_recd = end_idx
                    
                    # Update progress
                    progress = int((total_samps_recd / num_samps) * 100)
                    self.progress_signal.emit(progress)

            if not self.is_sampling: # If stopped prematurely
                self.status_label.setText("Status: Sampling stopped by user.")
            else:
                self.status_label.setText("Status: Saving to file...")
                samps.tofile(filename)
                self.status_label.setText(f"Status: Done. Saved {total_samps_recd} samples to {filename}")

        except Exception as e:
            self.status_label.setText(f"Status: Error - {e}")
        finally:
            self.is_sampling = False
            self.start_button.setEnabled(True)

    def update_progress(self, value):
        self.progress_bar.setValue(value)

# --- Signal Generator Tab ---
class SignalGeneratorTab(QWidget):
    def __init__(self, usrp):
        super().__init__()
        self.usrp = usrp
        self.is_transmitting = False
        self.tx_thread = None

        self.layout = QVBoxLayout(self)
        form_layout = QFormLayout()

        self.freq_input = QLineEdit(str(DEFAULT_CENTER_FREQ / 1e6))
        self.rate_input = QLineEdit(str(DEFAULT_SAMPLE_RATE / 1e6))
        self.gain_input = QLineEdit(str(DEFAULT_GAIN))
        self.amp_input = QLineEdit("0.5")
        self.wave_freq_input = QLineEdit("100000") # 100 kHz tone
        self.wave_type_combo = QComboBox()
        self.wave_type_combo.addItems(["Sine", "Square", "Constant"])

        form_layout.addRow("Center Frequency (MHz):", self.freq_input)
        form_layout.addRow("Sample Rate (MS/s):", self.rate_input)
        form_layout.addRow("Gain (dB):", self.gain_input)
        form_layout.addRow("Amplitude (0-1):", self.amp_input)
        form_layout.addRow("Waveform Freq (Hz):", self.wave_freq_input)
        form_layout.addRow("Waveform Type:", self.wave_type_combo)

        self.layout.addLayout(form_layout)

        button_layout = QHBoxLayout()
        self.start_button = QPushButton("Start Transmission")
        self.start_button.clicked.connect(self.start_tx)
        self.stop_button = QPushButton("Stop Transmission")
        self.stop_button.clicked.connect(self.stop_tx)
        self.stop_button.setEnabled(False)

        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        self.layout.addLayout(button_layout)
        
        self.status_label = QLabel("Status: Idle")
        self.layout.addWidget(self.status_label)

    def start_tx(self):
        if self.is_transmitting:
            return

        try:
            self.is_transmitting = True
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.status_label.setText("Status: Starting transmission...")

            self.tx_thread = threading.Thread(target=self.tx_worker)
            self.tx_thread.daemon = True
            self.tx_thread.start()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start transmission: {e}")
            self.stop_tx()

    def stop_tx(self):
        if not self.is_transmitting:
            return
            
        self.is_transmitting = False
        if self.tx_thread:
            self.tx_thread.join(timeout=1)
            
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.status_label.setText("Status: Idle")
        print("Transmission stopped.")

    def tx_worker(self):
        try:
            center_freq = float(self.freq_input.text()) * 1e6
            sample_rate = float(self.rate_input.text()) * 1e6
            gain = float(self.gain_input.text())
            amplitude = float(self.amp_input.text())
            wave_freq = float(self.wave_freq_input.text())
            wave_type = self.wave_type_combo.currentText()

            # Configure USRP
            self.usrp.set_tx_rate(sample_rate)
            self.usrp.set_tx_freq(uhd.types.TuneRequest(center_freq))
            self.usrp.set_tx_gain(gain)

            # Setup streamer
            st_args = uhd.usrp.StreamArgs("fc32", "sc16")
            st_args.channels = [0]
            tx_streamer = self.usrp.get_tx_stream(st_args)

            # Generate waveform
            num_samps_in_buff = 10000
            t = np.arange(num_samps_in_buff) / sample_rate
            
            if wave_type == "Sine":
                waveform = amplitude * np.exp(1j * 2 * np.pi * wave_freq * t).astype(np.complex64)
            elif wave_type == "Square":
                waveform = amplitude * (np.sign(np.sin(2 * np.pi * wave_freq * t))).astype(np.complex64)
                waveform.imag = 0 # Make it a real signal for simplicity
            else: # Constant
                waveform = amplitude * np.ones(num_samps_in_buff, dtype=np.complex64)
            
            self.status_label.setText(f"Status: Transmitting {wave_type} wave...")
            
            metadata = uhd.types.TXMetadata()
            metadata.start_of_burst = True
            metadata.end_of_burst = False
            metadata.has_time_spec = False

            # Stream continuously
            while self.is_transmitting:
                tx_streamer.send(waveform, metadata)
                metadata.start_of_burst = False
                
        except Exception as e:
            self.status_label.setText(f"Status: Error - {e}")
        finally:
            # Send a zero buffer to clear the transmitter
            tx_streamer.send(np.zeros(1000, dtype=np.complex64), uhd.types.TXMetadata(end_of_burst=True))
            self.is_transmitting = False


# --- GPIO Controller Tab ---
class GPIOControllerTab(QWidget):
    def __init__(self, usrp):
        super().__init__()
        self.usrp = usrp
        # B200mini has GPIOs on bank 0
        self.gpio_bank = "FP0" 
        
        self.layout = QVBoxLayout(self)
        form_layout = QFormLayout()

        # Get number of GPIOs
        try:
            # We need to set the bank direction before we can get source/sink info
            # Set all to output initially
            self.usrp.set_gpio_attr(self.gpio_bank, "DDR", 0xFFFF, 0xFFFF)
            self.num_gpios = len(self.usrp.get_gpio_sources(self.gpio_bank))
        except Exception as e:
            self.num_gpios = 0
            QMessageBox.warning(self, "GPIO Warning", f"Could not initialize GPIOs: {e}")

        self.gpio_checkboxes = []
        if self.num_gpios > 0:
            for i in range(self.num_gpios):
                checkbox = QCheckBox(f"GPIO {i}")
                checkbox.stateChanged.connect(lambda state, pin=i: self.set_gpio_state(pin, state))
                form_layout.addRow(checkbox)
                self.gpio_checkboxes.append(checkbox)
        else:
            self.layout.addWidget(QLabel("No GPIOs available or error during initialization."))

        self.layout.addLayout(form_layout)
        
        # Button to read current state
        self.read_button = QPushButton("Read GPIO States")
        self.read_button.clicked.connect(self.read_gpio_states)
        self.layout.addWidget(self.read_button)
        
        self.read_gpio_states() # Initial read

    def set_gpio_state(self, pin, state):
        """Set a single GPIO pin to high or low."""
        try:
            # Mask for the specific pin
            pin_mask = 1 << pin
            # Set direction to output for this pin
            self.usrp.set_gpio_attr(self.gpio_bank, "DDR", pin_mask, pin_mask)
            
            # Set the state
            if state: # Checked (High)
                self.usrp.set_gpio_attr(self.gpio_bank, "OUT", pin_mask, pin_mask)
                print(f"Set GPIO {pin} to HIGH")
            else: # Unchecked (Low)
                self.usrp.set_gpio_attr(self.gpio_bank, "OUT", 0x0000, pin_mask)
                print(f"Set GPIO {pin} to LOW")

        except Exception as e:
            QMessageBox.critical(self, "GPIO Error", f"Failed to set GPIO {pin} state: {e}")

    def read_gpio_states(self):
        """Read and update the checkboxes with the current GPIO states."""
        if not self.num_gpios > 0:
            return
            
        try:
            # Set all pins to input to read their state
            self.usrp.set_gpio_attr(self.gpio_bank, "DDR", 0x0000, 0xFFFF)
            
            # Read the entire bank
            current_state = self.usrp.get_gpio_attr(self.gpio_bank, "IN")
            
            for i, checkbox in enumerate(self.gpio_checkboxes):
                pin_mask = 1 << i
                is_high = (current_state & pin_mask) != 0
                # Block signals to prevent stateChanged from firing
                checkbox.blockSignals(True)
                checkbox.setChecked(is_high)
                checkbox.blockSignals(False)
            
            print(f"Read GPIO state: {bin(current_state)}")

        except Exception as e:
            QMessageBox.critical(self, "GPIO Error", f"Failed to read GPIO states: {e}")

# --- Main execution ---
if __name__ == "__main__":
    app = QApplication(sys.argv)
    # Use a modern style if available
    if 'Fusion' in pg.Qt.QtWidgets.QStyleFactory.keys():
        app.setStyle('Fusion')
    main_win = B200miniControlPanel()
    main_win.show()
    sys.exit(app.exec_())
