import sys
from pathlib import Path
import sounddevice as sd
import numpy as np
from datetime import datetime
import queue
import threading

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.hybrid_stt import HybridSTT

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QTextEdit, QLabel, QFrame, QStatusBar, QComboBox, QSpinBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QPalette, QColor


class AudioThread(QThread):
    """Thread for handling audio recording and processing"""
    transcription_ready = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, stt_engine):
        super().__init__()
        self.stt = stt_engine
        self.is_recording = False
        self.audio_queue = queue.Queue()
        self.audio_buffer = []
        self.sample_rate = 16000
        self.chunk_ = 15
        self.running = True
        
    def audio_callback(self, indata, frames, time, status):
        """Audio input callback"""
        if self.is_recording:
            self.audio_queue.put(indata.copy())
    
    def run(self):
        """Main audio processing loop"""
        with sd.InputStream(
            callback=self.audio_callback,
            channels=1,
            samplerate=self.sample_rate,
            blocksize=int(self.sample_rate * 0.1)
        ):
            while self.running:
                try:
                    chunk = self.audio_queue.get(timeout=0.1)
                    self.audio_buffer.append(chunk)
                    
                    buffer_samples = sum(len(c) for c in self.audio_buffer)
                    buffer_duration = buffer_samples / self.sample_rate
                    
                    if buffer_duration >= self.chunk_duration:
                        audio_data = np.concatenate(self.audio_buffer).flatten()
                        
                        # Transcribe
                        result = self.stt.transcribe(
                            audio_array=audio_data,
                            sample_rate=self.sample_rate
                        )
                        
                        self.transcription_ready.emit(result)
                        self.audio_buffer = []
                        
                except queue.Empty:
                    continue
                except Exception as e:
                    self.error_occurred.emit(str(e))
    
    def start_recording(self):
        self.is_recording = True
        
    def pause_recording(self):
        self.is_recording = False
        self.audio_buffer = []
    
    def stop(self):
        self.running = False
        self.is_recording = False


class HybridSTTGUI(QMainWindow):
    """Professional GUI for Hybrid STT"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hybrid STT - Real-time Transcription")
        self.setGeometry(100, 100, 900, 700)
        
        # Initialize STT
        self.init_stt()
        
        # Setup UI
        self.init_ui()
        
        # Start audio thread
        self.audio_thread = AudioThread(self.stt)
        self.audio_thread.transcription_ready.connect(self.on_transcription_ready)
        self.audio_thread.error_occurred.connect(self.on_error)
        self.audio_thread.start()
        
        # Status timer
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.update_status)
        self.status_timer.start(100)
        
    def init_stt(self):
        """Initialize STT engine"""
        self.status_label_init = QLabel("Initializing STT engines...")
        self.status_label_init.setAlignment(Qt.AlignCenter)
        self.status_label_init.setStyleSheet("font-size: 14px; padding: 20px;")
        
        self.setCentralWidget(self.status_label_init)
        QApplication.processEvents()
        
        self.stt = HybridSTT()
        
    def init_ui(self):
        """Initialize the user interface"""
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout()
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(15, 15, 15, 15)
        
        # Header
        header = self.create_header()
        main_layout.addWidget(header)
        
        # Control panel
        controls = self.create_controls()
        main_layout.addWidget(controls)
        
        # Transcription display
        transcription = self.create_transcription_display()
        main_layout.addLayout(transcription)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        
        central_widget.setLayout(main_layout)
        
        # Apply styling
        self.apply_styles()
        
    def create_header(self):
        """Create header section"""
        header = QFrame()
        header.setFrameShape(QFrame.StyledPanel)
        header.setStyleSheet("""
            QFrame {
                background-color: #2c3e50;
                border-radius: 5px;
                padding: 15px;
            }
        """)
        
        layout = QVBoxLayout()
        
        title = QLabel("Hybrid Speech-to-Text System")
        title.setStyleSheet("color: white; font-size: 24px; font-weight: bold;")
        title.setAlignment(Qt.AlignCenter)
        
        subtitle = QLabel("Whisper (EN, AR) • IndicSTT (ML) • Auto Language Detection")
        subtitle.setStyleSheet("color: #ecf0f1; font-size: 12px;")
        subtitle.setAlignment(Qt.AlignCenter)
        
        layout.addWidget(title)
        layout.addWidget(subtitle)
        
        header.setLayout(layout)
        return header
    
    def create_controls(self):
        """Create control panel"""
        controls = QFrame()
        controls.setFrameShape(QFrame.StyledPanel)
        
        layout = QHBoxLayout()
        
        # Record button
        self.record_btn = QPushButton("Start Recording")
        self.record_btn.setCheckable(True)
        self.record_btn.setMinimumHeight(50)
        self.record_btn.clicked.connect(self.toggle_recording)
        
        # Pause button
        self.pause_btn = QPushButton("Pause")
        self.pause_btn.setMinimumHeight(50)
        self.pause_btn.setEnabled(False)
        self.pause_btn.clicked.connect(self.pause_recording)
        
        # Clear button
        clear_btn = QPushButton("Clear")
        clear_btn.setMinimumHeight(50)
        clear_btn.clicked.connect(self.clear_transcription)
        
        # Recording duration
        duration_label = QLabel("Duration (s):")
        self.duration_spin = QSpinBox()
        self.duration_spin.setMinimum(1)
        self.duration_spin.setMaximum(10)
        self.duration_spin.setValue(3)
        self.duration_spin.valueChanged.connect(self.on_duration_changed)
        
        layout.addWidget(self.record_btn, 2)
        layout.addWidget(self.pause_btn, 1)
        layout.addWidget(clear_btn, 1)
        layout.addWidget(duration_label)
        layout.addWidget(self.duration_spin)
        
        controls.setLayout(layout)
        return controls
    
    def create_transcription_display(self):
        """Create transcription display area"""
        layout = QVBoxLayout()
        
        # Current engine indicator
        engine_layout = QHBoxLayout()
        
        engine_label = QLabel("Current Engine:")
        self.engine_indicator = QLabel("Whisper")
        self.engine_indicator.setStyleSheet("""
            background-color: #3498db;
            color: white;
            padding: 5px 15px;
            border-radius: 3px;
            font-weight: bold;
        """)
        
        self.lang_indicator = QLabel("Language: --")
        self.lang_indicator.setStyleSheet("""
            background-color: #95a5a6;
            color: white;
            padding: 5px 15px;
            border-radius: 3px;
        """)
        
        engine_layout.addWidget(engine_label)
        engine_layout.addWidget(self.engine_indicator)
        engine_layout.addWidget(self.lang_indicator)
        engine_layout.addStretch()
        
        layout.addLayout(engine_layout)
        
        # Transcription text area
        text_label = QLabel("Transcription:")
        text_label.setStyleSheet("font-weight: bold; font-size: 14px; margin-top: 10px;")
        
        self.transcription_text = QTextEdit()
        self.transcription_text.setReadOnly(True)
        self.transcription_text.setMinimumHeight(350)
        self.transcription_text.setStyleSheet("""
            QTextEdit {
                background-color: #ecf0f1;
                border: 2px solid #bdc3c7;
                border-radius: 5px;
                padding: 10px;
                font-size: 13px;
                font-family: 'Courier New', monospace;
            }
        """)
        
        layout.addWidget(text_label)
        layout.addWidget(self.transcription_text)
        
        return layout
    
    def apply_styles(self):
        """Apply global styles"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #ecf0f1;
            }
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 10px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #21618c;
            }
            QPushButton:checked {
                background-color: #e74c3c;
            }
            QPushButton:checked:hover {
                background-color: #c0392b;
            }
            QPushButton:disabled {
                background-color: #95a5a6;
            }
            QFrame {
                background-color: white;
                border-radius: 5px;
                padding: 10px;
            }
            QSpinBox {
                padding: 5px;
                font-size: 13px;
                border: 2px solid #bdc3c7;
                border-radius: 3px;
            }
        """)
    
    def toggle_recording(self):
        """Toggle recording on/off"""
        if self.record_btn.isChecked():
            self.audio_thread.start_recording()
            self.record_btn.setText("Stop Recording")
            self.pause_btn.setEnabled(True)
            self.status_bar.showMessage("Recording...")
        else:
            self.audio_thread.pause_recording()
            self.record_btn.setText("Start Recording")
            self.pause_btn.setEnabled(False)
            self.status_bar.showMessage("Stopped")
    
    def pause_recording(self):
        """Pause recording"""
        self.audio_thread.pause_recording()
        self.record_btn.setChecked(False)
        self.record_btn.setText("Start Recording")
        self.pause_btn.setEnabled(False)
        self.status_bar.showMessage("Paused")
    
    def clear_transcription(self):
        """Clear transcription display"""
        self.transcription_text.clear()
        self.status_bar.showMessage("Cleared")
    
    def on_duration_changed(self, value):
        """Update recording duration"""
        self.audio_thread.chunk_duration = value
    
    def on_transcription_ready(self, result):
        """Handle new transcription result"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        engine = result['engine']
        language = result['language'].upper()
        text = result['text']
        
        # Update engine indicator
        if engine == 'indic':
            self.engine_indicator.setText("IndicSTT")
            self.engine_indicator.setStyleSheet("""
                background-color: #27ae60;
                color: white;
                padding: 5px 15px;
                border-radius: 3px;
                font-weight: bold;
            """)
        else:
            self.engine_indicator.setText("Whisper")
            self.engine_indicator.setStyleSheet("""
                background-color: #3498db;
                color: white;
                padding: 5px 15px;
                border-radius: 3px;
                font-weight: bold;
            """)
        
        # Update language indicator
        self.lang_indicator.setText(f"Language: {language}")
        
        # Append to transcription
        entry = f"[{timestamp}] [{engine.upper()}] {text}\n"
        self.transcription_text.append(entry)
        
        # Scroll to bottom
        self.transcription_text.verticalScrollBar().setValue(
            self.transcription_text.verticalScrollBar().maximum()
        )
    
    def on_error(self, error_msg):
        """Handle errors"""
        self.status_bar.showMessage(f"Error: {error_msg}")
    
    def update_status(self):
        """Update status indicators"""
        if self.audio_thread.is_recording:
            current = self.status_bar.currentMessage()
            if "Recording" in current:
                dots = current.count('.')
                self.status_bar.showMessage("Recording" + "." * ((dots % 3) + 1))
    
    def closeEvent(self, event):
        """Handle window close"""
        self.audio_thread.stop()
        self.audio_thread.wait()
        event.accept()


def main():
    app = QApplication(sys.argv)
    
    # Set application font
    font = QFont("Segoe UI", 10)
    app.setFont(font)
    
    window = HybridSTTGUI()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

