import sys  
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout,   
                             QHBoxLayout, QTextEdit, QComboBox, QWidget, QLabel, QCheckBox)  
from PyQt5.QtCore import Qt, QThread, pyqtSignal  
from speech_recognizer import SpeechRecognizer  

class TextUpdateThread(QThread):  
    text_update = pyqtSignal(str)  
    
    def __init__(self, speech_recognizer):  
        super().__init__()  
        self.speech_recognizer = speech_recognizer  
        self.running = True  
    
    def run(self):  
        while self.running:  
            try:  
                text = self.speech_recognizer.text_queue.get(timeout=0.1)  
                self.text_update.emit(text)  
            except:  
                continue  
    
    def stop(self):  
        self.running = False  

import sys  
import numpy as np  
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout,   
                             QHBoxLayout, QTextEdit, QComboBox, QWidget, QLabel,   
                             QProgressBar)  
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer  
from speech_recognizer import SpeechRecognizer  

class VolumeThread(QThread):  
    volume_update = pyqtSignal(float)  
    
    def __init__(self, speech_recognizer):  
        super().__init__()  
        self.speech_recognizer = speech_recognizer  
        self.running = True  
    
    def run(self):  
        while self.running:  
            try:  
                # Get latest audio data for volume calculation  
                if not self.speech_recognizer.audio_queue.empty():  
                    audio_data = self.speech_recognizer.audio_queue.get()  
                    volume = self.speech_recognizer.calculate_volume(audio_data)  
                    self.volume_update.emit(volume * 100)  # Scale for progress bar  
            except Exception as e:  
                print(f"Volume thread error: {e}")  
                
    def stop(self):  
        self.running = False  

class SpeechToTextApp(QMainWindow):  
    def __init__(self):  
        super().__init__()  
        self.setWindowTitle('Advanced Speech-to-Text Application')  
        self.setGeometry(100, 100, 700, 500)  
        
        # Initialize speech recognizer  
        self.speech_recognizer = SpeechRecognizer()  
        
        # Create central widget and layout  
        central_widget = QWidget()  
        main_layout = QVBoxLayout()  
        
        # Language model selection  
        model_layout = QHBoxLayout()  
        self.model_combo = QComboBox()  
        self.model_combo.addItems(self.speech_recognizer.get_available_models())  
        model_layout.addWidget(QLabel('Select Language Model:'))  
        model_layout.addWidget(self.model_combo)  
        
        # Device selection  
        device_layout = QHBoxLayout()  
        self.device_combo = QComboBox()  
        self.device_combo.addItems(self.speech_recognizer.get_audio_devices())  
        device_layout.addWidget(QLabel('Select Audio Device:'))  
        device_layout.addWidget(self.device_combo)  
        
        # Volume Indicators  
        volume_layout = QHBoxLayout()  
        self.mic_volume_label = QLabel('Mic Volume:')  
        self.mic_volume_bar = QProgressBar()  
        self.mic_volume_bar.setMaximum(100)  
        volume_layout.addWidget(self.mic_volume_label)  
        volume_layout.addWidget(self.mic_volume_bar)  
        
        # Text display area  
        self.text_display = QTextEdit()  
        self.text_display.setReadOnly(True)  
        
        # Control buttons  
        button_layout = QHBoxLayout()  
        self.start_button = QPushButton('Start Recording')  
        self.stop_button = QPushButton('Stop Recording')  
        self.stop_button.setEnabled(False)  
        
        button_layout.addWidget(self.start_button)  
        button_layout.addWidget(self.stop_button)  
        
        # Add layouts to main layout  
        main_layout.addLayout(model_layout)  
        main_layout.addLayout(device_layout)  
        main_layout.addLayout(volume_layout)  
        main_layout.addWidget(self.text_display)  
        main_layout.addLayout(button_layout)  
        
        central_widget.setLayout(main_layout)  
        self.setCentralWidget(central_widget)  
        
        # Connect signals and slots  
        self.start_button.clicked.connect(self.start_recording)  
        self.stop_button.clicked.connect(self.stop_recording)  
        
        # Threads  
        self.text_update_thread = None  
        self.volume_thread = None  
        # Add continuous transcription checkbox  
        self.continuous_mode_checkbox = QCheckBox('Continuous Transcription')  
        button_layout.addWidget(self.continuous_mode_checkbox)      
        
    def start_recording(self):  
        try:  
            device_index = self.device_combo.currentIndex()  
            model_name = self.model_combo.currentText()  
            
            # Check if continuous mode is enabled  
            if self.continuous_mode_checkbox.isChecked():  
                self.speech_recognizer.start_continuous_transcription(  
                    device_index,   
                    model_name,  
                    silence_threshold=0.01,  # Adjustable   
                    silence_duration=1.0     # Adjustable  
                )  
            else:  
                # Regular recording mode  
                self.speech_recognizer.start_recording(  
                    device_index,   
                    model_name  
                )  
            
            # Start text update thread (same as before)  
            self.text_update_thread = TextUpdateThread(self.speech_recognizer)  
            self.text_update_thread.text_update.connect(self.update_text_display)  
            self.text_update_thread.start()  
            
            # Start volume thread (same as before)  
            self.volume_thread = VolumeThread(self.speech_recognizer)  
            self.volume_thread.volume_update.connect(self.update_volume)  
            self.volume_thread.start()  
            
            # Update button states  
            self.start_button.setEnabled(False)  
            self.stop_button.setEnabled(True)  
        
        except Exception as e:  
            self.show_error(f"Recording start error: {e}")  
    
    def stop_recording(self):  
        try:  
            # Check if continuous mode was active  
            if self.continuous_mode_checkbox.isChecked():  
                self.speech_recognizer.stop_continuous_transcription()  
            else:  
                self.speech_recognizer.stop_recording()  
            
            # Stop threads (same as before)  
            if self.text_update_thread:  
                self.text_update_thread.stop()  
                self.text_update_thread.wait()  
            
            if self.volume_thread:  
                self.volume_thread.stop()  
                self.volume_thread.wait()  
            
            # Reset UI  
            self.start_button.setEnabled(True)  
            self.stop_button.setEnabled(False)  
            self.mic_volume_bar.setValue(0)  
        
        except Exception as e:  
            self.show_error(f"Recording stop error: {e}")  
    
    def update_text_display(self, text):  
        self.text_display.append(text)  
        # Scroll to bottom  
        scrollbar = self.text_display.verticalScrollBar()  
        scrollbar.setValue(scrollbar.maximum())  
    
    def update_volume(self, volume):  
        self.mic_volume_bar.setValue(int(volume))  
    
    def show_error(self, message):  
        from PyQt5.QtWidgets import QMessageBox  
        error_box = QMessageBox()  
        error_box.setIcon(QMessageBox.Critical)  
        error_box.setText("Error")  
        error_box.setInformativeText(message)  
        error_box.setWindowTitle("Speech-to-Text Error")  
        error_box.exec_()  

def main():  
    app = QApplication(sys.argv)  
    speech_to_text_app = SpeechToTextApp()  
    speech_to_text_app.show()  
    sys.exit(app.exec_())  

if __name__ == '__main__':  
    main()