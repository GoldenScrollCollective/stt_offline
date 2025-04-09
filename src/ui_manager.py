import sys  
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout,   
                             QHBoxLayout, QTextEdit, QComboBox, QWidget, QLabel)  
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

class SpeechToTextApp(QMainWindow):  
    def __init__(self):  
        super().__init__()  
        self.setWindowTitle('Speech-to-Text Application')  
        self.setGeometry(100, 100, 600, 400)  
        
        # Initialize speech recognizer  
        self.speech_recognizer = SpeechRecognizer()  
        
        # Create central widget and layout  
        central_widget = QWidget()  
        main_layout = QVBoxLayout()  
        
        # Device selection  
        device_layout = QHBoxLayout()  
        self.device_combo = QComboBox()  
        self.device_combo.addItems(self.speech_recognizer.get_audio_devices())  
        device_layout.addWidget(QLabel('Select Audio Device:'))  
        device_layout.addWidget(self.device_combo)  
        
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
        main_layout.addLayout(device_layout)  
        main_layout.addWidget(self.text_display)  
        main_layout.addLayout(button_layout)  
        
        central_widget.setLayout(main_layout)  
        self.setCentralWidget(central_widget)  
        
        # Connect signals and slots  
        self.start_button.clicked.connect(self.start_recording)  
        self.stop_button.clicked.connect(self.stop_recording)  
        
        # Text update thread  
        self.text_update_thread = None  
    
    def start_recording(self):  
        device_index = self.device_combo.currentIndex()  
        self.speech_recognizer.start_recording(device_index)  
        
        # Start text update thread  
        self.text_update_thread = TextUpdateThread(self.speech_recognizer)  
        self.text_update_thread.text_update.connect(self.update_text_display)  
        self.text_update_thread.start()  
        
        # Update button states  
        self.start_button.setEnabled(False)  
        self.stop_button.setEnabled(True)  
    
    def stop_recording(self):  
        self.speech_recognizer.stop_recording()  
        
        # Stop text update thread  
        if self.text_update_thread:  
            self.text_update_thread.stop()  
            self.text_update_thread.wait()  
        
        # Update button states  
        self.start_button.setEnabled(True)  
        self.stop_button.setEnabled(False)  
    
    def update_text_display(self, text):  
        self.text_display.append(text)  
        # Scroll to bottom  
        self.text_display.verticalScrollBar().setValue(  
            self.text_display.verticalScrollBar().maximum()  
        )  

def main():  
    app = QApplication(sys.argv)  
    speech_to_text_app = SpeechToTextApp()  
    speech_to_text_app.show()  
    sys.exit(app.exec_())  

if __name__ == '__main__':  
    main()