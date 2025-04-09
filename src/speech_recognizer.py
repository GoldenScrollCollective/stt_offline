import json  
import vosk  
import sounddevice as sd  
import numpy as np  
import queue  
import threading  

class SpeechRecognizer:  
    def __init__(self, config_path='./config/config.json'):  
        # Load configuration  
        with open(config_path, 'r') as config_file:  
            self.config = json.load(config_file)  
        
        # Initialize Vosk model  
        self.model = vosk.Model(self.config['model_path'])  
        self.recognizer = vosk.KaldiRecognizer(self.model, self.config['sample_rate'])  
        
        # Audio capture setup  
        self.is_recording = False  
        self.audio_queue = queue.Queue()  
        self.text_queue = queue.Queue()  
        
        # Get available audio devices  
        self.devices = sd.query_devices()  
        self.device_names = [device['name'] for device in self.devices]  
    
    def get_audio_devices(self):  
        return self.device_names  
    
    def start_recording(self, device_index=None):  
        self.is_recording = True  
        self.audio_queue = queue.Queue()  
        self.text_queue = queue.Queue()  
        
        # Use specified device or default  
        device_index = device_index if device_index is not None else self.config['device_index']  
        
        def audio_callback(indata, frames, time, status):  
            if status:  
                print(status)  
            self.audio_queue.put(indata.copy())  
        
        def recognition_thread():  
            while self.is_recording:  
                try:  
                    data = self.audio_queue.get(timeout=0.1)  
                    if self.recognizer.AcceptWaveform(data.tobytes()):  
                        result = json.loads(self.recognizer.Result())  
                        if 'text' in result and result['text'].strip():  
                            self.text_queue.put(result['text'])  
                except queue.Empty:  
                    continue  
        
        # Start audio stream  
        self.stream = sd.InputStream(  
            samplerate=self.config['sample_rate'],   
            channels=self.config['channels'],  
            dtype='float32',  
            callback=audio_callback,  
            device=device_index  
        )  
        self.stream.start()  
        
        # Start recognition thread  
        self.recognition_thread = threading.Thread(target=recognition_thread)  
        self.recognition_thread.start()  
    
    def stop_recording(self):  
        self.is_recording = False  
        if hasattr(self, 'stream'):  
            self.stream.stop()  
            self.stream.close()  
        
        if hasattr(self, 'recognition_thread'):  
            self.recognition_thread.join()  
        
        # Get final result  
        final_result = self.recognizer.FinalResult()  
        return final_result