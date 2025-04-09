import json  
import os  
import vosk  
import sounddevice as sd  
import soundfile as sf  
import numpy as np  
import queue  
import threading  
import logging  
from datetime import datetime  

class SpeechRecognizer:  
    def __init__(self, config_path='./config/config.json'):  
        # Logging setup  
        logging.basicConfig(  
            level=logging.INFO,   
            format='%(asctime)s - %(levelname)s: %(message)s',  
            filename='speech_recognition.log'  
        )  
        self.logger = logging.getLogger(__name__)  

        # Load configuration  
        try:  
            with open(config_path, 'r') as config_file:  
                self.config = json.load(config_file)  
        except Exception as e:  
            self.logger.error(f"Config file error: {e}")  
            self.config = {  
                "model_path": "./models/vosk-model-small-en-us-0.15",  
                "sample_rate": 16000,  
                "channels": 1,  
                "device_index": 0,  
                "chunk_size": 2048  
            }  

        # Available language models  
        self.available_models = {  
            "English (US)": "./models/vosk-model-small-en-us-0.15",  
            "Russian": "./models/vosk-model-small-ru-0.22",  
            "Spanish": "./models/vosk-model-small-es-0.42"  
        }  

        # Initialize Vosk model  
        self.set_language_model(self.config['model_path'])  
        
        # Audio capture setup  
        self.is_recording = False  
        self.audio_queue = queue.Queue()  
        self.text_queue = queue.Queue()  
        self.recorded_audio = []  
        
        # Continuous transcription mode  
        self.continuous_mode = False  
        self.silence_threshold = 0.01  # Adjust based on your environment  
        self.silence_duration = 1.0  # Seconds of silence to trigger pause  
        self.last_speech_time = None  
        
        # Get available audio devices  
        self.devices = sd.query_devices()  
        self.device_names = [device['name'] for device in self.devices]  

    def set_language_model(self, model_path):  
        try:  
            self.model = vosk.Model(model_path)  
            self.recognizer = vosk.KaldiRecognizer(self.model, self.config['sample_rate'])  
            self.logger.info(f"Loaded model from {model_path}")  
        except Exception as e:  
            self.logger.error(f"Model loading error: {e}")  
            raise  

    def get_available_models(self):  
        return list(self.available_models.keys())  

    def get_audio_devices(self):  
        return self.device_names  
    
    def calculate_volume(self, audio_data):  
        # Calculate RMS volume  
        return np.sqrt(np.mean(audio_data**2))  

    def start_recording(self, device_index=None, model_name=None):  
        # Reset recording state  
        self.is_recording = True  
        self.audio_queue = queue.Queue()  
        self.text_queue = queue.Queue()  
        self.recorded_audio = []  

        # Set language model if specified  
        if model_name:  
            model_path = self.available_models.get(model_name)  
            if model_path:  
                self.set_language_model(model_path)  

        # Use specified device or default  
        device_index = device_index if device_index is not None else self.config['device_index']  
        
        def audio_callback(indata, frames, time, status):  
            if status:  
                self.logger.warning(status)  
            
            # Store audio for recording and volume calculation  
            self.audio_queue.put(indata.copy())  
            self.recorded_audio.append(indata.copy())  
        
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
                except Exception as e:  
                    self.logger.error(f"Recognition error: {e}")  
        
        try:  
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
            
            self.logger.info("Recording started")  
        except Exception as e:  
            self.logger.error(f"Recording start error: {e}")  
            self.is_recording = False  
    
    def stop_recording(self):  
        self.is_recording = False  
        
        # Stop stream  
        if hasattr(self, 'stream'):  
            self.stream.stop()  
            self.stream.close()  
        
        # Wait for recognition thread  
        if hasattr(self, 'recognition_thread'):  
            self.recognition_thread.join()  
        
        # Export recording  
        self.export_recording()  
        
        # Get final result  
        final_result = self.recognizer.FinalResult()  
        self.logger.info("Recording stopped")  
        return final_result  

    def export_recording(self):  
        # Ensure recordings directory exists  
        os.makedirs('recordings', exist_ok=True)  
        
        # Generate unique filename  
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  
        filename = f"recordings/recording_{timestamp}.wav"  
        
        try:  
            # Combine recorded audio  
            if self.recorded_audio:  
                recording = np.concatenate(self.recorded_audio, axis=0)  
                
                # Export to WAV  
                sf.write(filename, recording, self.config['sample_rate'])  
                self.logger.info(f"Recording exported: {filename}")  
                return filename  
        except Exception as e:  
            self.logger.error(f"Recording export error: {e}")  
        
        return None
    
    def start_continuous_transcription(self, device_index=None, model_name=None,   
                                       silence_threshold=0.01,   
                                       silence_duration=1.0):  
        """  
        Start continuous transcription mode  
        - Automatically manages recording based on speech activity  
        - Stops recording during prolonged silence  
        """  
        # Reset state  
        self.continuous_mode = True  
        self.silence_threshold = silence_threshold  
        self.silence_duration = silence_duration  
        self.audio_queue = queue.Queue()  
        self.text_queue = queue.Queue()  
        self.recorded_audio = []  
        self.last_speech_time = datetime.now()  

        # Set language model if specified  
        if model_name:  
            model_path = self.available_models.get(model_name)  
            if model_path:  
                self.set_language_model(model_path)  

        # Use specified device or default  
        device_index = device_index if device_index is not None else self.config['device_index']  
        
        def audio_callback(indata, frames, time, status):  
            if status:  
                self.logger.warning(status)  
            
            # Calculate volume  
            volume = self.calculate_volume(indata)  
            
            # Check for speech activity  
            if volume > self.silence_threshold:  
                self.last_speech_time = datetime.now()  
                self.audio_queue.put(indata.copy())  
                self.recorded_audio.append(indata.copy())  
        
        def continuous_recognition_thread():  
            while self.continuous_mode:  
                try:  
                    # Check for silence duration  
                    current_time = datetime.now()  
                    silence_elapsed = (current_time - self.last_speech_time).total_seconds()  
                    
                    if silence_elapsed > self.silence_duration and self.recorded_audio:  
                        # Process the recorded segment  
                        self.process_audio_segment()  
                        
                        # Reset for next segment  
                        self.recorded_audio = []  
                    
                    # Process available audio  
                    if not self.audio_queue.empty():  
                        data = self.audio_queue.get()  
                        if self.recognizer.AcceptWaveform(data.tobytes()):  
                            result = json.loads(self.recognizer.Result())  
                            if 'text' in result and result['text'].strip():  
                                self.text_queue.put(result['text'])  
                    
                    # Small sleep to prevent CPU overuse  
                    threading.Event().wait(0.1)  
                
                except Exception as e:  
                    self.logger.error(f"Continuous recognition error: {e}")  
        
        try:  
            # Start audio stream  
            self.stream = sd.InputStream(  
                samplerate=self.config['sample_rate'],   
                channels=self.config['channels'],  
                dtype='float32',  
                callback=audio_callback,  
                device=device_index  
            )  
            self.stream.start()  
            
            # Start continuous recognition thread  
            self.recognition_thread = threading.Thread(target=continuous_recognition_thread)  
            self.recognition_thread.start()  
            
            self.logger.info("Continuous transcription started")  
        
        except Exception as e:  
            self.logger.error(f"Continuous transcription start error: {e}")  
            self.continuous_mode = False  

    def process_audio_segment(self):  
        """  
        Process a segment of recorded audio  
        - Export the segment  
        - Perform final recognition  
        """  
        try:  
            if self.recorded_audio:  
                # Combine recorded audio  
                recording = np.concatenate(self.recorded_audio, axis=0)  
                
                # Export segment  
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  
                segment_filename = f"recordings/segment_{timestamp}.wav"  
                os.makedirs('recordings', exist_ok=True)  
                sf.write(segment_filename, recording, self.config['sample_rate'])  
                
                # Perform final recognition on the segment  
                final_result = self.recognizer.FinalResult()  
                if final_result:  
                    self.logger.info(f"Segment processed: {final_result}")  
        
        except Exception as e:  
            self.logger.error(f"Audio segment processing error: {e}")  

    def stop_continuous_transcription(self):  
        """  
        Stop continuous transcription  
        - Process any remaining audio  
        - Stop threads and streams  
        """  
        self.continuous_mode = False  
        
        # Process final segment if any  
        if self.recorded_audio:  
            self.process_audio_segment()  
        
        # Stop stream  
        if hasattr(self, 'stream'):  
            self.stream.stop()  
            self.stream.close()  
        
        # Wait for recognition thread  
        if hasattr(self, 'recognition_thread'):  
            self.recognition_thread.join()  
        
        self.logger.info("Continuous transcription stopped")     