import pyaudio
import numpy as np
import threading
import time
from scipy import signal
import queue

class AudioDistorter:
    def __init__(self, sample_rate=44100, chunk_size=1024):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.audio_queue = queue.Queue()
        
        # Audio parameters
        self.format = pyaudio.paFloat32
        self.channels = 1
        self.input_device_index = None
        self.output_device_index = None
        
        # Distortion parameters
        self.distortion_gain = 5.0
        self.distortion_threshold = 0.3
        self.bitcrush_bits = 8
        self.delay_samples = int(0.1 * sample_rate)  # 100ms delay
        self.delay_buffer = np.zeros(self.delay_samples)
        self.delay_feedback = 0.3
        
        # Initialize PyAudio
        self.p = pyaudio.PyAudio()
        self.is_running = False
        
    def list_audio_devices(self):
        """List available audio devices"""
        print("Available audio devices:")
        for i in range(self.p.get_device_count()):
            info = self.p.get_device_info_by_index(i)
            print(f"{i}: {info['name']} - {info['maxInputChannels']} in, {info['maxOutputChannels']} out")
    
    def soft_clip(self, audio, threshold=0.7):
        """Apply soft clipping distortion"""
        return np.tanh(audio * self.distortion_gain) * threshold
    
    def hard_clip(self, audio, threshold=0.5):
        """Apply hard clipping distortion"""
        return np.clip(audio * self.distortion_gain, -threshold, threshold)
    
    def bitcrush(self, audio, bits=8):
        """Apply bitcrushing effect"""
        levels = 2 ** bits
        return np.round(audio * levels) / levels
    
    def add_delay(self, audio):
        """Add delay effect with feedback"""
        output = np.zeros_like(audio)
        
        for i in range(len(audio)):
            # Get delayed sample
            delayed_sample = self.delay_buffer[0]
            
            # Shift delay buffer
            self.delay_buffer[:-1] = self.delay_buffer[1:]
            
            # Add current sample + feedback to delay buffer
            self.delay_buffer[-1] = audio[i] + delayed_sample * self.delay_feedback
            
            # Output is current sample + delayed sample
            output[i] = audio[i] + delayed_sample * 0.5
            
        return output
    
    def apply_distortion(self, audio_data):
        """Apply various distortion effects"""
        # Convert to numpy array
        audio = np.frombuffer(audio_data, dtype=np.float32)
        
        # Apply effects in chain
        distorted = self.soft_clip(audio, self.distortion_threshold)
        distorted = self.bitcrush(distorted, self.bitcrush_bits)
        distorted = self.add_delay(distorted)
        
        # Normalize to prevent clipping
        if np.max(np.abs(distorted)) > 0:
            distorted = distorted / np.max(np.abs(distorted)) * 0.8
        
        return distorted.astype(np.float32).tobytes()
    
    def audio_callback(self, in_data, frame_count, time_info, status):
        """Audio input callback"""
        if status:
            print(f"Audio status: {status}")
        
        # Apply distortion
        distorted_data = self.apply_distortion(in_data)
        
        # Put processed audio in queue for output
        try:
            self.audio_queue.put_nowait(distorted_data)
        except queue.Full:
            pass  # Skip if queue is full
        
        return (None, pyaudio.paContinue)
    
    def output_callback(self, in_data, frame_count, time_info, status):
        """Audio output callback"""
        if status:
            print(f"Output status: {status}")
        
        try:
            # Get processed audio from queue
            data = self.audio_queue.get_nowait()
            return (data, pyaudio.paContinue)
        except queue.Empty:
            # Return silence if no data available
            return (np.zeros(frame_count, dtype=np.float32).tobytes(), pyaudio.paContinue)
    
    def start_processing(self):
        """Start real-time audio processing"""
        self.is_running = True
        
        try:
            # Open input stream
            self.input_stream = self.p.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.input_device_index,
                frames_per_buffer=self.chunk_size,
                stream_callback=self.audio_callback
            )
            
            # Open output stream
            self.output_stream = self.p.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                output=True,
                output_device_index=self.output_device_index,
                frames_per_buffer=self.chunk_size,
                stream_callback=self.output_callback
            )
            
            print("Starting audio processing...")
            print("Press Ctrl+C to stop")
            
            # Start streams
            self.input_stream.start_stream()
            self.output_stream.start_stream()
            
            # Keep running until interrupted
            while self.is_running:
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\nStopping audio processing...")
            self.stop_processing()
        except Exception as e:
            print(f"Error: {e}")
            self.stop_processing()
    
    def stop_processing(self):
        """Stop audio processing"""
        self.is_running = False
        
        if hasattr(self, 'input_stream'):
            self.input_stream.stop_stream()
            self.input_stream.close()
        
        if hasattr(self, 'output_stream'):
            self.output_stream.stop_stream()
            self.output_stream.close()
        
        self.p.terminate()
    
    def set_distortion_parameters(self, gain=5.0, threshold=0.3, bits=8, delay_feedback=0.3):
        """Adjust distortion parameters in real-time"""
        self.distortion_gain = gain
        self.distortion_threshold = threshold
        self.bitcrush_bits = bits
        self.delay_feedback = delay_feedback
        print(f"Updated parameters: gain={gain}, threshold={threshold}, bits={bits}, delay_feedback={delay_feedback}")

def main():
    # Create audio distorter
    distorter = AudioDistorter()
    
    # List available devices
    distorter.list_audio_devices()
    
    # You can set specific input/output devices if needed
    # distorter.input_device_index = 0  # Set to your microphone
    # distorter.output_device_index = 1  # Set to your speakers/headphones
    
    print("\nStarting real-time audio distortion...")
    print("Make sure to use headphones to avoid feedback!")
    print("\nControls:")
    print("- The audio will be processed with soft clipping, bitcrushing, and delay")
    print("- Adjust parameters by modifying the code")
    
    try:
        distorter.start_processing()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        distorter.stop_processing()

if __name__ == "__main__":
    main()
