import pyaudio
import numpy as np
import wave
import threading
import time
from scipy import signal
import os
import tempfile

class SpeechDistorter:
    def __init__(self, sample_rate=44100, chunk_size=1024):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = 1
        self.format = pyaudio.paInt16
        self.frames = []
        
        # Distortion parameters
        self.distortion_gain = 3.0
        self.distortion_threshold = 0.5
        self.bitcrush_bits = 6
        self.pitch_shift = 0.5  # Lower = deeper voice
        
        # Initialize PyAudio
        self.p = pyaudio.PyAudio()
        self.is_recording = False
        
        # Create temp directory for audio files
        self.temp_dir = tempfile.mkdtemp()
        
    def list_audio_devices(self):
        """List available audio devices"""
        print("Available audio devices:")
        input_devices = []
        output_devices = []
        
        for i in range(self.p.get_device_count()):
            info = self.p.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                input_devices.append((i, info['name']))
                print(f"INPUT  {i}: {info['name']}")
            if info['maxOutputChannels'] > 0:
                output_devices.append((i, info['name']))
                print(f"OUTPUT {i}: {info['name']}")
        
        return input_devices, output_devices
    
    def record_audio(self, duration=5, input_device=None):
        """Record audio for specified duration"""
        print(f"Recording for {duration} seconds...")
        print("Start speaking now!")
        
        # Get device info to determine correct channels
        if input_device is not None:
            device_info = self.p.get_device_info_by_index(input_device)
            max_channels = device_info['maxInputChannels']
            # Use minimum of requested channels and device max channels
            channels = min(self.channels, max_channels)
        else:
            channels = self.channels
        
        # Try different channel configurations if needed
        stream = None
        for ch in [channels, 1, 2]:  # Try requested, then mono, then stereo
            try:
                stream = self.p.open(
                    format=self.format,
                    channels=ch,
                    rate=self.sample_rate,
                    input=True,
                    input_device_index=input_device,
                    frames_per_buffer=self.chunk_size
                )
                self.channels = ch  # Update channels to working value
                print(f"Recording with {ch} channel(s)")
                break
            except Exception as e:
                print(f"Failed with {ch} channels: {e}")
                continue
        
        if stream is None:
            raise Exception("Could not open audio stream with any channel configuration")
        
        self.frames = []
        self.is_recording = True
        
        # Record audio
        for i in range(0, int(self.sample_rate / self.chunk_size * duration)):
            data = stream.read(self.chunk_size)
            self.frames.append(data)
            
            # Show progress
            progress = (i + 1) / (self.sample_rate / self.chunk_size * duration)
            print(f"\rRecording... {progress*100:.1f}%", end="", flush=True)
        
        print("\nRecording complete!")
        
        # Stop and close stream
        stream.stop_stream()
        stream.close()
        self.is_recording = False
        
        return self.frames
    
    def save_audio(self, frames, filename):
        """Save audio frames to WAV file"""
        wf = wave.open(filename, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.p.get_sample_size(self.format))
        wf.setframerate(self.sample_rate)
        wf.writeframes(b''.join(frames))
        wf.close()
    
    def load_audio(self, filename):
        """Load audio from WAV file"""
        wf = wave.open(filename, 'rb')
        frames = wf.readframes(wf.getnframes())
        channels = wf.getnchannels()
        wf.close()
        
        audio_data = np.frombuffer(frames, dtype=np.int16)
        
        # Convert stereo to mono if needed
        if channels == 2 and len(audio_data) > 0:
            audio_data = audio_data.reshape(-1, 2)
            audio_data = np.mean(audio_data, axis=1).astype(np.int16)
            print("Converted stereo to mono")
        
        return audio_data
    
    def robot_voice_effect(self, audio):
        """Apply robot voice effect"""
        # Convert to float
        audio_float = audio.astype(np.float32) / 32767.0
        
        # Apply ring modulation for robot effect
        t = np.linspace(0, len(audio_float) / self.sample_rate, len(audio_float))
        carrier = np.sin(2 * np.pi * 30 * t)  # 30 Hz carrier
        modulated = audio_float * (1 + 0.5 * carrier)
        
        return modulated
    
    def demon_voice_effect(self, audio):
        """Apply demon/deep voice effect"""
        # Convert to float
        audio_float = audio.astype(np.float32) / 32767.0
        
        # Pitch shift down (demon voice)
        # Simple pitch shift using resampling
        shift_factor = 0.7  # Lower = deeper
        indices = np.arange(0, len(audio_float), shift_factor)
        indices = indices[indices < len(audio_float)].astype(int)
        shifted = audio_float[indices]
        
        # Add some distortion
        shifted = np.tanh(shifted * 2.0) * 0.8
        
        return shifted
    
    def chipmunk_voice_effect(self, audio):
        """Apply chipmunk/high voice effect"""
        # Convert to float
        audio_float = audio.astype(np.float32) / 32767.0
        
        # Pitch shift up (chipmunk voice)
        shift_factor = 1.5  # Higher = more chipmunk
        indices = np.arange(0, len(audio_float), shift_factor)
        indices = indices[indices < len(audio_float)].astype(int)
        shifted = audio_float[indices]
        
        return shifted
    
    def glitch_effect(self, audio):
        """Apply glitch/digital corruption effect"""
        # Convert to float
        audio_float = audio.astype(np.float32) / 32767.0
        
        # Random glitches
        glitch_probability = 0.05  # 5% chance of glitch per sample
        glitch_mask = np.random.random(len(audio_float)) < glitch_probability
        
        # Apply random values where glitches occur
        glitched = audio_float.copy()
        glitched[glitch_mask] = np.random.uniform(-0.5, 0.5, np.sum(glitch_mask))
        
        # Add bitcrushing
        levels = 2 ** self.bitcrush_bits
        glitched = np.round(glitched * levels) / levels
        
        return glitched
    
    def reverse_effect(self, audio):
        """Apply reverse effect"""
        audio_float = audio.astype(np.float32) / 32767.0
        return np.flip(audio_float)
    
    def echo_effect(self, audio):
        """Apply echo effect"""
        audio_float = audio.astype(np.float32) / 32767.0
        
        # Create echo
        delay_samples = int(0.3 * self.sample_rate)  # 300ms delay
        echo_strength = 0.4
        
        # Pad audio for echo
        padded = np.zeros(len(audio_float) + delay_samples)
        padded[:len(audio_float)] = audio_float
        padded[delay_samples:] += audio_float * echo_strength
        
        return padded
    
    def apply_distortion_effect(self, audio, effect_type="robot"):
        """Apply selected distortion effect"""
        effects = {
            "robot": self.robot_voice_effect,
            "demon": self.demon_voice_effect,
            "chipmunk": self.chipmunk_voice_effect,
            "glitch": self.glitch_effect,
            "reverse": self.reverse_effect,
            "echo": self.echo_effect
        }
        
        if effect_type not in effects:
            print(f"Unknown effect: {effect_type}")
            return audio
        
        print(f"Applying {effect_type} effect...")
        distorted = effects[effect_type](audio)
        
        # Normalize and convert back to int16
        if len(distorted) > 0:
            distorted = distorted / np.max(np.abs(distorted)) * 0.8
        
        return (distorted * 32767).astype(np.int16)
    
    def play_audio(self, audio_data, output_device=None):
        """Play audio data"""
        print("Playing distorted audio...")
        
        # Get device info to determine correct channels for output
        if output_device is not None:
            device_info = self.p.get_device_info_by_index(output_device)
            max_channels = device_info['maxOutputChannels']
            output_channels = min(2, max_channels)  # Prefer stereo if available
        else:
            output_channels = 1
        
        # Convert mono to stereo if needed
        if output_channels == 2 and len(audio_data.shape) == 1:
            # Duplicate mono to stereo
            audio_stereo = np.column_stack((audio_data, audio_data))
            audio_data = audio_stereo.flatten()
        
        # Try different channel configurations for output
        stream = None
        for ch in [output_channels, 1, 2]:  # Try preferred, then mono, then stereo
            try:
                stream = self.p.open(
                    format=self.format,
                    channels=ch,
                    rate=self.sample_rate,
                    output=True,
                    output_device_index=output_device
                )
                print(f"Playing with {ch} channel(s)")
                break
            except Exception as e:
                print(f"Failed with {ch} channels: {e}")
                continue
        
        if stream is None:
            raise Exception("Could not open audio output stream")
        
        # Convert to bytes and play
        audio_bytes = audio_data.tobytes()
        
        # Play in chunks
        chunk_size = self.chunk_size * 2 * (2 if output_channels == 2 else 1)  # 2 bytes per sample * channels
        for i in range(0, len(audio_bytes), chunk_size):
            chunk = audio_bytes[i:i + chunk_size]
            stream.write(chunk)
        
        # Stop and close stream
        stream.stop_stream()
        stream.close()
        print("Playback complete!")
    
    def interactive_session(self):
        """Run interactive session"""
        print("=== Speech Distortion Program ===")
        print("This program will record your speech and play it back with distortion effects!")
        print()
        
        # List devices
        input_devices, output_devices = self.list_audio_devices()
        print()
        
        # Device selection
        try:
            input_choice = input("Enter input device number (or press Enter for default): ").strip()
            input_device = int(input_choice) if input_choice else None
            
            output_choice = input("Enter output device number (or press Enter for default): ").strip()
            output_device = int(output_choice) if output_choice else None
        except ValueError:
            input_device = None
            output_device = None
            print("Using default devices")
        
        print()
        
        # Effect selection
        effects = ["robot", "demon", "chipmunk", "glitch", "reverse", "echo"]
        print("Available effects:")
        for i, effect in enumerate(effects):
            print(f"{i+1}. {effect.capitalize()}")
        
        while True:
            try:
                print("\n" + "="*50)
                
                # Get recording duration
                duration_input = input("Recording duration in seconds (default 5): ").strip()
                duration = int(duration_input) if duration_input else 5
                
                # Get effect choice
                effect_choice = input("Choose effect (1-6) or 'q' to quit: ").strip()
                
                if effect_choice.lower() == 'q':
                    break
                
                effect_index = int(effect_choice) - 1
                if 0 <= effect_index < len(effects):
                    effect = effects[effect_index]
                else:
                    print("Invalid choice, using robot effect")
                    effect = "robot"
                
                # Record audio
                print(f"\nRecording {duration} seconds with {effect} effect...")
                frames = self.record_audio(duration, input_device)
                
                # Save original
                original_file = os.path.join(self.temp_dir, "original.wav")
                self.save_audio(frames, original_file)
                
                # Load and process
                audio_data = self.load_audio(original_file)
                
                if len(audio_data) == 0:
                    print("No audio recorded!")
                    continue
                
                # Apply distortion
                distorted_audio = self.apply_distortion_effect(audio_data, effect)
                
                # Play back
                self.play_audio(distorted_audio, output_device)
                
                # Ask to save
                save_choice = input("Save distorted audio? (y/n): ").strip().lower()
                if save_choice == 'y':
                    filename = f"distorted_{effect}_{int(time.time())}.wav"
                    
                    # Save distorted audio
                    wf = wave.open(filename, 'wb')
                    wf.setnchannels(self.channels)
                    wf.setsampwidth(2)  # 16-bit
                    wf.setframerate(self.sample_rate)
                    wf.writeframes(distorted_audio.tobytes())
                    wf.close()
                    
                    print(f"Saved as: {filename}")
                
            except KeyboardInterrupt:
                print("\nStopping...")
                break
            except Exception as e:
                print(f"Error: {e}")
                continue
    
    def cleanup(self):
        """Clean up resources"""
        self.p.terminate()
        
        # Clean up temp files
        try:
            import shutil
            shutil.rmtree(self.temp_dir)
        except:
            pass

def main():
    """Main function"""
    distorter = SpeechDistorter()
    
    try:
        distorter.interactive_session()
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        distorter.cleanup()

if __name__ == "__main__":
    main()