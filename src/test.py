"""
Real-time Audio Privacy Processor
A privacy-enhancing audio processor that applies voice effects to protect user identity.
Enhanced version with clear command mode and distortion-based privacy mode.
"""

import numpy as np
import sounddevice as sd
import speech_recognition as sr
import threading
import queue
import time
from scipy import signal
from scipy.io.wavfile import write
import tempfile
import os
import warnings
warnings.filterwarnings("ignore")

class AudioPrivacyProcessor:
    def __init__(self, sample_rate=16000, block_size=2048, input_device=None, output_device=None):
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.input_device = input_device
        self.output_device = output_device
        self.audio_queue = queue.Queue()
        self.output_queue = queue.Queue()
        self.wake_words = ['alexa', 'google', 'hey', 'ok', 'okay']
        self.wake_word_detected = False
        self.command_mode_timer = 0
        self.command_duration = 5.0  # seconds to apply command voice effect
        
        # Audio buffer for wake word detection
        self.audio_buffer = np.array([])
        self.buffer_duration = 2.0  # seconds
        self.max_buffer_size = int(self.buffer_duration * self.sample_rate)
        
        # Voice effect parameters - separated into clear and distortion effects
        self.command_effects = {
            'robot': {'pitch_shift': 0.9, 'modulate': True, 'mod_freq': 8.0, 'mod_depth': 0.2},
            'chipmunk': {'pitch_shift': 1.4, 'modulate': False},
            'deep': {'pitch_shift': 0.7, 'modulate': False},
            'smooth': {'pitch_shift': 1.1, 'modulate': True, 'mod_freq': 3.0, 'mod_depth': 0.1},
            'clear': {'pitch_shift': 1.0, 'modulate': False}  # Minimal effect for testing
        }
        
        # Privacy effects include noise and distortion
        self.privacy_effects = {
            'distorted': {'pitch_shift': 0.85, 'add_noise': True, 'noise_level': 0.08, 'distortion': True},
            'static': {'pitch_shift': 1.0, 'add_noise': True, 'noise_level': 0.12, 'distortion': False},
            'scrambled': {'pitch_shift': 1.2, 'add_noise': True, 'noise_level': 0.06, 'distortion': True, 'modulate': True, 'mod_freq': 12.0, 'mod_depth': 0.4},
            'whisper_noise': {'pitch_shift': 0.95, 'add_noise': True, 'noise_level': 0.10, 'distortion': False}
        }
        
        self.current_command_effect = 'robot'
        self.privacy_effect = 'distorted'
        
        # Initialize speech recognizer with optimized settings
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 150
        self.recognizer.pause_threshold = 0.3
        self.recognizer.phrase_threshold = 0.3
        
        print("Audio Privacy Processor initialized")
        print(f"Available command effects: {list(self.command_effects.keys())}")
        print(f"Available privacy effects: {list(self.privacy_effects.keys())}")

    def list_audio_devices(self):
        """List all available audio input and output devices"""
        print("\n=== Available Audio Devices ===")
        devices = sd.query_devices()
        
        print("\nINPUT DEVICES (Microphones):")
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                print(f"  [{i}] {device['name']} - {device['max_input_channels']} channels")
        
        print("\nOUTPUT DEVICES (Speakers/Headphones):")
        for i, device in enumerate(devices):
            if device['max_output_channels'] > 0:
                print(f"  [{i}] {device['name']} - {device['max_output_channels']} channels")
        
        print(f"\nDefault input device: [{sd.default.device[0]}] {devices[sd.default.device[0]]['name']}")
        print(f"Default output device: [{sd.default.device[1]}] {devices[sd.default.device[1]]['name']}")
        
        return devices

    def select_audio_devices(self):
        """Interactive device selection"""
        devices = self.list_audio_devices()
        
        print("\n" + "="*50)
        
        # Select input device
        while True:
            try:
                choice = input(f"\nSelect INPUT device (0-{len(devices)-1}, or Enter for default): ").strip()
                if choice == "":
                    self.input_device = None
                    print(f"Using default input device: {devices[sd.default.device[0]]['name']}")
                    break
                else:
                    device_id = int(choice)
                    if 0 <= device_id < len(devices) and devices[device_id]['max_input_channels'] > 0:
                        self.input_device = device_id
                        print(f"Selected input device: {devices[device_id]['name']}")
                        break
                    else:
                        print("Invalid device ID or device doesn't support input. Please try again.")
            except ValueError:
                print("Please enter a valid number or press Enter for default.")
        
        # Select output device
        while True:
            try:
                choice = input(f"\nSelect OUTPUT device (0-{len(devices)-1}, or Enter for default): ").strip()
                if choice == "":
                    self.output_device = None
                    print(f"Using default output device: {devices[sd.default.device[1]]['name']}")
                    break
                else:
                    device_id = int(choice)
                    if 0 <= device_id < len(devices) and devices[device_id]['max_output_channels'] > 0:
                        self.output_device = device_id
                        print(f"Selected output device: {devices[device_id]['name']}")
                        break
                    else:
                        print("Invalid device ID or device doesn't support output. Please try again.")
            except ValueError:
                print("Please enter a valid number or press Enter for default.")
        
        print("\n" + "="*50)

    def apply_pitch_shift_clean(self, audio_data, shift_factor):
        """Apply clean pitch shifting without artifacts"""
        if shift_factor == 1.0:
            return audio_data
            
        # Use a more sophisticated pitch shifting approach
        # Simple time-domain method with interpolation
        original_length = len(audio_data)
        
        # Create new time indices
        new_indices = np.arange(0, original_length, shift_factor)
        new_indices = new_indices[new_indices < original_length]
        
        # Interpolate for smoother results
        shifted = np.interp(np.arange(original_length), new_indices, audio_data[new_indices.astype(int)])
        
        return shifted

    def apply_distortion(self, audio_data, intensity=0.3):
        """Apply controlled distortion for privacy masking"""
        # Soft clipping distortion
        threshold = 0.5
        distorted = np.where(np.abs(audio_data) > threshold,
                           threshold * np.sign(audio_data) + (audio_data - threshold * np.sign(audio_data)) * intensity,
                           audio_data)
        return distorted

    def add_noise(self, audio_data, noise_level=0.05):
        """Add background noise for privacy masking"""
        noise = np.random.normal(0, noise_level, len(audio_data))
        return audio_data + noise

    def apply_clean_modulation(self, audio_data, freq=5.0, depth=0.2):
        """Apply clean amplitude modulation for robotic effect"""
        t = np.arange(len(audio_data)) / self.sample_rate
        modulation = 1 + depth * np.sin(2 * np.pi * freq * t)
        return audio_data * modulation

    def apply_command_effect(self, audio_data, effect_name):
        """Apply clean command voice effect (no distortion/noise)"""
        if effect_name not in self.command_effects:
            return audio_data
            
        effect = self.command_effects[effect_name]
        processed_audio = audio_data.copy()
        
        # Apply pitch shift with clean algorithm
        if 'pitch_shift' in effect:
            processed_audio = self.apply_pitch_shift_clean(processed_audio, effect['pitch_shift'])
        
        # Apply clean modulation if specified
        if effect.get('modulate', False):
            mod_freq = effect.get('mod_freq', 5.0)
            mod_depth = effect.get('mod_depth', 0.2)
            processed_audio = self.apply_clean_modulation(processed_audio, mod_freq, mod_depth)
        
        # Normalize to prevent clipping but maintain quality
        max_val = np.max(np.abs(processed_audio))
        if max_val > 0:
            processed_audio = processed_audio / max_val * 0.8  # Keep good volume level
            
        return processed_audio

    def apply_privacy_effect(self, audio_data, effect_name):
        """Apply privacy masking effect (includes distortion and noise)"""
        if effect_name not in self.privacy_effects:
            return audio_data
            
        effect = self.privacy_effects[effect_name]
        processed_audio = audio_data.copy()
        
        # Apply pitch shift
        if 'pitch_shift' in effect:
            processed_audio = self.apply_pitch_shift_clean(processed_audio, effect['pitch_shift'])
        
        # Add noise for privacy masking
        if effect.get('add_noise', False):
            noise_level = effect.get('noise_level', 0.05)
            processed_audio = self.add_noise(processed_audio, noise_level)
        
        # Apply distortion for privacy masking
        if effect.get('distortion', False):
            processed_audio = self.apply_distortion(processed_audio, 0.4)
        
        # Apply modulation if specified
        if effect.get('modulate', False):
            mod_freq = effect.get('mod_freq', 8.0)
            mod_depth = effect.get('mod_depth', 0.3)
            processed_audio = self.apply_clean_modulation(processed_audio, mod_freq, mod_depth)
        
        # Normalize
        max_val = np.max(np.abs(processed_audio))
        if max_val > 0:
            processed_audio = processed_audio / max_val * 0.6  # Reduce volume for privacy
            
        return processed_audio

    def detect_wake_word(self, audio_data):
        """Detect wake words in audio data using speech recognition"""
        try:
            # Skip if audio is too quiet
            if np.max(np.abs(audio_data)) < 0.01:
                return False
                
            # Save audio to temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            # Ensure audio is in proper range
            audio_int = np.clip(audio_data * 32767, -32767, 32767).astype(np.int16)
            write(temp_file.name, self.sample_rate, audio_int)
            temp_file.close()
            
            # Recognize speech
            with sr.AudioFile(temp_file.name) as source:
                # Adjust for ambient noise
                self.recognizer.adjust_for_ambient_noise(source, duration=0.1)
                audio = self.recognizer.record(source)
                
                # Try recognition
                text = self.recognizer.recognize_google(audio, language='en-US')
                text = text.lower().strip()
                
                print(f"Detected speech: '{text}'")
                
                # Check for wake words
                for wake_word in self.wake_words:
                    if wake_word in text or any(wake_word in word for word in text.split()):
                        print(f"✓ Wake word '{wake_word}' detected! Switching to CLEAR COMMAND MODE")
                        os.unlink(temp_file.name)
                        return True
                        
            os.unlink(temp_file.name)
            return False
            
        except sr.WaitTimeoutError:
            pass
        except sr.UnknownValueError:
            pass  # No speech detected
        except sr.RequestError as e:
            print(f"Speech recognition error: {e}")
        except Exception as e:
            print(f"Wake word detection error: {e}")
        
        # Clean up temp file if it exists
        try:
            os.unlink(temp_file.name)
        except:
            pass
        return False

    def process_audio_block(self, audio_data):
        """Process a single block of audio data"""
        current_time = time.time()
        
        # Add to buffer for wake word detection
        self.audio_buffer = np.concatenate([self.audio_buffer, audio_data])
        if len(self.audio_buffer) > self.max_buffer_size:
            self.audio_buffer = self.audio_buffer[-self.max_buffer_size:]
        
        # Check if we're in command mode (after wake word detection)
        if self.wake_word_detected:
            if current_time - self.command_mode_timer < self.command_duration:
                # Apply CLEAR command effect (no distortion, clean voice effect only)
                return self.apply_command_effect(audio_data, self.current_command_effect)
            else:
                # Command mode expired
                self.wake_word_detected = False
                print("→ Command mode ended, returning to PRIVACY DISTORTION mode")
        
        # Check for wake words in buffered audio (less frequently for performance)
        if not self.wake_word_detected and len(self.audio_buffer) >= self.max_buffer_size:
            # Check every 8th block to reduce processing load
            if hasattr(self, 'wake_word_check_counter'):
                self.wake_word_check_counter += 1
            else:
                self.wake_word_check_counter = 0
                
            if self.wake_word_check_counter % 8 == 0:
                if self.detect_wake_word(self.audio_buffer[-int(1.5 * self.sample_rate):]):
                    self.wake_word_detected = True
                    self.command_mode_timer = current_time
                    print(f"→ COMMAND MODE ACTIVE - Effect: '{self.current_command_effect}' (CLEAR AUDIO)")
                    return self.apply_command_effect(audio_data, self.current_command_effect)
        
        # Default: apply privacy masking with distortion to all other audio
        return self.apply_privacy_effect(audio_data, self.privacy_effect)

    def audio_callback(self, indata, outdata, frames, time, status):
        """Audio callback function for real-time processing"""
        if status:
            # Only print critical errors
            if 'overflow' not in str(status).lower():
                print(f"Audio callback status: {status}")
        
        # Convert to mono if stereo
        if indata.ndim > 1:
            audio_mono = np.mean(indata, axis=1)
        else:
            audio_mono = indata.flatten()
        
        # Process the audio
        try:
            processed_audio = self.process_audio_block(audio_mono)
            
            # Convert back to output format
            if outdata.ndim > 1:
                outdata[:] = processed_audio.reshape(-1, 1)
            else:
                outdata[:] = processed_audio
                
        except Exception as e:
            print(f"Processing error: {e}")
            # Apply minimal privacy effect as fallback
            try:
                minimal_effect = self.apply_privacy_effect(audio_mono, 'whisper_noise')
                if outdata.ndim > 1:
                    outdata[:] = minimal_effect.reshape(-1, 1)
                else:
                    outdata[:] = minimal_effect
            except:
                # Last resort: silence
                outdata.fill(0)

    def set_command_effect(self, effect_name):
        """Set the voice effect for command mode"""
        if effect_name in self.command_effects:
            self.current_command_effect = effect_name
            print(f"Command effect set to: {effect_name}")
        else:
            print(f"Unknown command effect: {effect_name}")
            print(f"Available command effects: {list(self.command_effects.keys())}")

    def set_privacy_effect(self, effect_name):
        """Set the voice effect for privacy mode"""
        if effect_name in self.privacy_effects:
            self.privacy_effect = effect_name
            print(f"Privacy effect set to: {effect_name}")
        else:
            print(f"Unknown privacy effect: {effect_name}")
            print(f"Available privacy effects: {list(self.privacy_effects.keys())}")

    def start_processing(self):
        """Start real-time audio processing"""
        print("\n" + "="*60)
        print("🎤 STARTING REAL-TIME AUDIO PROCESSING")
        print("="*60)
        print("📱 PRIVACY MODE: Distorted audio (default)")
        print("🗣️  COMMAND MODE: Clear voice effects after wake words")
        print("⏰ Command mode duration: 5 seconds")
        print("🔄 Press Ctrl+C to stop")
        print("="*60)
        print(f"🎵 Command effect: {self.current_command_effect}")
        print(f"🔒 Privacy effect: {self.privacy_effect}")
        print("="*60)
        
        try:
            with sd.Stream(
                samplerate=self.sample_rate,
                blocksize=self.block_size,
                channels=1,
                dtype='float32',
                device=(self.input_device, self.output_device),
                callback=self.audio_callback
            ):
                print("\n✅ Audio processing started!")
                print("💬 Try saying: 'Hey', 'OK Google', 'Alexa'...")
                print("🔊 Your voice is now being processed...")
                while True:
                    time.sleep(0.1)
                    
        except KeyboardInterrupt:
            print("\n\n🛑 Stopping audio processing...")
            print("✅ Audio processor stopped successfully")
        except Exception as e:
            print(f"❌ Error during processing: {e}")

def main():
    """Main function to run the audio privacy processor"""
    print("🎤 === ENHANCED AUDIO PRIVACY PROCESSOR ===")
    print("🔒 Privacy Protection: Distorts voice when not in command mode")
    print("🗣️  Clear Commands: Clean voice effects when wake words detected")
    print("🎵 Wake words: Hey, OK, Alexa, Google")
    print("")
    
    # Create processor instance
    processor = AudioPrivacyProcessor(sample_rate=16000, block_size=2048)
    
    # Device selection
    print("STEP 1: Audio Device Selection")
    device_choice = input("Select specific audio devices? (y/n, default: n): ").lower().strip()
    
    if device_choice == 'y':
        processor.select_audio_devices()
    else:
        print("Using default audio devices...")
    
    print("\nSTEP 2: Voice Effect Configuration")
    print("Current settings:")
    print(f"- Command mode effect: {processor.current_command_effect} (clear, no distortion)")
    print(f"- Privacy mode effect: {processor.privacy_effect} (distorted for privacy)")
    
    # Allow customization
    response = input("Change effects? (y/n, default: n): ").lower().strip()
    if response == 'y':
        print(f"Command effects (clear): {list(processor.command_effects.keys())}")
        command_effect = input("Command effect: ").strip()
        if command_effect:
            processor.set_command_effect(command_effect)
            
        print(f"Privacy effects (distorted): {list(processor.privacy_effects.keys())}")
        privacy_effect = input("Privacy effect: ").strip()
        if privacy_effect:
            processor.set_privacy_effect(privacy_effect)
    
    # Display final configuration
    print(f"\n🔧 === FINAL CONFIGURATION ===")
    if processor.input_device is not None:
        devices = sd.query_devices()
        print(f"🎤 Input: [{processor.input_device}] {devices[processor.input_device]['name']}")
    else:
        print("🎤 Input: Default system microphone")
    
    if processor.output_device is not None:
        devices = sd.query_devices()
        print(f"🔊 Output: [{processor.output_device}] {devices[processor.output_device]['name']}")
    else:
        print("🔊 Output: Default system speakers")
    
    print(f"🎵 Command Effect: {processor.current_command_effect} (CLEAR)")
    print(f"🔒 Privacy Effect: {processor.privacy_effect} (DISTORTED)")
    print("="*40)
    
    input("\n▶️  Press Enter to start processing...")
    
    # Start processing
    processor.start_processing()

if __name__ == "__main__":
    main()