import pyaudio
import numpy as np
import wave
import time
import os
import tempfile
from collections import deque
import matplotlib.pyplot as plt
from scipy import signal as scipy_signal
from scipy.fft import fft, fftfreq

class BatchVoicePrivacyProcessor:
    def __init__(self, sample_rate=44100, chunk_size=1024):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = 1
        self.format = pyaudio.paInt16
        
        # Wake words to detect
        self.wake_words = ['alexa', 'hello alexa', 'hey alexa', 'hello google', 'hey google', 'ok google', 'hey siri']
        
        # Voice activity detection parameters
        self.energy_threshold = 500  # Adjust based on your environment
        self.silence_threshold = 200
        self.min_silence_duration = 1.5  # seconds of silence to mark end of sentence
        
        # Initialize PyAudio
        self.p = pyaudio.PyAudio()
        
        # Create temp directory
        self.temp_dir = tempfile.mkdtemp()
        
        # Selected effects
        self.selected_effects = []
        
    def list_audio_devices(self):
        """List available audio devices"""
        print("\n" + "="*70)
        print("AVAILABLE AUDIO DEVICES:")
        print("="*70)
        for i in range(self.p.get_device_count()):
            info = self.p.get_device_info_by_index(i)
            device_type = []
            if info['maxInputChannels'] > 0:
                device_type.append("INPUT")
            if info['maxOutputChannels'] > 0:
                device_type.append("OUTPUT")
            
            if device_type:
                print(f"{i}: [{'/'.join(device_type)}] {info['name']}")
        print("="*70 + "\n")
    
    def record_audio(self, duration, input_device=None):
        """Record audio for specified duration"""
        print(f"\n🎤 Recording for {duration} seconds...")
        print("💬 Start your conversation now!")
        print("   (Include wake words like 'Hey Alexa' when giving commands)")
        
        frames = []
        
        try:
            stream = self.p.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=input_device,
                frames_per_buffer=self.chunk_size
            )
            
            total_chunks = int(self.sample_rate / self.chunk_size * duration)
            
            for i in range(total_chunks):
                data = stream.read(self.chunk_size, exception_on_overflow=False)
                frames.append(data)
                
                # Show progress
                progress = (i + 1) / total_chunks
                bar_length = 40
                filled = int(bar_length * progress)
                bar = '█' * filled + '░' * (bar_length - filled)
                print(f"\r  [{bar}] {progress*100:.1f}%", end="", flush=True)
            
            print("\n✓ Recording complete!")
            
            stream.stop_stream()
            stream.close()
            
            return frames
            
        except Exception as e:
            print(f"\n❌ Error recording audio: {e}")
            return None
    
    def frames_to_audio(self, frames):
        """Convert audio frames to numpy array"""
        audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
        return audio_data
    
    def audio_to_float(self, audio_int16):
        """Convert int16 audio to float32 normalized"""
        return audio_int16.astype(np.float32) / 32768.0
    
    def float_to_audio(self, audio_float):
        """Convert float32 audio back to int16"""
        return (audio_float * 32767).astype(np.int16)
    
    def calculate_energy(self, audio_segment):
        """Calculate energy of audio segment"""
        return np.sqrt(np.mean(audio_segment.astype(np.float32) ** 2))
    
    def detect_speech_segments(self, audio_data):
        """Detect segments where speech is present"""
        chunk_duration = 0.1  # 100ms chunks
        chunk_samples = int(self.sample_rate * chunk_duration)
        
        speech_segments = []
        is_speaking = False
        segment_start = 0
        silence_chunks = 0
        silence_threshold_chunks = int(self.min_silence_duration / chunk_duration)
        
        for i in range(0, len(audio_data), chunk_samples):
            chunk = audio_data[i:i + chunk_samples]
            if len(chunk) == 0:
                break
                
            energy = self.calculate_energy(chunk)
            
            if energy > self.energy_threshold:
                if not is_speaking:
                    # Start of new speech segment
                    segment_start = i
                    is_speaking = True
                silence_chunks = 0
            else:
                if is_speaking:
                    silence_chunks += 1
                    if silence_chunks >= silence_threshold_chunks:
                        # End of speech segment
                        speech_segments.append((segment_start, i))
                        is_speaking = False
                        silence_chunks = 0
        
        # Close last segment if still speaking
        if is_speaking:
            speech_segments.append((segment_start, len(audio_data)))
        
        return speech_segments
    
    def detect_wake_word_segments(self, audio_data, speech_segments):
        """Detect which speech segments contain wake words (improved heuristic)"""
        wake_segments = []
        
        print("\n   📊 Speech segment analysis:")
        for i, (start, end) in enumerate(speech_segments):
            segment_audio = audio_data[start:end]
            energy = self.calculate_energy(segment_audio)
            duration = (end - start) / self.sample_rate
            
            start_time = start / self.sample_rate
            end_time = end / self.sample_rate
            
            print(f"      Segment {i+1}: {start_time:.2f}s - {end_time:.2f}s | Duration: {duration:.2f}s | Energy: {energy:.0f}")
            
            # More lenient heuristics for wake word detection
            # Include segment if it's moderately long and has decent energy
            if 0.3 < duration < 5.0 and energy > self.energy_threshold * 0.8:
                # Include this segment and potentially the next one (command)
                if i + 1 < len(speech_segments):
                    # Include current + next segment
                    wake_segments.append((start, speech_segments[i + 1][1]))
                    print(f"      ✓ Detected as potential wake word + command")
                else:
                    wake_segments.append((start, end))
                    print(f"      ✓ Detected as potential wake word")
        
        return wake_segments
    
    def manual_segment_selection(self, audio_data, speech_segments):
        """Allow manual selection of wake word segments with detailed analysis"""
        print("\n" + "="*70)
        print("MANUAL WAKE WORD SEGMENT SELECTION")
        print("="*70)
        print("\n📊 Detailed Speech Segment Analysis:")
        print("-" * 70)
        
        # Show detailed analysis for each segment
        for i, (start, end) in enumerate(speech_segments):
            segment_audio = audio_data[start:end]
            energy = self.calculate_energy(segment_audio)
            duration = (end - start) / self.sample_rate
            start_time = start / self.sample_rate
            end_time = end / self.sample_rate
            
            # Determine if it looks like a wake word candidate
            is_candidate = (0.3 < duration < 5.0 and energy > self.energy_threshold * 0.8)
            candidate_marker = "⭐ LIKELY WAKE WORD" if is_candidate else "   regular speech"
            
            print(f"\nSegment {i+1}: [{candidate_marker}]")
            print(f"  ⏱️  Time: {start_time:.2f}s - {end_time:.2f}s")
            print(f"  ⏳ Duration: {duration:.2f} seconds")
            print(f"  📊 Energy Level: {energy:.0f}")
            print(f"  🎯 Energy Ratio: {energy/self.energy_threshold:.2f}x threshold")
        
        print("\n" + "-" * 70)
        print("\n💡 Instructions:")
        print("   • Enter segment numbers that contain wake words + commands")
        print("   • Example: 2,4 (for segments 2 and 4)")
        print("   • Look for segments marked with ⭐ - they're likely wake words")
        print("   • Press Enter to use automatic detection instead")
        
        choice = input("\n👉 Your selection: ").strip()
        
        if not choice:
            print("   → Using automatic detection")
            return None  # Use automatic detection
        
        try:
            selected_indices = [int(x.strip()) - 1 for x in choice.split(",")]
            wake_segments = []
            
            print("\n✅ Selection confirmed:")
            for idx in selected_indices:
                if 0 <= idx < len(speech_segments):
                    wake_segments.append(speech_segments[idx])
                    start_time = speech_segments[idx][0] / self.sample_rate
                    end_time = speech_segments[idx][1] / self.sample_rate
                    print(f"   ✓ Segment {idx+1}: {start_time:.2f}s - {end_time:.2f}s")
                else:
                    print(f"   ✗ Invalid segment number: {idx+1}")
            
            return wake_segments
            
        except ValueError:
            print("   ✗ Invalid input, using automatic detection")
            return None
    
    # ==================== EFFECT FUNCTIONS ====================
    
    def robot_voice_effect(self, audio_float):
        """Apply robot voice effect (CLEAR - for wake word)"""
        t = np.linspace(0, len(audio_float) / self.sample_rate, len(audio_float))
        carrier = np.sin(2 * np.pi * 40 * t)  # 40 Hz carrier - still intelligible
        modulated = audio_float * (1 + 0.3 * carrier)  # Reduced modulation for clarity
        return modulated
    
    def demon_voice_effect(self, audio_float):
        """Apply demon/deep voice effect (CLEAR - for wake word)"""
        # Moderate pitch shift down - still understandable
        shift_factor = 0.85  # Slightly deeper, but clear
        indices = np.arange(0, len(audio_float), shift_factor)
        indices = np.clip(indices, 0, len(audio_float) - 1).astype(int)
        shifted = audio_float[indices]
        
        # Pad or trim to original length
        if len(shifted) < len(audio_float):
            shifted = np.pad(shifted, (0, len(audio_float) - len(shifted)), mode='edge')
        else:
            shifted = shifted[:len(audio_float)]
        
        # Light distortion for character, but keep clarity
        distorted = np.tanh(shifted * 1.5) * 0.9
        
        return distorted
    
    def chipmunk_voice_effect(self, audio_float):
        """Apply chipmunk/high voice effect (CLEAR - for wake word)"""
        # Moderate pitch shift up - still clear
        shift_factor = 1.3  # Higher but understandable
        indices = np.arange(0, len(audio_float), shift_factor)
        indices = np.clip(indices, 0, len(audio_float) - 1).astype(int)
        shifted = audio_float[indices]
        
        # Pad or trim
        if len(shifted) < len(audio_float):
            shifted = np.pad(shifted, (0, len(audio_float) - len(shifted)), mode='edge')
        else:
            shifted = shifted[:len(audio_float)]
        
        return shifted * 0.95
    
    def glitch_effect(self, audio_float):
        """Apply glitch effect (LIGHT - for wake word)"""
        glitch_probability = 0.01  # Very light glitching
        glitch_mask = np.random.random(len(audio_float)) < glitch_probability
        
        glitched = audio_float.copy()
        glitched[glitch_mask] = np.random.uniform(-0.2, 0.2, np.sum(glitch_mask))
        
        return glitched
    
    def echo_effect(self, audio_float):
        """Apply echo effect (CLEAR - for wake word)"""
        delay_samples = int(0.15 * self.sample_rate)  # 150ms delay
        echo_strength = 0.3  # Light echo
        
        padded = np.zeros(len(audio_float) + delay_samples)
        padded[:len(audio_float)] = audio_float
        padded[delay_samples:len(audio_float) + delay_samples] += audio_float * echo_strength
        
        return padded[:len(audio_float)]
    
    def reverse_effect(self, audio_float):
        """Apply reverse effect"""
        return np.flip(audio_float)
    
    def bitcrush_light(self, audio_float):
        """Apply light bitcrushing (CLEAR - for wake word)"""
        bits = 10  # Higher bits = more clarity
        levels = 2 ** bits
        return np.round(audio_float * levels) / levels
    
    # ==================== DISTORTION FUNCTIONS ====================
    
    def apply_heavy_distortion(self, audio_float):
        """Apply HEAVY distortion for private conversations - VERY UNCLEAR"""
        # Extreme pitch shift down
        shift_factor = 0.5  # Very deep
        indices = np.arange(0, len(audio_float), shift_factor)
        indices = np.clip(indices, 0, len(audio_float) - 1).astype(int)
        distorted = audio_float[indices]
        
        # Pad or trim
        if len(distorted) < len(audio_float):
            distorted = np.pad(distorted, (0, len(audio_float) - len(distorted)), mode='edge')
        else:
            distorted = distorted[:len(audio_float)]
        
        # HEAVY bitcrush - very lo-fi
        bits = 3  # Very low bits
        levels = 2 ** bits
        distorted = np.round(distorted * levels) / levels
        
        # Add significant noise
        noise = np.random.normal(0, 0.1, len(distorted))
        distorted = distorted + noise
        
        # Severe clipping and distortion
        distorted = np.tanh(distorted * 4.0) * 0.6
        
        # Add warble/wobble effect
        t = np.linspace(0, len(distorted) / self.sample_rate, len(distorted))
        wobble = np.sin(2 * np.pi * 3 * t) * 0.3
        distorted = distorted * (1 + wobble)
        
        # Normalize
        max_val = np.max(np.abs(distorted))
        if max_val > 0:
            distorted = distorted / max_val * 0.8
        
        return distorted
    
    def apply_selected_effects(self, audio_float):
        """Apply selected effects to wake word and command - CLEAR and INTELLIGIBLE"""
        processed = audio_float.copy()
        
        # Apply each selected effect with careful processing to maintain clarity
        for effect in self.selected_effects:
            if effect == "robot":
                processed = self.robot_voice_effect(processed)
            elif effect == "demon":
                processed = self.demon_voice_effect(processed)
            elif effect == "chipmunk":
                processed = self.chipmunk_voice_effect(processed)
            elif effect == "glitch":
                processed = self.glitch_effect(processed)
            elif effect == "echo":
                processed = self.echo_effect(processed)
            elif effect == "reverse":
                processed = self.reverse_effect(processed)
            elif effect == "bitcrush":
                processed = self.bitcrush_light(processed)
        
        # If no effects selected, keep original (clear voice)
        if not self.selected_effects:
            processed = audio_float
        
        # Normalize with headroom for clarity
        max_val = np.max(np.abs(processed))
        if max_val > 0:
            processed = processed / max_val * 0.90  # Keep clear and loud
        
        return processed
    
    def save_audio(self, audio_data, filename):
        """Save audio to WAV file"""
        wf = wave.open(filename, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(self.sample_rate)
        wf.writeframes(audio_data.tobytes())
        wf.close()
        print(f"💾 Saved to: {filename}")
    
    def process_audio_with_wake_detection(self, audio_data, manual_mode=False):
        """Process audio: heavy distortion for private, selected effects for wake word + command"""
        print("\n🔍 Analyzing audio for speech and wake words...")
        
        # Detect speech segments
        speech_segments = self.detect_speech_segments(audio_data)
        print(f"   Found {len(speech_segments)} speech segments")
        
        if len(speech_segments) == 0:
            print("   ⚠️  No speech detected! Check your microphone or sensitivity settings.")
            # Return heavily distorted audio
            audio_float = self.audio_to_float(audio_data)
            processed = self.apply_heavy_distortion(audio_float)
            return self.float_to_audio(processed), []
        
        # Get wake word segments (manual or automatic)
        if manual_mode:
            wake_segments = self.manual_segment_selection(audio_data, speech_segments)
            if wake_segments is None:
                # User chose automatic
                wake_segments = self.detect_wake_word_segments(audio_data, speech_segments)
        else:
            wake_segments = self.detect_wake_word_segments(audio_data, speech_segments)
        
        if len(wake_segments) == 0:
            print("\n   ⚠️  No wake words detected!")
            print("   💡 Tip: Use manual mode to select segments, or adjust sensitivity")
            # Ask if user wants to manually select
            retry = input("   Would you like to manually select segments? (y/n): ").strip().lower()
            if retry == 'y':
                wake_segments = self.manual_segment_selection(audio_data, speech_segments)
                if wake_segments is None or len(wake_segments) == 0:
                    print("   → Processing with all speech as private conversation")
        else:
            print(f"   Detected {len(wake_segments)} wake word + command segment(s)")
        
        # Convert to float for processing
        audio_float = self.audio_to_float(audio_data)
        processed_audio = np.zeros_like(audio_float)
        
        # Create a mask for wake word + command segments
        wake_mask = np.zeros(len(audio_float), dtype=bool)
        
        for start, end in wake_segments:
            wake_mask[start:end] = True
            duration = (end - start) / self.sample_rate
            print(f"   🎤 Wake word + command: {start/self.sample_rate:.2f}s - {end/self.sample_rate:.2f}s ({duration:.2f}s)")
        
        print("\n🎭 Applying effects...")
        print(f"   🔒 Private conversation: HEAVY distortion (unintelligible)")
        if self.selected_effects:
            print(f"   🎪 Wake word + command: {', '.join(self.selected_effects).upper()} (CLEAR & intelligible)")
        else:
            print(f"   🎪 Wake word + command: CLEAR voice (no effects)")
        
        # Process in chunks
        chunk_size = self.sample_rate  # 1 second chunks
        for i in range(0, len(audio_float), chunk_size):
            end_idx = min(i + chunk_size, len(audio_float))
            chunk = audio_float[i:end_idx]
            mask_chunk = wake_mask[i:end_idx]
            
            # Apply heavy distortion to private conversation parts
            distorted_chunk = self.apply_heavy_distortion(chunk)
            
            # Apply selected effects to wake word + command parts (CLEAR)
            effects_chunk = self.apply_selected_effects(chunk)
            
            # Mix: use clear effects for wake segments, heavy distortion for private
            processed_audio[i:end_idx] = np.where(mask_chunk, effects_chunk, distorted_chunk)
        
        print("   ✓ Effects applied successfully")
        
        # Convert back to int16 and return with wake segments info
        return self.float_to_audio(processed_audio), wake_segments
    
    def play_audio(self, audio_data, output_device=None):
        """Play audio data"""
        print("\n🔊 Playing processed audio...")
        
        try:
            stream = self.p.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                output=True,
                output_device_index=output_device
            )
            
            # Play in chunks
            chunk_size = self.chunk_size * 2  # 2 bytes per sample (int16)
            audio_bytes = audio_data.tobytes()
            
            total_chunks = len(audio_bytes) // chunk_size
            for i in range(0, len(audio_bytes), chunk_size):
                chunk = audio_bytes[i:i + chunk_size]
                stream.write(chunk)
                
                # Show progress
                progress = (i // chunk_size + 1) / total_chunks
                bar_length = 40
                filled = int(bar_length * progress)
                bar = '█' * filled + '░' * (bar_length - filled)
                print(f"\r  [{bar}] {progress*100:.1f}%", end="", flush=True)
            
            print("\n✓ Playback complete!")
            
            stream.stop_stream()
            stream.close()
            
        except Exception as e:
            print(f"\n❌ Error playing audio: {e}")
    
    def visualize_audio(self, original_data, processed_data, wake_segments=None):
        """Create 6 comprehensive graphs of the audio output"""
        print("\n📊 Generating audio visualizations...")
        
        # Convert to float for analysis
        original_float = self.audio_to_float(original_data)
        processed_float = self.audio_to_float(processed_data)
        
        # Create time array
        time_array = np.linspace(0, len(original_float) / self.sample_rate, len(original_float))
        
        # Create figure with 6 subplots
        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        fig.suptitle('Audio Analysis - Original vs Processed', fontsize=16, fontweight='bold')
        
        # ========== GRAPH 1: Waveform Comparison ==========
        ax1 = axes[0, 0]
        ax1.plot(time_array, original_float, label='Original', alpha=0.7, linewidth=0.5, color='blue')
        ax1.plot(time_array, processed_float, label='Processed', alpha=0.7, linewidth=0.5, color='red')
        
        # Mark wake word segments
        if wake_segments:
            for start, end in wake_segments:
                start_time = start / self.sample_rate
                end_time = end / self.sample_rate
                ax1.axvspan(start_time, end_time, alpha=0.2, color='green', label='Wake Word' if start == wake_segments[0][0] else '')
        
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Amplitude')
        ax1.set_title('1. Waveform Comparison')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, time_array[-1])
        
        # ========== GRAPH 2: Spectrogram (Original) ==========
        ax2 = axes[0, 1]
        frequencies, times, spectrogram = scipy_signal.spectrogram(
            original_float, 
            fs=self.sample_rate,
            nperseg=1024,
            noverlap=512
        )
        
        spec_db = 10 * np.log10(spectrogram + 1e-10)
        im2 = ax2.pcolormesh(times, frequencies, spec_db, shading='gouraud', cmap='viridis')
        ax2.set_ylabel('Frequency (Hz)')
        ax2.set_xlabel('Time (seconds)')
        ax2.set_title('2. Spectrogram - Original Audio')
        ax2.set_ylim(0, 8000)  # Focus on speech range
        plt.colorbar(im2, ax=ax2, label='Power (dB)')
        
        # ========== GRAPH 3: Spectrogram (Processed) ==========
        ax3 = axes[1, 0]
        frequencies, times, spectrogram = scipy_signal.spectrogram(
            processed_float, 
            fs=self.sample_rate,
            nperseg=1024,
            noverlap=512
        )
        
        spec_db = 10 * np.log10(spectrogram + 1e-10)
        im3 = ax3.pcolormesh(times, frequencies, spec_db, shading='gouraud', cmap='plasma')
        ax3.set_ylabel('Frequency (Hz)')
        ax3.set_xlabel('Time (seconds)')
        ax3.set_title('3. Spectrogram - Processed Audio')
        ax3.set_ylim(0, 8000)
        plt.colorbar(im3, ax=ax3, label='Power (dB)')
        
        # ========== GRAPH 4: Frequency Spectrum Comparison ==========
        ax4 = axes[1, 1]
        
        # Calculate FFT for both
        n_samples = min(len(original_float), self.sample_rate * 5)  # Use first 5 seconds
        original_fft = fft(original_float[:n_samples])
        processed_fft = fft(processed_float[:n_samples])
        freqs = fftfreq(n_samples, 1/self.sample_rate)
        
        # Only positive frequencies
        positive_freqs = freqs[:n_samples//2]
        original_magnitude = np.abs(original_fft[:n_samples//2])
        processed_magnitude = np.abs(processed_fft[:n_samples//2])
        
        ax4.plot(positive_freqs, 20 * np.log10(original_magnitude + 1e-10), label='Original', alpha=0.7, color='blue')
        ax4.plot(positive_freqs, 20 * np.log10(processed_magnitude + 1e-10), label='Processed', alpha=0.7, color='red')
        ax4.set_xlabel('Frequency (Hz)')
        ax4.set_ylabel('Magnitude (dB)')
        ax4.set_title('4. Frequency Spectrum Comparison')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim(0, 8000)  # Focus on speech range
        
        # ========== GRAPH 5: Energy Over Time ==========
        ax5 = axes[2, 0]
        
        # Calculate energy in windows
        window_size = int(0.1 * self.sample_rate)  # 100ms windows
        original_energy = []
        processed_energy = []
        energy_times = []
        
        for i in range(0, len(original_float) - window_size, window_size // 2):
            window_orig = original_float[i:i+window_size]
            window_proc = processed_float[i:i+window_size]
            
            original_energy.append(np.sqrt(np.mean(window_orig ** 2)))
            processed_energy.append(np.sqrt(np.mean(window_proc ** 2)))
            energy_times.append(i / self.sample_rate)
        
        ax5.plot(energy_times, original_energy, label='Original', alpha=0.7, color='blue', linewidth=2)
        ax5.plot(energy_times, processed_energy, label='Processed', alpha=0.7, color='red', linewidth=2)
        
        # Mark wake word segments
        if wake_segments:
            for start, end in wake_segments:
                start_time = start / self.sample_rate
                end_time = end / self.sample_rate
                ax5.axvspan(start_time, end_time, alpha=0.2, color='green')
        
        ax5.set_xlabel('Time (seconds)')
        ax5.set_ylabel('Energy (RMS)')
        ax5.set_title('5. Energy Over Time')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # ========== GRAPH 6: Statistical Comparison ==========
        ax6 = axes[2, 1]
        
        # Calculate various statistics
        stats_labels = ['Mean\nAmplitude', 'Max\nAmplitude', 'RMS\nEnergy', 'Dynamic\nRange (dB)', 'Zero\nCrossings']
        
        original_stats = [
            np.mean(np.abs(original_float)),
            np.max(np.abs(original_float)),
            np.sqrt(np.mean(original_float ** 2)),
            20 * np.log10(np.max(np.abs(original_float)) / (np.mean(np.abs(original_float)) + 1e-10)),
            np.sum(np.diff(np.sign(original_float)) != 0) / len(original_float) * 1000
        ]
        
        processed_stats = [
            np.mean(np.abs(processed_float)),
            np.max(np.abs(processed_float)),
            np.sqrt(np.mean(processed_float ** 2)),
            20 * np.log10(np.max(np.abs(processed_float)) / (np.mean(np.abs(processed_float)) + 1e-10)),
            np.sum(np.diff(np.sign(processed_float)) != 0) / len(processed_float) * 1000
        ]
        
        x = np.arange(len(stats_labels))
        width = 0.35
        
        bars1 = ax6.bar(x - width/2, original_stats, width, label='Original', alpha=0.8, color='blue')
        bars2 = ax6.bar(x + width/2, processed_stats, width, label='Processed', alpha=0.8, color='red')
        
        ax6.set_ylabel('Value')
        ax6.set_title('6. Statistical Comparison')
        ax6.set_xticks(x)
        ax6.set_xticklabels(stats_labels, fontsize=9)
        ax6.legend()
        ax6.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax6.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}',
                        ha='center', va='bottom', fontsize=7)
        
        # Adjust layout and save
        plt.tight_layout()
        
        # Save the figure
        timestamp = int(time.time())
        filename = f"audio_analysis_{timestamp}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"   ✓ Visualization saved as: {filename}")
        
        # Show the plot
        plt.show()
        
        print("   ✓ All 6 graphs generated successfully!")
        
        return filename
    
    def select_effects(self):
        """Interactive effect selection"""
        print("\n" + "="*70)
        print("EFFECT SELECTION FOR WAKE WORD + COMMAND")
        print("="*70)
        print("\nAvailable effects (will be CLEAR and intelligible):")
        print("  1. Robot    - Robotic/vocoder sound")
        print("  2. Demon    - Deep, darker voice")
        print("  3. Chipmunk - High-pitched voice")
        print("  4. Glitch   - Light digital artifacts")
        print("  5. Echo     - Echo/reverb effect")
        print("  6. Reverse  - Backwards audio")
        print("  7. Bitcrush - Retro digital sound")
        print("  0. None     - Keep original clear voice")
        print("\n💡 Note: Multiple effects can be combined!")
        print("="*70 + "\n")
        
        effects_map = {
            "1": "robot",
            "2": "demon",
            "3": "chipmunk",
            "4": "glitch",
            "5": "echo",
            "6": "reverse",
            "7": "bitcrush"
        }
        
        selected = input("Enter effect numbers separated by commas (e.g., 1,2,5) or 0 for none: ").strip()
        
        if selected == "0":
            self.selected_effects = []
            print("✓ No effects - wake word + command will be CLEAR")
        else:
            choices = [c.strip() for c in selected.split(",")]
            self.selected_effects = [effects_map[c] for c in choices if c in effects_map]
            
            if self.selected_effects:
                print(f"✓ Selected effects: {', '.join(self.selected_effects).upper()}")
            else:
                print("✓ No valid effects selected - wake word + command will be CLEAR")
    
    def interactive_session(self):
        """Run interactive batch processing session"""
        print("\n" + "="*70)
        print("       BATCH VOICE PRIVACY PROCESSOR WITH CUSTOMIZABLE EFFECTS")
        print("="*70)
        print("\n📋 How it works:")
        print("   1. Records your entire conversation")
        print("   2. You choose which effects to apply to wake word + command")
        print("   3. Wake word + command get CLEAR effects (intelligible)")
        print("   4. Private conversations get HEAVY distortion (unintelligible)")
        print("   5. Plays back the processed audio")
        print("\n💡 Key difference:")
        print("   • Wake word + command: CLEAR with fun effects")
        print("   • Private conversation: HEAVY distortion for privacy")
        print("="*70 + "\n")
        
        # List and select devices
        self.list_audio_devices()
        
        try:
            input_choice = input("Enter INPUT device number (or Enter for default): ").strip()
            input_device = int(input_choice) if input_choice else None
            
            output_choice = input("Enter OUTPUT device number (or Enter for default): ").strip()
            output_device = int(output_choice) if output_choice else None
        except ValueError:
            input_device = None
            output_device = None
            print("Using default devices")
        
        # Adjust sensitivity
        print("\n📊 Sensitivity Settings:")
        try:
            sensitivity = input("Voice detection sensitivity (1=low, 2=medium, 3=high) [default=2]: ").strip()
            if sensitivity == "1":
                self.energy_threshold = 300
                print("✓ Low sensitivity")
            elif sensitivity == "3":
                self.energy_threshold = 700
                print("✓ High sensitivity")
            else:
                self.energy_threshold = 500
                print("✓ Medium sensitivity")
        except:
            print("✓ Using default medium sensitivity")
        
        # Detection mode
        print("\n🎯 Wake Word Detection Mode:")
        print("  1. Automatic - Let the system detect wake words")
        print("  2. Manual - You select which segments contain wake words")
        detection_mode = input("Choose mode (1 or 2) [default=1]: ").strip()
        manual_mode = (detection_mode == "2")
        
        while True:
            try:
                print("\n" + "="*70)
                
                # Select effects for this recording
                self.select_effects()
                
                # Get recording duration
                duration_input = input("\n⏱️  Recording duration in seconds (default 30, 'q' to quit): ").strip()
                
                if duration_input.lower() == 'q':
                    break
                
                duration = int(duration_input) if duration_input else 30
                
                # Record audio
                frames = self.record_audio(duration, input_device)
                
                if frames is None:
                    continue
                
                # Convert to numpy array
                audio_data = self.frames_to_audio(frames)
                
                if len(audio_data) == 0:
                    print("❌ No audio recorded!")
                    continue
                
                # Save original
                original_file = os.path.join(self.temp_dir, "original.wav")
                self.save_audio(audio_data, original_file)
                print(f"   Original saved temporarily")
                
                # Process audio with wake word detection
                processed_audio, wake_segments = self.process_audio_with_wake_detection(audio_data, manual_mode)
                
                # Play back processed audio
                self.play_audio(processed_audio, output_device)
                
                # Generate visualizations
                visualize = input("\n📊 Generate audio analysis graphs? (y/n): ").strip().lower()
                if visualize == 'y':
                    self.visualize_audio(audio_data, processed_audio, wake_segments)
                
                # Ask to save
                save_choice = input("\n💾 Save processed audio? (y/n): ").strip().lower()
                if save_choice == 'y':
                    filename = f"privacy_protected_{int(time.time())}.wav"
                    self.save_audio(processed_audio, filename)
                else:
                    print("   Not saved")
                
            except KeyboardInterrupt:
                print("\n\nStopping...")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                import traceback
                traceback.print_exc()
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
    processor = BatchVoicePrivacyProcessor()
    
    try:
        processor.interactive_session()
    except KeyboardInterrupt:
        print("\n\nExiting...")
    finally:
        processor.cleanup()
        print("\n👋 Goodbye!\n")

if __name__ == "__main__":
    main()