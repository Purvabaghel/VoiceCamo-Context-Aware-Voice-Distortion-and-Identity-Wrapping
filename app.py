import streamlit as st
import numpy as np
import wave
import tempfile
import os
from scipy import signal as scipy_signal
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
import io
import pyaudio
import threading
import time

class StreamlitVoicePrivacyProcessor:
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        self.channels = 1
        self.chunk_size = 1024
        self.format = pyaudio.paInt16
        
        # Voice activity detection parameters
        self.energy_threshold = 500
        self.silence_threshold = 200
        self.min_silence_duration = 1.5
        
        # Initialize PyAudio
        self.p = pyaudio.PyAudio()
        
    def get_audio_devices(self):
        """Get list of available audio devices"""
        devices = []
        for i in range(self.p.get_device_count()):
            info = self.p.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                devices.append({
                    'index': i,
                    'name': info['name']
                })
        return devices
    
    def record_audio(self, duration, input_device=None, progress_callback=None):
        """Record audio for specified duration"""
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
                
                if progress_callback:
                    progress = (i + 1) / total_chunks
                    progress_callback(progress)
            
            stream.stop_stream()
            stream.close()
            
            return frames
            
        except Exception as e:
            st.error(f"Error recording audio: {e}")
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
        chunk_duration = 0.1
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
                    segment_start = i
                    is_speaking = True
                silence_chunks = 0
            else:
                if is_speaking:
                    silence_chunks += 1
                    if silence_chunks >= silence_threshold_chunks:
                        speech_segments.append((segment_start, i))
                        is_speaking = False
                        silence_chunks = 0
        
        if is_speaking:
            speech_segments.append((segment_start, len(audio_data)))
        
        return speech_segments
    
    # Effect functions
    def robot_voice_effect(self, audio_float):
        t = np.linspace(0, len(audio_float) / self.sample_rate, len(audio_float))
        carrier = np.sin(2 * np.pi * 40 * t)
        modulated = audio_float * (1 + 0.3 * carrier)
        return modulated
    
    def demon_voice_effect(self, audio_float):
        shift_factor = 0.85
        indices = np.arange(0, len(audio_float), shift_factor)
        indices = np.clip(indices, 0, len(audio_float) - 1).astype(int)
        shifted = audio_float[indices]
        
        if len(shifted) < len(audio_float):
            shifted = np.pad(shifted, (0, len(audio_float) - len(shifted)), mode='edge')
        else:
            shifted = shifted[:len(audio_float)]
        
        distorted = np.tanh(shifted * 1.5) * 0.9
        return distorted
    
    def chipmunk_voice_effect(self, audio_float):
        shift_factor = 1.3
        indices = np.arange(0, len(audio_float), shift_factor)
        indices = np.clip(indices, 0, len(audio_float) - 1).astype(int)
        shifted = audio_float[indices]
        
        if len(shifted) < len(audio_float):
            shifted = np.pad(shifted, (0, len(audio_float) - len(shifted)), mode='edge')
        else:
            shifted = shifted[:len(audio_float)]
        
        return shifted * 0.95
    
    def glitch_effect(self, audio_float):
        glitch_probability = 0.01
        glitch_mask = np.random.random(len(audio_float)) < glitch_probability
        glitched = audio_float.copy()
        glitched[glitch_mask] = np.random.uniform(-0.2, 0.2, np.sum(glitch_mask))
        return glitched
    
    def echo_effect(self, audio_float):
        delay_samples = int(0.15 * self.sample_rate)
        echo_strength = 0.3
        
        padded = np.zeros(len(audio_float) + delay_samples)
        padded[:len(audio_float)] = audio_float
        padded[delay_samples:len(audio_float) + delay_samples] += audio_float * echo_strength
        
        return padded[:len(audio_float)]
    
    def reverse_effect(self, audio_float):
        return np.flip(audio_float)
    
    def bitcrush_light(self, audio_float):
        bits = 10
        levels = 2 ** bits
        return np.round(audio_float * levels) / levels
    
    def apply_heavy_distortion(self, audio_float):
        """Apply HEAVY distortion for private conversations"""
        shift_factor = 0.5
        indices = np.arange(0, len(audio_float), shift_factor)
        indices = np.clip(indices, 0, len(audio_float) - 1).astype(int)
        distorted = audio_float[indices]
        
        if len(distorted) < len(audio_float):
            distorted = np.pad(distorted, (0, len(audio_float) - len(distorted)), mode='edge')
        else:
            distorted = distorted[:len(audio_float)]
        
        bits = 3
        levels = 2 ** bits
        distorted = np.round(distorted * levels) / levels
        
        noise = np.random.normal(0, 0.1, len(distorted))
        distorted = distorted + noise
        
        distorted = np.tanh(distorted * 4.0) * 0.6
        
        t = np.linspace(0, len(distorted) / self.sample_rate, len(distorted))
        wobble = np.sin(2 * np.pi * 3 * t) * 0.3
        distorted = distorted * (1 + wobble)
        
        max_val = np.max(np.abs(distorted))
        if max_val > 0:
            distorted = distorted / max_val * 0.8
        
        return distorted
    
    def apply_selected_effects(self, audio_float, selected_effects):
        """Apply selected effects"""
        processed = audio_float.copy()
        
        for effect in selected_effects:
            if effect == "Robot":
                processed = self.robot_voice_effect(processed)
            elif effect == "Demon":
                processed = self.demon_voice_effect(processed)
            elif effect == "Chipmunk":
                processed = self.chipmunk_voice_effect(processed)
            elif effect == "Glitch":
                processed = self.glitch_effect(processed)
            elif effect == "Echo":
                processed = self.echo_effect(processed)
            elif effect == "Reverse":
                processed = self.reverse_effect(processed)
            elif effect == "Bitcrush":
                processed = self.bitcrush_light(processed)
        
        max_val = np.max(np.abs(processed))
        if max_val > 0:
            processed = processed / max_val * 0.90
        
        return processed
    
    def process_audio(self, audio_data, selected_effects, wake_segments):
        """Process audio with effects"""
        audio_float = self.audio_to_float(audio_data)
        processed_audio = np.zeros_like(audio_float)
        
        wake_mask = np.zeros(len(audio_float), dtype=bool)
        for start, end in wake_segments:
            wake_mask[start:end] = True
        
        chunk_size = self.sample_rate
        for i in range(0, len(audio_float), chunk_size):
            end_idx = min(i + chunk_size, len(audio_float))
            chunk = audio_float[i:end_idx]
            mask_chunk = wake_mask[i:end_idx]
            
            distorted_chunk = self.apply_heavy_distortion(chunk)
            effects_chunk = self.apply_selected_effects(chunk, selected_effects)
            
            processed_audio[i:end_idx] = np.where(mask_chunk, effects_chunk, distorted_chunk)
        
        return self.float_to_audio(processed_audio)
    
    def create_visualization(self, original_data, processed_data, wake_segments):
        """Create visualization"""
        original_float = self.audio_to_float(original_data)
        processed_float = self.audio_to_float(processed_data)
        time_array = np.linspace(0, len(original_float) / self.sample_rate, len(original_float))
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Audio Analysis - Original vs Processed', fontsize=14, fontweight='bold')
        
        # Waveform
        ax1 = axes[0, 0]
        ax1.plot(time_array, original_float, label='Original', alpha=0.7, linewidth=0.5, color='blue')
        ax1.plot(time_array, processed_float, label='Processed', alpha=0.7, linewidth=0.5, color='red')
        
        if wake_segments:
            for start, end in wake_segments:
                start_time = start / self.sample_rate
                end_time = end / self.sample_rate
                ax1.axvspan(start_time, end_time, alpha=0.2, color='green')
        
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Amplitude')
        ax1.set_title('Waveform Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Spectrogram Original
        ax2 = axes[0, 1]
        frequencies, times, spectrogram = scipy_signal.spectrogram(
            original_float, fs=self.sample_rate, nperseg=1024, noverlap=512
        )
        spec_db = 10 * np.log10(spectrogram + 1e-10)
        im2 = ax2.pcolormesh(times, frequencies, spec_db, shading='gouraud', cmap='viridis')
        ax2.set_ylabel('Frequency (Hz)')
        ax2.set_xlabel('Time (seconds)')
        ax2.set_title('Spectrogram - Original')
        ax2.set_ylim(0, 8000)
        plt.colorbar(im2, ax=ax2)
        
        # Spectrogram Processed
        ax3 = axes[1, 0]
        frequencies, times, spectrogram = scipy_signal.spectrogram(
            processed_float, fs=self.sample_rate, nperseg=1024, noverlap=512
        )
        spec_db = 10 * np.log10(spectrogram + 1e-10)
        im3 = ax3.pcolormesh(times, frequencies, spec_db, shading='gouraud', cmap='plasma')
        ax3.set_ylabel('Frequency (Hz)')
        ax3.set_xlabel('Time (seconds)')
        ax3.set_title('Spectrogram - Processed')
        ax3.set_ylim(0, 8000)
        plt.colorbar(im3, ax=ax3)
        
        # Energy over time
        ax4 = axes[1, 1]
        window_size = int(0.1 * self.sample_rate)
        original_energy = []
        processed_energy = []
        energy_times = []
        
        for i in range(0, len(original_float) - window_size, window_size // 2):
            window_orig = original_float[i:i+window_size]
            window_proc = processed_float[i:i+window_size]
            original_energy.append(np.sqrt(np.mean(window_orig ** 2)))
            processed_energy.append(np.sqrt(np.mean(window_proc ** 2)))
            energy_times.append(i / self.sample_rate)
        
        ax4.plot(energy_times, original_energy, label='Original', alpha=0.7, color='blue')
        ax4.plot(energy_times, processed_energy, label='Processed', alpha=0.7, color='red')
        ax4.set_xlabel('Time (seconds)')
        ax4.set_ylabel('Energy (RMS)')
        ax4.set_title('Energy Over Time')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def cleanup(self):
        """Clean up resources"""
        self.p.terminate()

def save_wav(audio_data, sample_rate, channels=1):
    """Save audio to WAV bytes"""
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data.tobytes())
    buffer.seek(0)
    return buffer

def main():
    st.set_page_config(page_title="Voice Privacy Processor", layout="wide", page_icon="🎤")
    
    st.title("🎤 Voice Privacy Processor")
    st.markdown("### Real-time audio recording with privacy protection!")
    
    # Initialize processor
    if 'processor' not in st.session_state:
        st.session_state.processor = StreamlitVoicePrivacyProcessor()
    
    processor = st.session_state.processor
    
    # Sidebar settings
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Audio device selection
        devices = processor.get_audio_devices()
        device_names = [d['name'] for d in devices]
        device_indices = [d['index'] for d in devices]
        
        if device_names:
            selected_device_name = st.selectbox(
                "Input Device:",
                options=device_names,
                index=0
            )
            selected_device_index = device_indices[device_names.index(selected_device_name)]
        else:
            st.error("No input devices found!")
            selected_device_index = None
        
        st.markdown("---")
        
        sensitivity = st.select_slider(
            "Voice Detection Sensitivity",
            options=["Low", "Medium", "High"],
            value="Medium"
        )
        
        if sensitivity == "Low":
            processor.energy_threshold = 300
        elif sensitivity == "High":
            processor.energy_threshold = 700
        else:
            processor.energy_threshold = 500
        
        st.markdown("---")
        st.header("🎭 Wake Word Effects")
        st.markdown("*Select effects for wake words + commands*")
        
        effects = st.multiselect(
            "Choose effects (clear & intelligible):",
            ["Robot", "Demon", "Chipmunk", "Glitch", "Echo", "Reverse", "Bitcrush"],
            default=[]
        )
        
        st.markdown("---")
        st.info("""
        **Workflow:**
        1. 🎙️ Record audio
        2. 🔍 Detect speech segments
        3. ✋ Select wake word segments
        4. 🎬 Process audio
        5. 🔊 Listen & download
        """)
    
    # Step 1: Record Audio
    st.header("🎙️ Step 1: Record Audio")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        duration = st.slider("Recording duration (seconds):", min_value=5, max_value=120, value=30, step=5)
    
    with col2:
        st.markdown("###")
        record_button = st.button("🔴 Start Recording", type="primary", use_container_width=True)
    
    if record_button:
        if selected_device_index is not None:
            st.info(f"🎤 Recording for {duration} seconds... Speak now!")
            
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def update_progress(progress):
                progress_bar.progress(progress)
                status_text.text(f"Recording: {progress*100:.0f}%")
            
            # Record audio
            frames = processor.record_audio(duration, selected_device_index, update_progress)
            
            if frames:
                audio_data = processor.frames_to_audio(frames)
                st.session_state.audio_data = audio_data
                st.session_state.sample_rate = processor.sample_rate
                
                progress_bar.empty()
                status_text.empty()
                st.success("✅ Recording complete!")
                
                # Save to temporary WAV for playback
                wav_buffer = save_wav(audio_data, processor.sample_rate)
                st.audio(wav_buffer, format="audio/wav")
                
                # Clear downstream data
                if 'speech_segments' in st.session_state:
                    del st.session_state.speech_segments
                if 'wake_segments' in st.session_state:
                    del st.session_state.wake_segments
                if 'processed_audio' in st.session_state:
                    del st.session_state.processed_audio
        else:
            st.error("No input device selected!")
    
    # Step 2: Detect Speech Segments
    if 'audio_data' in st.session_state:
        st.markdown("---")
        st.header("🔍 Step 2: Detect Speech Segments")
        
        if st.button("🔎 Analyze Audio & Detect Speech", type="primary", use_container_width=True):
            with st.spinner("Analyzing audio..."):
                audio_data = st.session_state.audio_data
                speech_segments = processor.detect_speech_segments(audio_data)
                st.session_state.speech_segments = speech_segments
            
            st.success(f"✅ Found {len(speech_segments)} speech segments!")
    
    # Step 3: Manual Segment Selection
    if 'speech_segments' in st.session_state:
        st.markdown("---")
        st.header("✋ Step 3: Select Wake Word Segments")
        
        speech_segments = st.session_state.speech_segments
        
        if len(speech_segments) == 0:
            st.warning("⚠️ No speech segments detected. Try adjusting sensitivity in the sidebar.")
        else:
            st.markdown("**📊 Detected Speech Segments:**")
            st.markdown("*Select the segments that contain wake words + commands*")
            
            # Create a visual table with segment info
            segment_data = []
            for i, (start, end) in enumerate(speech_segments):
                segment_audio = st.session_state.audio_data[start:end]
                energy = processor.calculate_energy(segment_audio)
                duration = (end - start) / processor.sample_rate
                start_time = start / processor.sample_rate
                end_time = end / processor.sample_rate
                
                # Determine if likely wake word
                is_candidate = (0.3 < duration < 5.0 and energy > processor.energy_threshold * 0.8)
                
                segment_data.append({
                    "Segment": i + 1,
                    "Start (s)": f"{start_time:.2f}",
                    "End (s)": f"{end_time:.2f}",
                    "Duration (s)": f"{duration:.2f}",
                    "Energy": f"{energy:.0f}",
                    "Likely Wake Word": "⭐ Yes" if is_candidate else "No"
                })
            
            # Display as dataframe
            import pandas as pd
            df = pd.DataFrame(segment_data)
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            st.markdown("---")
            st.markdown("**🎯 Select Wake Word Segments:**")
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # Multi-select for segments
                default_selections = [i+1 for i, (start, end) in enumerate(speech_segments) 
                        if (0.3 < (end-start)/processor.sample_rate < 5.0 and 
                            processor.calculate_energy(st.session_state.audio_data[start:end]) > processor.energy_threshold * 0.8)]
                
                selected_segments = st.multiselect(
                    "Choose segment numbers that contain wake words + commands:",
                    options=list(range(1, len(speech_segments) + 1)),
                    default=default_selections,
                    help="Segments marked with ⭐ are likely wake words"
                )
            
            with col2:
                if st.button("🔄 Select All", use_container_width=True):
                    selected_segments = list(range(1, len(speech_segments) + 1))
                
                if st.button("❌ Clear All", use_container_width=True):
                    selected_segments = []
            
            if selected_segments:
                # Convert selected segment numbers to actual segments
                wake_segments = [speech_segments[i-1] for i in selected_segments]
                st.session_state.wake_segments = wake_segments
                
                st.success(f"✅ Selected {len(selected_segments)} segment(s) as wake words")
                
                # Show selected segments
                st.markdown("**Selected Wake Word Segments:**")
                for seg_num in selected_segments:
                    start, end = speech_segments[seg_num - 1]
                    start_time = start / processor.sample_rate
                    end_time = end / processor.sample_rate
                    duration = (end - start) / processor.sample_rate
                    st.text(f"  • Segment {seg_num}: {start_time:.2f}s - {end_time:.2f}s (Duration: {duration:.2f}s)")
            else:
                st.warning("⚠️ Please select at least one segment containing wake words")
    
    # Step 4: Process Audio
    if 'wake_segments' in st.session_state and st.session_state.wake_segments:
        st.markdown("---")
        st.header("🎬 Step 4: Process Audio")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Processing Summary:**")
            st.info(f"""
            - **Wake word segments:** {len(st.session_state.wake_segments)}
            - **Effects:** {', '.join(effects) if effects else 'None (clear voice)'}
            - **Private parts:** Heavy distortion
            """)
        
        with col2:
            if st.button("🚀 Process Audio Now", type="primary", use_container_width=True):
                with st.spinner("Processing audio with effects..."):
                    audio_data = st.session_state.audio_data
                    wake_segments = st.session_state.wake_segments
                    
                    # Process
                    processed_audio = processor.process_audio(
                        audio_data, effects, wake_segments
                    )
                    
                    st.session_state.processed_audio = processed_audio
                
                st.success("✅ Processing complete!")
                st.balloons()
    
    # Step 5: Results
    if 'processed_audio' in st.session_state:
        st.markdown("---")
        st.header("📤 Step 5: Processed Audio Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎧 Listen to Processed Audio")
            processed_wav = save_wav(
                st.session_state.processed_audio,
                processor.sample_rate
            )
            st.audio(processed_wav, format="audio/wav")
        
        with col2:
            st.subheader("💾 Download")
            st.download_button(
                label="📥 Download Processed Audio",
                data=processed_wav,
                file_name="privacy_protected.wav",
                mime="audio/wav",
                use_container_width=True
            )
            
            # Show statistics
            st.metric("Speech Segments", len(st.session_state.speech_segments))
            st.metric("Wake Word Segments", len(st.session_state.wake_segments))
        
        # Visualization
        st.markdown("---")
        st.header("📊 Audio Analysis & Visualization")
        
        if st.button("📈 Generate Visualization", use_container_width=True):
            with st.spinner("Creating visualizations..."):
                fig = processor.create_visualization(
                    st.session_state.audio_data,
                    st.session_state.processed_audio,
                    st.session_state.wake_segments
                )
                st.pyplot(fig)

if __name__ == "__main__":
    main()