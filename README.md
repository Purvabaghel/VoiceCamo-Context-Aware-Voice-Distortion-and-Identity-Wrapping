# 🎙️ VoiceCamo — Real-Time Voice Anonymization using DSP & Streamlit

VoiceCamo is a lightweight, real-time **voice privacy and anonymization system** built using **Digital Signal Processing (DSP)** and **Streamlit**.  
Instead of relying on heavy machine-learning models, this project explores **deterministic signal-domain transformations** that distort identifiable voice features while maintaining intelligibility.

---

## 🚀 Project Overview

Modern voice assistants, call centers, and online communication platforms expose users to privacy risks.  
VoiceCamo aims to solve this by providing **real-time voice anonymization** that works on consumer-grade hardware without GPU requirements.

This project investigates:
- How classical DSP techniques affect speaker identity
- How transformations like clipping, quantization, echo insertion, and pitch shifting degrade identifiable acoustic cues
- Whether real-time anonymization is feasible without ML models

---

## 🎯 Objectives

- Build an interactive **Streamlit-based interface** for recording, uploading, processing, and downloading anonymized audio.
- Implement a set of **signal-domain distortions** that disguise a speaker’s voice.
- Provide waveform and spectral visualizations for analysis.
- Explore trade-offs between privacy, intelligibility, and distortion.

---

## 🔧 Features

### 🔹 **1. Audio Input**
- Record directly in Streamlit  
- Upload `.wav` files  
- Automatic sample-rate handling  

---

### 🔹 **2. DSP-Based Anonymization Methods**

| Technique | Purpose | Effect |
|----------|---------|--------|
| **Clipping** | Reduce amplitude characteristics | Removes fine-grained voice cues |
| **Quantization / Bit-Depth Reduction** | Lower signal precision | Adds digital distortion and privacy |
| **Pitch Shifting** | Modify pitch without changing speed | Alters perceived identity |
| **Time Shifting** | Shift waveform slightly | Breaks alignment cues used in recognition |
| **Echo Insertion** | Add delayed signal copies | Smears temporal structure |
| **Resampling** | Change playback speed indirectly | Alters formants and timbre |

Each method modifies recognizable speaker signatures such as:
- formant structure  
- harmonic distribution  
- temporal cues  
- pitch and prosody  

---

### 🔹 **3. Visual Analysis Tools**

The interface displays:

#### 📈 **Waveform Plots**
Shows amplitude changes before and after anonymization.

#### 🎛️ **FFT (Spectral) Plots**
Displays frequency-domain changes  
(e.g., pitch shifts, harmonics distortion).

#### ▶️ **Audio Playback**
Listen to:
- Original audio  
- Anonymized output  

---

## 🧠 Research Aspect

VoiceCamo includes a mini-study of manually crafted DSP methods as alternatives to ML-based anonymizers.

### Theoretical exploration includes:
- Speaker-identifying acoustic cues  
- Signal-domain transformation math  
- Privacy vs. audio quality trade-offs  
- Real-time constraints on distortion techniques  

This provides insight into whether traditional DSP can meaningfully degrade speaker identity.

---

## 🗂️ Project Structure (Recommended)

```plaintext
VoiceCamo/
│── README.md
│── app.py               # Streamlit UI
│── processor.py         # All DSP transformations
│── utils.py             # Helper functions
│── assets/
│   ├── plots/
│   ├── samples/
│── requirements.txt
│── screenshots/
```
## 🏃 How It Works — Workflow Summary

1. **User records or uploads audio**
2. **The system loads waveform** → converts it to **NumPy**
3. **User selects an anonymization method**
4. **DSP transformation is applied in real time**
5. **Results are plotted and previewed**
6. **User downloads anonymized audio**

---

## 🎤 Why DSP Instead of ML?

- ✔ **No GPU required**  
- ✔ **Real-time performance**  
- ✔ **Fully offline — privacy by design**  
- ✔ **Transparent transform (no black-box model)**  

While ML-based anonymizers are powerful, they are often computationally heavy.  
**VoiceCamo demonstrates that classical DSP techniques can still anonymize voices effectively for many real-world scenarios.**

---

## 📌 Applications

- Privacy-preserving voice collection  
- Anonymized interviews  
- Secure audio sharing  
- Research on speaker recognition resistance  
- Classroom demonstrations of DSP concepts  

---

## 📷 Screenshots (Optional)

_Add waveform and FFT screenshots here._

---

## 📄 License

**MIT License** (modify as needed)

---

## 🤝 Contributing

Pull requests and suggestions are welcome!  
Feel free to open an issue to report bugs or request features.

---

## 🙌 Acknowledgments

This project is part of a study on **voice privacy, DSP, and real-time anonymization techniques**.  
Inspired by research on **speaker identification** and **privacy-preserving audio systems**.

## ✍️ Authors

**VoiceCamo was developed by:**

- **Purva Baghel**  
- **Archita Kulkarni**  
- **Poorva Pohekar**

