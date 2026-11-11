# AI-Based Satellite Signal Anomaly Detector (simulation)

Detects **jamming** and **spoofing** in satellite (GPS L1) radio signals using the signal's frequency
"fingerprint" and a Random Forest classifier, shown on a live Streamlit dashboard.

This repository is the **software simulation**. No hardware is needed.

![Spectrograms of the three signal types](docs/spectrogram_comparison.png)

## How it works

| Step | What happens |
|------|--------------|
| 1. Signal | 1,024 complex IQ samples (Normal, Jamming or Spoofing) |
| 2. FFT | 256-point FFT, keep the first 128 bins, divide by the largest bin |
| 3. Scale | StandardScaler |
| 4. Classify | Random Forest, 200 trees, max depth 20 |
| 5. Score | Anomaly score = 1 − P(Normal). Above 0.35 raises an alert |
| 6. Dashboard | Waveform, spectrogram, FFT, constellation, confidence, history, event log |

The three simulated signals:
- **Normal:** a clean 1 MHz carrier plus a little noise.
- **Jamming:** Normal plus a strong extra tone at 2.5 MHz.
- **Spoofing:** a stronger carrier with drifting phase.

## Run it

```bash
git clone https://github.com/<your-username>/satellite-anomaly-detector.git
cd satellite-anomaly-detector
python -m venv venv
venv\Scripts\activate          # Windows   (Linux/macOS: source venv/bin/activate)
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501, choose a signal mode in the sidebar and press **Start**.

## Retrain the model

```bash
python train_model.py
```

This rebuilds 6,000 noisy signals from `data/`, trains the model and rewrites `models/`.

## Project layout

```
app.py              Streamlit dashboard
train_model.py      dataset + training script
data/               base IQ signals (.npy)
models/             rf_model.pkl, scaler.pkl, confusion_matrix.npy
docs/               figures
```

## Results and limits

- 100% accuracy on 1,200 held-out simulated signals.
- The signals are synthetic and very different from each other, so this shows the pipeline works.
  It is **not** a claim about real-world accuracy.
- Models were saved with scikit-learn 1.8.0 (pinned in `requirements.txt`).

## Hardware version

The real-hardware version (RTL-SDR V3 + active GPS antenna, retrained on real captures) is a separate step.
Real jamming and spoofing are not transmitted (illegal); attacks are injected in software on real captures.

## Team

JNN College of Engineering, Shivamogga, Dept. of E&TE, 2026-2027.
Prakruthi M, Rekha K M, Shannu, Shreegowri B R. Guide: Dr. Ushadevi M B.

## License

MIT
