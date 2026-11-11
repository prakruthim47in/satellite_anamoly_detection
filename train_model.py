"""Rebuild the dataset and retrain the Random Forest from the three base signals in data/.

    python train_model.py              # writes to models/
    python train_model.py --out /tmp/m # write somewhere else

6,000 signals (3,000 Normal, 1,500 Jamming, 1,500 Spoofing) are made by adding random
Gaussian noise to the base signals. Features: 256-point FFT, first 128 bins, divided by the max.
"""
import argparse, os, pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="models")
ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args()
rng = np.random.default_rng(a.seed)
base = os.path.dirname(os.path.abspath(__file__))
sig = [np.load(os.path.join(base, "data", f"{n}_signal.npy")) for n in ("normal", "jamming", "spoofing")]
counts = [3000, 1500, 1500]

def featurize(iq):
    f = np.abs(np.fft.fft(iq, n=256))[:128]
    return (f / (f.max() + 1e-9)).astype(np.float32)

X, y = [], []
for label, (s, n) in enumerate(zip(sig, counts)):
    for _ in range(n):
        sd = rng.uniform(0.05, 0.25)
        noisy = s + sd * (rng.standard_normal(len(s)) + 1j * rng.standard_normal(len(s)))
        X.append(featurize(noisy)); y.append(label)
X, y = np.array(X), np.array(y)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
sc = StandardScaler().fit(Xtr)
rf = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1).fit(sc.transform(Xtr), ytr)
pred = rf.predict(sc.transform(Xte)); cm = confusion_matrix(yte, pred)
print(cm); print(classification_report(yte, pred, target_names=["Normal", "Jamming", "Spoofing"], digits=3))
os.makedirs(a.out, exist_ok=True)
pickle.dump(rf, open(os.path.join(a.out, "rf_model.pkl"), "wb"))
pickle.dump(sc, open(os.path.join(a.out, "scaler.pkl"), "wb"))
np.save(os.path.join(a.out, "confusion_matrix.npy"), cm)
print("saved to", a.out)
