import numpy as np
from pathlib import Path
from librosa.feature import melspectrogram
from librosa import load, power_to_db

def wav_to_feat(wav_path: Path) -> np.ndarray:
    y, sr = load(str(wav_path), sr=None)
    S = melspectrogram(
        y=y,
        sr=sr,
        n_fft=2048,
        hop_length=512,
        n_mels=128,
        power=2.0,
    )
    S_db = power_to_db(S, ref=np.max)
    mean = float(np.mean(S_db))
    std = float(np.std(S_db)) + 1e-8
    S_db = (S_db - mean) / std

    feat = np.concatenate(
        [S_db.mean(axis=1), S_db.std(axis=1), S_db.max(axis=1)],
        axis=0,
    )
    return feat.astype(np.float32)

def load_train_data(train_dir: str) -> tuple[np.ndarray, np.ndarray]:
    train_path = Path(train_dir)
    targets_path = train_path / "targets.tsv"
    print(f"[train] Reading labels from: {targets_path}")

    X_list = []
    y_list = []

    with open(targets_path, "r") as f:
        for line in f:
            sample_id, gender = line.split()
            wav_path = train_path / f"{sample_id}.wav"
            X_list.append(wav_to_feat(wav_path))
            y_list.append(int(gender))

    X = np.stack(X_list)
    y = np.array(y_list, dtype=np.int64)
    counts = np.bincount(y, minlength=2)
    print(f"[train] Loaded: X={X.shape}, y={y.shape}, class_counts={counts.tolist()}")
    return X, y


def load_test_data(test_dir: str) -> tuple[np.ndarray, list[str]]:
    test_path = Path(test_dir)
    wav_paths = sorted(test_path.glob("*.wav"), key=lambda p: p.stem)
    print(f"[test] Found {len(wav_paths)} wav files in: {test_path}")

    ids = []
    X_list = []
    for wav_path in wav_paths:
        ids.append(wav_path.stem)
        X_list.append(wav_to_feat(wav_path))

    X = np.stack(X_list)
    print(f"[test] Loaded: X_test={X.shape}")
    return X, ids