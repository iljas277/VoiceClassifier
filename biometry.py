import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from data_converter import load_train_data, load_test_data
from model import BiometryModel, write_answers, train_func, predict_test


def main():
    torch.manual_seed(42)

    print("[main] Loading train data...")
    X, y = load_train_data("./train")

    rng = np.random.default_rng(42)
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    rng.shuffle(idx0)
    rng.shuffle(idx1)

    val_frac = 0.2
    n0_val = int(len(idx0) * val_frac)
    n1_val = int(len(idx1) * val_frac)

    val_idx = np.concatenate([idx0[:n0_val], idx1[:n1_val]])
    train_idx = np.concatenate([idx0[n0_val:], idx1[n1_val:]])
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    feat_mean = X_train.mean(axis=0, keepdims=True)
    feat_std = X_train.std(axis=0, keepdims=True) + 1e-8
    X_train = (X_train - feat_mean) / feat_std
    X_val = (X_val - feat_mean) / feat_std

    train_ds = torch.utils.data.TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    val_ds = torch.utils.data.TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long),
    )

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=64, shuffle=False)

    model = BiometryModel(input_size=X.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
    loss_func = nn.CrossEntropyLoss()

    print("[main] Training started...")
    train_func(
        model=model,
        num_epochs=10,
        dataloaders={"train": train_loader, "valid": val_loader},
        optimizer=optimizer,
        loss_func=loss_func,
    )
    print("[main] Training finished.")

    print("[main] Loading test data and running inference...")
    X_test, test_ids = load_test_data("./test")
    X_test = (X_test - feat_mean) / feat_std
    test_preds = predict_test(model, X_test)
    print(f"[main] Inference finished. Got {len(test_preds)} predictions.")

    write_answers(test_ids, test_preds, out_path="answers.tsv")


if __name__ == "__main__":
    main()