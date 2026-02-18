import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np

class BiometryModel(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(512, 256)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(p=0.3)
        self.fc3 = nn.Linear(256, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.drop2(x)
        x = self.fc3(x)
        return x


def fit_one_epoch(model, train_dataloader, optimizer, loss_func) -> float:
    model.train()
    correct = 0
    total = 0

    for X_batch, y_batch in tqdm(train_dataloader):
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = loss_func(logits, y_batch)
        loss.backward()
        optimizer.step()

        preds = logits.argmax(dim=-1)
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)

    return correct / max(1, total)


def eval_one_epoch(model, val_dataloader) -> float:
    model.eval()
    correct = 0
    total = 0

    for X_batch, y_batch in tqdm(val_dataloader):
        with torch.no_grad():
            logits = model(X_batch)
            preds = logits.argmax(dim=-1)
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)

    return correct / max(1, total)


def train_func(model, num_epochs, dataloaders, optimizer, loss_func):
    best_val = -1.0
    best_state = None

    for epoch in range(num_epochs):
        print(f"\nEpoch: {epoch + 1}")

        acc_train = fit_one_epoch(model, dataloaders["train"], optimizer, loss_func)
        print(f"Accuracy_train: {acc_train:.6f}")

        acc_val = eval_one_epoch(model, dataloaders["valid"])
        print(f"Accuracy_val: {acc_val:.6f}")

        if acc_val > best_val:
            best_val = acc_val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"[main] New best val: {best_val:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"[main] Restored best model: {best_val:.6f}")


def predict_test(model, X_test: np.ndarray, batch_size: int = 64) -> list[int]:
    model.eval()
    test_dataset = torch.utils.data.TensorDataset(torch.tensor(X_test, dtype=torch.float32))
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    preds_all = []
    for (X_batch,) in test_dataloader:
        with torch.no_grad():
            logits = model(X_batch)
            preds = logits.argmax(dim=-1).cpu().numpy().tolist()
        preds_all.extend(preds)
    return preds_all


def write_answers(ids: list[str], preds: list[int], out_path: str = "answers.tsv") -> None:
    print(f"[out] Writing {len(ids)} predictions to: {out_path}")
    with open(out_path, "w") as f:
        for sample_id, gender in zip(ids, preds):
            f.write(f"{sample_id}\t{gender}\n")
    print("[out] Done.")