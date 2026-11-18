
import numpy as np
import torch
import torch.nn as nn

from pathlib import Path
from torch.utils.data import Dataset, DataLoader

# ============================================
# 1. PATHS AND CONFIGURATION
# ============================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = BASE_DIR / "data" / "training" / "training_data.npz"

MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILE = MODEL_DIR / "best_lstm_model.pth"

BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001

INPUT_SIZE = 12
HIDDEN_SIZE_1 = 64
HIDDEN_SIZE_2 = 32
OUTPUT_SIZE = 1

DROPOUT = 0.2

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)

print("Using device:", DEVICE)

# ============================================
# 2. LOAD TRAINING DATA
# ============================================

print("\nLoading prepared training data...")

data = np.load(DATA_FILE)

X_train = torch.tensor(
    data["X_train"], dtype=torch.float32
)

y_train = torch.tensor(
    data["y_train"], dtype=torch.float32
)

X_val = torch.tensor(
    data["X_val"], dtype=torch.float32
)

y_val = torch.tensor(
    data["y_val"], dtype=torch.float32
)

print("Training:", X_train.shape, y_train.shape)
print("Validation:", X_val.shape, y_val.shape)

# ============================================
# 3. DATASET AND DATALOADERS
# ============================================

class StockDataset(Dataset):

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

train_dataset = StockDataset(X_train, y_train)
val_dataset = StockDataset(X_val, y_val)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ============================================
# 4. DEFINE LSTM MODEL
# ============================================

class StockLSTM(nn.Module):

    def __init__(self):
        super().__init__()

        self.lstm1 = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE_1,
            batch_first=True
        )

        self.dropout = nn.Dropout(DROPOUT)

        self.lstm2 = nn.LSTM(
            input_size=HIDDEN_SIZE_1,
            hidden_size=HIDDEN_SIZE_2,
            batch_first=True
        )

        self.fc = nn.Linear(
            HIDDEN_SIZE_2,
            OUTPUT_SIZE
        )

    def forward(self, x):

        x, _ = self.lstm1(x)

        x = self.dropout(x)

        x, _ = self.lstm2(x)

        x = x[:, -1, :]

        x = self.fc(x)

        return x

model = StockLSTM().to(DEVICE)

print("\nModel architecture:")
print(model)

# ============================================
# 5. LOSS AND OPTIMIZER
# ============================================

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

# ============================================
# 6. TRAINING LOOP
# ============================================

best_val_loss = float("inf")

print("\nStarting LSTM training...")

for epoch in range(EPOCHS):

    # Training
    model.train()

    train_loss = 0.0

    for X_batch, y_batch in train_loader:

        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        optimizer.zero_grad()

        predictions = model(X_batch)

        loss = criterion(
            predictions,
            y_batch
        )

        loss.backward()

        optimizer.step()

        train_loss += loss.item() * len(X_batch)

    train_loss /= len(train_dataset)

    # Validation
    model.eval()

    val_loss = 0.0

    with torch.no_grad():

        for X_batch, y_batch in val_loader:

            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            predictions = model(X_batch)

            loss = criterion(
                predictions,
                y_batch
            )

            val_loss += loss.item() * len(X_batch)

    val_loss /= len(val_dataset)

    # Save best model
    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "input_size": INPUT_SIZE,
                "hidden_size_1": HIDDEN_SIZE_1,
                "hidden_size_2": HIDDEN_SIZE_2,
                "dropout": DROPOUT,
                "best_val_loss": best_val_loss
            },
            MODEL_FILE
        )

    if (epoch + 1) % 10 == 0 or epoch == 0:

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Train Loss: {train_loss:.6f} "
            f"Validation Loss: {val_loss:.6f}"
        )

# ============================================
# 7. TRAINING COMPLETED
# ============================================

print("\n" + "=" * 50)
print("LSTM TRAINING COMPLETED")
print("=" * 50)

print("Best validation loss:", best_val_loss)
print("Saved model:", MODEL_FILE)
print("Device:", DEVICE)