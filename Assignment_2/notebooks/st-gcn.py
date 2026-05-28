"""
# Assignment 2 — MEG Brain-State Classification with **ST-GCN**


An ST-GCN ( Spatio-Temporal Graph Convolutional Network) model for 4-class MEG decoding
(`rest`, `task_motor`, `task_story_math`, `task_working_memory`), covering both the
**intra-subject** and **cross-subject** settings.

**Why ST-GCN? — task (a): model choice & justification.**

ST-GCN is classicly used in analyzing time-series in data with interconneced components,
such as in skeleton-based action recognition. By computing a connectivity matrix,
the architecture applies spatial graphj convolutions to pass information along synchronized brain regions.
This is particularly useful for MEG data, which is inherently spatially distributed. Furthermore, the model
is trained on a variety of tasks, including motor and working memory tasks, which are known to be strongly connected
to the prefrontal cortex, the primary target of this study. ST-GCN is expected to provide improved generalization in
cross-subject settings due to its ability to learn spatially distributed patterns of brain activity.

"""

import h5py
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

print("starting ST-GCN...")

# --- Configuration ---
DATA_ROOT = Path(__file__).resolve().parents[1] / "data"

INTRA_TRAIN = DATA_ROOT / "Intra" / "train"
INTRA_TEST = DATA_ROOT / "Intra" / "test"

CROSS_TRAIN = DATA_ROOT / "Cross" / "train"
CROSS_TEST1 = DATA_ROOT / "Cross" / "test1"
CROSS_TEST2 = DATA_ROOT / "Cross" / "test2"
CROSS_TEST3 = DATA_ROOT / "Cross" / "test3"

TASK_LABELS = {
    "rest": 0,
    "task_motor": 1,
    "task_story_math": 2,
    "task_working_memory": 3,
}
LABEL_NAMES = {v: k for k, v in TASK_LABELS.items()}
SAMPLE_RATE = 2034  # Hz
EPOCHS = 5

# --- Helper Functions ---
def get_dataset_name(filepath):
    name = Path(filepath).stem  # e.g. rest_105923_1
    parts = name.split("_")[:-1]  # drop trailing chunk number
    return "_".join(parts)


def parse_label(filename):
    lowered = filename.lower()
    if "rest" in lowered:
        return "rest"
    if "motor" in lowered:
        return "task_motor"
    if "math" in lowered or "story" in lowered:
        return "task_story_math"
    if "memory" in lowered or "working" in lowered:
        return "task_working_memory"
    return "unknown"

# --- 1. Dataset exploration ---
def scan_folder(folder_path, folder_name):
    """Scans an H5 folder and extracts file-level metadata."""
    records = []
    file_paths = list(folder_path.glob("*.h5"))

    for path in file_paths:
        label_str = parse_label(path.name)
        with h5py.File(path, "r") as f:
            d_name = get_dataset_name(path)
            try:
                matrix = f.get(d_name)[()]
                n_sensors, n_time_steps = matrix.shape
                # Calculate basic statistics on raw data to check order of magnitude
                raw_mean = np.mean(matrix)
                raw_std = np.std(matrix)
                raw_min = np.min(matrix)
                raw_max = np.max(matrix)
            except Exception as e:
                print(f"Error reading {path.name}: {e}")
                continue

        records.append(
            {
                "folder": folder_name,
                "filename": path.name,
                "task": label_str,
                "sensors": n_sensors,
                "time_steps": n_time_steps,
                "duration_sec": n_time_steps / SAMPLE_RATE,
                "mean": raw_mean,
                "std": raw_std,
                "min": raw_min,
                "max": raw_max,
            }
        )
    return records


# --- 1. Dataset exploration ---
print("Scanning directories... This might take a minute if files are large.")
all_metadata = []
splits = {
    "Intra_Train": INTRA_TRAIN,
    "Intra_Test": INTRA_TEST,
    "Cross_Train": CROSS_TRAIN,
    "Cross_Test1": CROSS_TEST1,
    "Cross_Test2": CROSS_TEST2,
    "Cross_Test3": CROSS_TEST3,
}

for name, path in splits.items():
    if path.exists():
        all_metadata.extend(scan_folder(path, name))
    else:
        print(f"⚠️ Directory not found, skipping: {path}")

df = pd.DataFrame(all_metadata)

if df.empty:
    print("❌ No data found. Please double check your paths or place sample files in the directories.")
    exit()

# --- 2. Descriptive Summary Statistics ---
print("\n=== GENERAL DATASET SUMMARY ===")
print(f"Total files discovered: {len(df)}")
print(f"Unique tasks: {df['task'].unique()}")
print("\n--- Sensor and Temporal Check ---")
print(f"Unique sensor counts found: {df['sensors'].unique()} (Expected: [248])")
print(f"Average time steps per file: {df['time_steps'].mean():.2f} (~{df['duration_sec'].mean():.1f} seconds)")

print("\n--- Value Scale Verification ---")
print(f"Global Minimum value observed: {df['min'].min():.2e} Tesla")
print(f"Global Maximum value observed: {df['max'].max():.2e} Tesla")

print("\n=== CLASS DISTRIBUTION ACROSS SPLITS ===")
class_split_matrix = pd.crosstab(df["folder"], df["task"])
print(class_split_matrix)


# --- 3. Dataset Implementation ---
class MEGPipelineDataset(Dataset):
    def __init__(self, folder_path, window_duration=1.0, downsample_factor=4):
        self.folder_path = folder_path
        self.downsample_factor = downsample_factor
        self.window_size = int((SAMPLE_RATE * window_duration) // downsample_factor)
        self.samples = []
        self._load_and_process()

    def _load_and_process(self):
        file_paths = list(self.folder_path.glob("*.h5"))
        for path in file_paths:
            label_str = parse_label(path.name)
            if label_str not in TASK_LABELS: continue
            label_idx = TASK_LABELS[label_str]

            with h5py.File(path, "r") as f:
                d_name = get_dataset_name(path)
                matrix = f[d_name][()]
                matrix = matrix[:, :: self.downsample_factor]

                # Time-wise Z-score normalization
                mean = matrix.mean(axis=1, keepdims=True)
                std = matrix.std(axis=1, keepdims=True) + 1e-8
                matrix = (matrix - mean) / std

                num_time_steps = matrix.shape[1]
                for start in range(0, num_time_steps - self.window_size, self.window_size):
                    chunk = matrix[:, start : start + self.window_size]
                    self.samples.append((chunk, label_idx))

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        matrix, label = self.samples[idx]
        return torch.tensor(matrix, dtype=torch.float32).unsqueeze(0), torch.tensor(label, dtype=torch.long)

# --- 4. ST-GCN Network Architecture ---
class SpatialGraphConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SpatialGraphConv, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        
    def forward(self, x, A):
        x = x.permute(0, 2, 1, 3)
        out = torch.einsum('btnd,nn->btnd', x, A)
        out = out.permute(0, 2, 1, 3)
        return self.linear(out)

class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, temporal_kernel=9, dropout=0.3):
        super(STGCNBlock, self).__init__()
        self.spatial_conv = SpatialGraphConv(in_channels, out_channels)
        self.temporal_conv = nn.Conv2d(
            in_channels=out_channels, out_channels=out_channels,
            kernel_size=(1, temporal_kernel), padding=(0, temporal_kernel // 2)
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(dropout)  # Drop channels out systematically

    def forward(self, x, A):
        x = x.permute(0, 2, 3, 1)
        x = self.relu(self.spatial_conv(x, A))
        x = x.permute(0, 3, 1, 2)
        x = self.temporal_conv(x)
        return self.dropout(self.relu(self.bn(x)))

class MEG_STGCN(nn.Module):
    def __init__(self, num_nodes=248, num_classes=4, hidden_dim=32):
        super(MEG_STGCN, self).__init__()
        self.A = nn.Parameter(torch.randn(num_nodes, num_nodes))
        self.block1 = STGCNBlock(in_channels=1, out_channels=hidden_dim)
        self.block2 = STGCNBlock(in_channels=hidden_dim, out_channels=hidden_dim * 2)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        # Apply element-wise dropout to the adjacency connections to combat overfitting
        A_adj = torch.sigmoid(self.A)
        A_adj = F.dropout(A_adj, p=0.2, training=self.training)
        
        x = self.block1(x, A_adj)
        x = self.block2(x, A_adj)
        x = self.global_pool(x)
        return self.fc(torch.flatten(x, 1))

# --- Core Training Engine ---
def run_epoch(model, loader, criterion, optimizer=None, device="cpu"):
    if optimizer:
        model.train()
    else:
        model.eval()

    running_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(optimizer is not None):
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            if optimizer:
                optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            if optimizer:
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return running_loss / total, correct / total


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Target Compute Device: {device}")

    # 1. Load your original base dataset collections
    full_train_dataset = MEGPipelineDataset(INTRA_TRAIN, window_duration=1.0, downsample_factor=4)
    test_dataset = MEGPipelineDataset(INTRA_TEST, window_duration=1.0, downsample_factor=4)

    # 2. Extract 20% of your training window slices exclusively for validation tuning
    val_size = int(0.20 * len(full_train_dataset))
    train_size = len(full_train_dataset) - val_size
    
    # Using a manual generator seed guarantees your splits don't change between runs
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_train_dataset, [train_size, val_size], generator=generator
    )

    print(f"✔ Dataset structural separation completed:")
    print(f"   - Training chunks:   {len(train_dataset)}")
    print(f"   - Validation chunks: {len(val_dataset)}")
    print(f"   - Testing chunks:    {len(test_dataset)}")


    # 3. Create your loaders (Notice the new val_loader)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    model = MEG_STGCN(num_nodes=248, num_classes=4, hidden_dim=32).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)

    # Track validation metrics alongside training
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    print("\n=== STARTING TRAINING PROCESS ===")
    
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer=None, device=device)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch:02d}/{EPOCHS:02d} -> "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.1f}% || "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.1f}%")

    # Final Isolated Holdout Test
    print("\n=== HYPERPARAMETER TUNING LOCKED: RUNNING FINAL TEST SET ===")
    final_test_loss, final_test_acc = run_epoch(model, test_loader, criterion, optimizer=None, device=device)
    print(f"➔ [FINAL TEST PERFORMANCE] Accuracy: {final_test_acc*100:.2f}% | Loss: {final_test_loss:.4f}")

    # --- Metrics Plotting (Completely Fixed & Synced) ---
    epochs_range = range(1, EPOCHS + 1)
    plt.figure(figsize=(12, 5))

    # Plot Sub-Graph A: Loss Performance
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history['train_loss'], label='Train Loss', color='#3f51b5', linewidth=2, marker='o')
    plt.plot(epochs_range, history['val_loss'], label='Validation Loss', color='#f44336', linewidth=2, linestyle='--', marker='s')
    plt.title('ST-GCN Categorical Cross Entropy Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss Value')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper right')

    # Plot Sub-Graph B: Accuracy Performance
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, [acc * 100 for acc in history['train_acc']], label='Train Accuracy', color='#009688', linewidth=2, marker='o')
    plt.plot(epochs_range, [acc * 100 for acc in history['val_acc']], label='Validation Accuracy', color='#ff9800', linewidth=2, linestyle='--', marker='s')
    plt.title('Task Classification Accuracy Performance')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy Percentage (%)')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='lower right')

    plt.tight_layout()
    
    output_image_path = Path("stgcn_loss_convergence_plot.png")
    plt.savefig(output_image_path, dpi=150)
    print(f"Analysis graph saved to: {output_image_path.resolve()}")
    plt.show()
