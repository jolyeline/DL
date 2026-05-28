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

print("ST-GCN notebook loaded successfully. Starting data exploration...")

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
        """
        Args:
            folder_path: Path object (e.g. INTRA_TRAIN)
            window_duration: Size of time chunks in seconds
            downsample_factor: Integer downsampling rate (reduces 2034Hz)
        """
        self.folder_path = folder_path
        self.downsample_factor = downsample_factor

        # Calculate exactly how many time steps make up our window after downsampling
        self.window_size = int((SAMPLE_RATE * window_duration) // downsample_factor)
        self.samples = []
        self._load_and_process()

    def _load_and_process(self):
        file_paths = list(self.folder_path.glob("*.h5"))
        for path in file_paths:
            label_str = parse_label(path.name)
            if label_str not in TASK_LABELS:
                continue
            label_idx = TASK_LABELS[label_str]

            with h5py.File(path, "r") as f:
                d_name = get_dataset_name(path)
                matrix = f[d_name][()]  # Shape: (248, 35624)

                # 1. Downsample to optimize training speeds
                matrix = matrix[:, :: self.downsample_factor]

                # 2. Time-wise Z-score normalization (Crucial step based on your EDA plot!)
                mean = matrix.mean(axis=1, keepdims=True)
                std = matrix.std(axis=1, keepdims=True) + 1e-8
                matrix = (matrix - mean) / std

                # 3. Sliding window slice engine
                num_time_steps = matrix.shape[1]
                for start in range(0, num_time_steps - self.window_size, self.window_size):
                    chunk = matrix[:, start : start + self.window_size]
                    self.samples.append((chunk, label_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        matrix, label = self.samples[idx]
        # Tensor Shape: (Channels=1, Nodes=248, Time=window_size)
        return torch.tensor(matrix, dtype=torch.float32).unsqueeze(0), torch.tensor(label, dtype=torch.long)


# --- 4. ST-GCN Model Architecture ---
class SpatialGraphConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SpatialGraphConv, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        
    def forward(self, x, A):
        # x shape: (B, N, T, C)
        x = x.permute(0, 2, 1, 3)
        out = torch.einsum('btnd,nn->btnd', x, A)
        out = out.permute(0, 2, 1, 3)
        return self.linear(out)

class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, temporal_kernel=9):
        super(STGCNBlock, self).__init__()
        self.spatial_conv = SpatialGraphConv(in_channels, out_channels)
        self.temporal_conv = nn.Conv2d(
            in_channels=out_channels, out_channels=out_channels,
            kernel_size=(1, temporal_kernel), padding=(0, temporal_kernel // 2)
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x, A):
        x = x.permute(0, 2, 3, 1)  # to (B, N, T, C)
        x = self.relu(self.spatial_conv(x, A))
        x = x.permute(0, 3, 1, 2)  # back to (B, C, N, T)
        x = self.temporal_conv(x)
        return self.relu(self.bn(x))

class MEG_STGCN(nn.Module):
    def __init__(self, num_nodes=248, num_classes=4, hidden_dim=32):
        super(MEG_STGCN, self).__init__()
        self.A = nn.Parameter(torch.randn(num_nodes, num_nodes))
        self.block1 = STGCNBlock(in_channels=1, out_channels=hidden_dim)
        self.block2 = STGCNBlock(in_channels=hidden_dim, out_channels=hidden_dim * 2)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        A_adj = torch.sigmoid(self.A)  # Bound matrix parameters strictly between 0 and 1
        x = self.block1(x, A_adj)
        x = self.block2(x, A_adj)
        x = self.global_pool(x)
        return self.fc(torch.flatten(x, 1))


# --- 5. Training Engine Block ---
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
    print(f"\nTraining execution engine online. Device target: {device}")

    # 1. Pipeline Dataset Loaders Initialization
    print("Slicing and processing Intra-Subject Datasets...")
    train_dataset = MEGPipelineDataset(INTRA_TRAIN, window_duration=1.0, downsample_factor=4)
    test_dataset = MEGPipelineDataset(INTRA_TEST, window_duration=1.0, downsample_factor=4)

    # 2. RUNNING SANITY CHECK BEFORE EXTRACTION
    print("\n=== RUNNING PIPELINE DATASET SANITY CHECK ===")
    print(f"✔ Total extracted train window slices: {len(train_dataset)}")
    print(f"✔ Total extracted test window slices:  {len(test_dataset)}")
    
    if len(train_dataset) > 0:
        sample_matrix, sample_label = train_dataset[0]
        print(f"✔ Matrix Input Shape: {sample_matrix.shape} (Expected: [1, 248, 508])")
        print(f"✔ Tensor Normalization Z-Score Check: Mean={sample_matrix.mean().item():.3f}, Std={sample_matrix.std().item():.3f}")
    print("=============================================\n")

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # 3. Model Architecture Instantiation 
    model = MEG_STGCN(num_nodes=248, num_classes=4, hidden_dim=32).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    # 4. Standard Base Optimization Run Loops
    print("Beginning Training Loops...")
    for epoch in range(1, 4):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = run_epoch(model, test_loader, criterion, optimizer=None, device=device)
        print(f"Epoch {epoch:02d} | Train Acc: {train_acc*100:.1f}% (Loss: {train_loss:.4f}) | Test Acc: {test_acc*100:.1f}%")