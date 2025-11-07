
import os
import torch
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as T

# --- config ---
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT, "data")
IMG_SIZE = 224
BATCH_SIZE = 64
VAL_SPLIT = 0.2
SEED = 42
NUM_WORKERS = 2
PIN = torch.cuda.is_available()

# --- transforms ---
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

train_tf = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.RandomHorizontalFlip(0.5),
    T.RandomRotation(10),
    T.ColorJitter(0.1, 0.1, 0.05),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

val_tf = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

def get_loaders(
    data_dir: str = DATA_DIR,
    batch_size: int = BATCH_SIZE,
    val_split: float = VAL_SPLIT,
    seed: int = SEED,
    num_workers: int = NUM_WORKERS,
    pin_memory: bool = PIN,
):

    full = torchvision.datasets.ImageFolder(root=data_dir, transform=train_tf)
    n = len(full)
    n_val = int(n * val_split)
    n_train = n - n_val

    g = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(full, [n_train, n_val], generator=g)

    # clean tf for val
    val_set.dataset = torchvision.datasets.ImageFolder(root=data_dir, transform=val_tf)

    class_to_idx = full.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=False
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size * 2, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=False
    )
    return train_loader, val_loader, idx_to_class


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_loader, val_loader, idx_to_class = get_loaders()
    print(f"Total images: {len(train_loader.dataset) + len(val_loader.dataset)}  |  "
          f"Train: {len(train_loader.dataset)}  Val: {len(val_loader.dataset)}")
    print("Num classes:", len(idx_to_class))
    print("Classes:", list(idx_to_class.values()))


    xb, yb = next(iter(train_loader))
    print("Batch:", xb.shape, yb.shape)
