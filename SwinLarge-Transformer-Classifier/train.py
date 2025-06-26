# pip install timm torchvision

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import timm
from PIL import Image
import os

# 1) Dataset for wafer defect images (adjust paths & labels as needed)
class WaferImageDataset(Dataset):
    def __init__(self, img_dir, transform):
        self.paths = []
        self.labels = []
        for label, sub in enumerate(["healthy", "defective"]):
            folder = os.path.join(img_dir, sub)
            for fn in os.listdir(folder):
                if fn.endswith(".png") or fn.endswith(".jpg"):
                    self.paths.append(os.path.join(folder, fn))
                    self.labels.append(label)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), torch.tensor(self.labels[idx], dtype=torch.long)

# 2) Transforms
transform = T.Compose([
    T.Resize(256),
    T.RandomResizedCrop(224, scale=(0.8,1.0)),
    T.RandomHorizontalFlip(),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

# 3) DataLoaders
train_ds = WaferImageDataset("data/train", transform)
val_ds   = WaferImageDataset("data/val",   transform)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=4)

# 4) Model: ConvNeXt-base
model = timm.create_model(
    'convnext_base',        # you can also try 'convnext_small' or 'convnext_large'
    pretrained=True,
    num_classes=2           # binary healthy vs defective
)
# Optional: freeze stem if you want to fine-tune only the head
# for name, p in model.named_parameters():
#     if "features.stem" in name:
#         p.requires_grad = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 5) Loss, optimizer, scheduler
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=1e-2)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

# 6) Training loop
def train_one_epoch():
    model.train()
    total_loss = 0
    correct = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        preds = model(imgs)
        loss = criterion(preds, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct += (preds.argmax(1) == labels).sum().item()
    return total_loss / len(train_ds), correct / len(train_ds)

def validate():
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs)
            total_loss += criterion(preds, labels).item() * imgs.size(0)
            correct += (preds.argmax(1) == labels).sum().item()
    return total_loss / len(val_ds), correct / len(val_ds)

for epoch in range(1, 11):
    train_loss, train_acc = train_one_epoch()
    val_loss, val_acc     = validate()
    scheduler.step()
    print(f"Epoch {epoch:02d}  "
          f"Train Loss {train_loss:.4f} Acc {train_acc:.3f}  "
          f"Val   Loss {val_loss:.4f} Acc {val_acc:.3f}")