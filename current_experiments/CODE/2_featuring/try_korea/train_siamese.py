import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from siamese_model import SiameseNetwork, ContrastiveLoss
import random


features_path = r'current_experiments\DATA\processed\experiment_001\experiment_001(1-4)_test_cleaned.npy'
labels_path = r'current_experiments\DATA\processed\experiment_001\experiment_001(1-4)_test_labels.npy'

features = np.load(features_path)
labels = np.load(labels_path)

class EEGPairDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __getitem__(self, index):
        x1 = self.features[index]
        y1 = self.labels[index]
        if random.randint(0, 1):
            idx2 = random.choice(np.where(self.labels == y1)[0])
            label = 0
        else:
            idx2 = random.choice(np.where(self.labels != y1)[0])
            label = 1
        x2 = self.features[idx2]
        return torch.tensor(x1, dtype=torch.float32), torch.tensor(x2, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

dataset = EEGPairDataset(features, labels)
loader = DataLoader(dataset, batch_size=128, shuffle=True)

model = SiameseNetwork(input_dim=features.shape[1])
criterion = ContrastiveLoss(margin=0.5)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(20):
    total_loss = 0
    for x1, x2, y in loader:
        optimizer.zero_grad()
        out1 = model(x1)
        out2 = model(x2)
        loss = criterion(out1, out2, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")

model.eval()
with torch.no_grad():
    reduced_features = model(torch.tensor(features, dtype=torch.float32)).numpy()

np.save("experiment_001(1-4)_test_8d_features.npy", reduced_features)
