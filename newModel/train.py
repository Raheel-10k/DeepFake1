import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import numpy as np
import cv2
import os
from tqdm import tqdm

# Configuration (EDIT THIS)
CFG = {
    'frame_size': 224,
    'num_frames': 15,
    'batch_size': 32,
    'lr': 1e-4,
    'epochs': 20,
    'num_classes': 2,
    'dct_channels': 3
}

# DCT Transform Class (new)
class DCTTransform:
    def __call__(self, img):
        img_np = np.array(img)
        ycrcb = cv2.cvtColor(img_np, cv2.COLOR_RGB2YCrCb)
        dct_blocks = [cv2.dct(np.float32(ycrcb[..., i])) for i in range(3)]
        return torch.tensor(np.stack(dct_blocks, axis=-1), dtype=torch.float32)

# Dataset Class (WORK HERE)
class DeepFakeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        for label, cls in enumerate(['real', 'fake']):
            cls_dir = os.path.join(root_dir, cls)
            for video_dir in os.listdir(cls_dir):
                self.samples.append((os.path.join(cls_dir, video_dir), label))
        self.transform = transform

    def __getitem__(self, idx):
        video_dir, label = self.samples[idx]
        frames = sorted(os.listdir(video_dir))[:CFG['num_frames']]
        selected_frames = [self.transform(cv2.imread(os.path.join(video_dir, f))) 
                         for f in frames[::len(frames)//CFG['num_frames']]]
        return torch.stack(selected_frames), torch.tensor(label)

    def __len__(self):
        return len(self.samples)

# Model Architecture (UNDERSTAND)
class EfficientTSM(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.efficientnet_b0(pretrained=True)
        self.backbone.features[0] = nn.Conv2d(CFG['dct_channels'], 32, kernel_size=3, stride=2, padding=1, bias=False)
        
        # Temporal Shift Module (IMP)
        self.tsm_conv = nn.Sequential(
            nn.Conv3d(1280, 512, kernel_size=(3,1,1), padding=(1,0,0)),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((None, 1, 1))
        )
        
        self.classifier = nn.Linear(512, CFG['num_classes'])

    def temporal_shift(self, x):
        shift_ratio = 0.25
        b, c, t, h, w = x.size()
        x = x.permute(0, 2, 1, 3, 4)
        x = x.contiguous().view(-1, c, h, w)
        shifted = torch.roll(x, shifts=int(c*shift_ratio), dims=1)
        return shifted.view(b, t, c, h, w).permute(0, 2, 1, 3, 4)

    def forward(self, x):
        b, t, c, h, w = x.size()
        x = x.view(-1, c, h, w)
        features = self.backbone.extract_features(x)
        _, c_feat, h_feat, w_feat = features.size()
        features = features.view(b, t, c_feat, h_feat, w_feat)
        features = self.temporal_shift(features)
        features = features.view(-1, c_feat, h_feat, w_feat)
        pooled = nn.functional.adaptive_avg_pool2d(features, 1)
        pooled = pooled.view(b, t, -1)
        temporal_features = self.tsm_conv(pooled.unsqueeze(2))
        return self.classifier(temporal_features.squeeze())

# Training Setup (EDIT HERE)
transform = transforms.Compose([
    transforms.Lambda(lambda x: cv2.resize(x, (CFG['frame_size'], CFG['frame_size']))),
    DCTTransform(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

dataset = DeepFakeDataset('path_to_training_data', transform=transform)
dataloader = DataLoader(dataset, batch_size=CFG['batch_size'], shuffle=True)

model = EfficientTSM()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=CFG['lr'])

# Training Loop (DONT TOUCH TILL TRAIN_DATA)
for epoch in range(CFG['epochs']):
    model.train()
    running_loss = 0.0
    for inputs, labels in tqdm(dataloader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1} Loss: {running_loss/len(dataloader)}")
    
torch.save(model.state_dict(), 'deepfake_model.pth')