import torch
import cv2
import numpy as np
from torchvision import transforms
from train import CFG, DCTTransform, EfficientTSM

class DeepFakeDetector:
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = EfficientTSM().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Lambda(lambda x: cv2.resize(x, (CFG['frame_size'], CFG['frame_size']))),
            DCTTransform(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])

    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(self.transform(frame))
            if len(frames) == CFG['num_frames']*2: break  # Get more frames for safety
        cap.release()
        
        # Select best frames with motion
        if len(frames) >= CFG['num_frames']:
            indices = np.linspace(0, len(frames)-1, CFG['num_frames'], dtype=int)
        else:
            indices = np.arange(len(frames))
        return torch.stack([frames[i] for i in indices])

    def predict(self, video_path):
        with torch.no_grad():
            inputs = self.process_video(video_path).unsqueeze(0).to(self.device)
            
            # Test-Time Augmentation
            flipped_inputs = torch.flip(inputs, dims=[-1])  # Horizontal flip
            outputs = self.model(inputs)
            outputs_flipped = self.model(flipped_inputs)
            
            # Ensemble predictions
            avg_outputs = (outputs + outputs_flipped) / 2
            return torch.softmax(avg_outputs, dim=1).cpu().numpy()[0]

if __name__ == "__main__":
    detector = DeepFakeDetector('deepfake_model.pth')
    prediction = detector.predict('test_video.mp4')
    print(f"Real: {prediction[0]:.2f}, Fake: {prediction[1]:.2f}")