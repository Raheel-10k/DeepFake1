import torch
import cv2
import numpy as np
from torchvision import transforms
from train import CFG, DCTTransform, EfficientTSM  # Reuse components

class DeepFakeDetector:
    def __init__(self, model_path):
        self.model = EfficientTSM()
        self.model.load_state_dict(torch.load(model_path))
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
            if len(frames) == CFG['num_frames']: break
        cap.release()
        return torch.stack(frames[:CFG['num_frames']])

    def predict(self, video_path):
        with torch.no_grad():
            inputs = self.process_video(video_path).unsqueeze(0)
            outputs = self.model(inputs)
            return torch.softmax(outputs, dim=1).numpy()[0]

if __name__ == "__main__":
    detector = DeepFakeDetector('deepfake_model.pth')
    prediction = detector.predict('test_video.mp4')
    print(f"Real: {prediction[0]:.2f}, Fake: {prediction[1]:.2f}")