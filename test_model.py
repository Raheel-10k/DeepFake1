import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
from sklearn.preprocessing import LabelEncoder

def extract_frames(video_path, frames_per_video=30):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, frames_per_video, dtype=int)
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (224, 224)) 
            frames.append(frame)
    
    cap.release()
    return np.array(frames)

def predict_video(video_path, model, label_encoder):
    frames = extract_frames(video_path)
    frames = np.expand_dims(frames, axis=0)
    prediction = model.predict(frames)
    predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])
    return predicted_label[0]

def main():
    model = load_model('deepfake_detection_model.h5')
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.load('label_encoder.npy')
    
    video_path = 'path_to_test_video.mp4'
    
    predicted_label = predict_video(video_path, model, label_encoder)
    print(f'The video is classified as: {predicted_label}')

if __name__ == "__main__":
    main()