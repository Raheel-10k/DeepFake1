import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Flatten, Input
from tensorflow.keras.models import Model
import cv2
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

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

def prepare_dataset(dataset_dir, frames_per_video=30):
    videos = []
    labels = []
    for label in ['fake', 'real', 'generated']:
        label_dir = os.path.join(dataset_dir, label)
        for video_file in os.listdir(label_dir):
            video_path = os.path.join(label_dir, video_file)
            frames = extract_frames(video_path, frames_per_video)
            videos.append(frames)
            labels.append(label)
    
    videos = np.array(videos)
    labels = np.array(labels)
    return videos, labels

def build_model(input_shape, num_classes):
    densenet = DenseNet121(weights='imagenet', include_top=False, input_shape=input_shape)
    densenet.trainable = False  # Freeze DenseNet121 layers
    
    input_layer = Input(shape=input_shape)
    x = TimeDistributed(densenet)(input_layer)
    x = TimeDistributed(Flatten())(x)
    x = LSTM(128)(x)
    output_layer = Dense(num_classes, activation='softmax')(x)
    
    model = Model(input_layer, output_layer)
    return model

def main():
    dataset_dir = 'path_to_your_dataset'
    videos, labels = prepare_dataset(dataset_dir)
    
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    labels = to_categorical(labels)
    
    # Build and compile the model
    model = build_model(input_shape=(30, 224, 224, 3), num_classes=3)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    history = model.fit(videos, labels, batch_size=8, epochs=10, validation_split=0.2)
    
    model.save('deepfake_detection_model.h5')
    np.save('label_encoder.npy', label_encoder.classes_)

if __name__ == "__main__":
    main()