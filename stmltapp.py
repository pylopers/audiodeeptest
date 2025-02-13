import streamlit as st
import librosa
import numpy as np
import torch
import tensorflow as tf
from transformers import Wav2Vec2Processor, Wav2Vec2Model, AutoFeatureExtractor, WavLMForXVector
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# Load models
@st.cache_resource
def load_models():
    wave2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-960h")
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
    wavlm_model = WavLMForXVector.from_pretrained("microsoft/wavlm-large")
    feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-large")
    siamese_model = tf.keras.models.load_model("feature_extractor_model.h5", compile=False)
    ensemble_classifier_model = tf.keras.models.load_model("ensemble_classifier.keras")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    wave2vec_model.to(device)
    wavlm_model.to(device)
    return wave2vec_model, processor, wavlm_model, feature_extractor, siamese_model, ensemble_classifier_model, device

wave2vec_model, processor, wavlm_model, feature_extractor, siamese_model, ensemble_classifier_model, device = load_models()

# Helper Functions
def mel_spectrogram_gen(audio):
    signal, sample_rate = librosa.load(audio, sr=22050, duration=2)
    hop_length = 512
    n_mels = 128
    mel_spectrogram = librosa.feature.melspectrogram(y=signal, sr=sample_rate, hop_length=hop_length, n_mels=n_mels)
    return librosa.power_to_db(mel_spectrogram, ref=np.max)

def pad_mel_spectrograms(mel_spectrogram, max_pad_len=87):
    pad_width = max_pad_len - mel_spectrogram.shape[1]
    if pad_width > 0:
        mel_spectrogram = np.pad(mel_spectrogram, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mel_spectrogram = mel_spectrogram[:, :max_pad_len]
    return mel_spectrogram

def extract_wave2vec_features(audio_file):
    audio_input, _ = librosa.load(audio_file, sr=16000)
    inputs = processor(audio_input, sampling_rate=16000, return_tensors="pt", padding=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = wave2vec_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

def extract_xvector_features(audio_file):
    audio_input, _ = librosa.load(audio_file, sr=16000)
    inputs = feature_extractor(audio_input, sampling_rate=16000, return_tensors="pt", padding=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = wavlm_model(**inputs).embeddings
    return outputs.mean(dim=1).squeeze().cpu().numpy()

def normalize_data(data):
    return (data - data.min()) / (data.max() - data.min())


# Inference function with comparison
def explain_prediction(audio_file):
    # Check if speech is present

    
    mel_spectrogram = mel_spectrogram_gen(audio_file)
    max_pad_len = 87  # Ensure this matches the value used in training
    mel_spectrogram_padded1 = pad_mel_spectrograms(mel_spectrogram, max_pad_len)
    mel_spectrogram_padded = normalize_data(mel_spectrogram_padded1)
    
    # Adjust shape to match model input
    mel_spectrogram_padded = np.expand_dims(mel_spectrogram_padded, axis=-1)  # Shape: (128, 87, 1)
    mel_spectrogram_padded = np.expand_dims(mel_spectrogram_padded, axis=0)   # Shape: (1, 128, 87, 1)

    # Feature extraction
    wave2vec_features = extract_wave2vec_features(audio_file)
    xvector_features = extract_xvector_features(audio_file)
    siamese_features = siamese_model.predict(mel_spectrogram_padded)

    # Reshape features
    wave2vec_features = wave2vec_features.reshape((1, -1))
    siamese_features = siamese_features.reshape((1, -1))
    xvector_features = xvector_features.reshape((1, -1))

    # Concatenate features for ensemble classifier
    combined_features = np.concatenate([wave2vec_features, siamese_features, xvector_features], axis=1)

    # Predict using ensemble classifier
    prediction = ensemble_classifier_model.predict(combined_features)
    probability_fake = prediction[0][0] * 100

    result = "Fake Audio" if prediction[0][0] >= 0.5 else "Real Audio"
    return mel_spectrogram, real_differences, fake_differences, result, probability_fake

# Streamlit App UI
st.title("Audio Deepfake Detection")

uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav","flac"])
if uploaded_file:
    with open("temp_audio_file", "wb") as f:
        f.write(uploaded_file.read())
    
    mel_spectrogram, real_differences, fake_differences, result, probability = explain_prediction("temp_audio_file")

    st.write(f"Prediction: {result}")
    st.write(f"Fake Probability: {probability:.2f}%")

    # Visualize Mel Spectrograms
    st.write("**Uploaded Audio Mel Spectrogram**")
    visualize_mel_spectrogram(mel_spectrogram, "Uploaded Audio")

    st.write("**Real Reference Mel Spectrogram**")
    visualize_mel_spectrogram(real_reference_mel_spectrogram, "Real Reference Audio")

    st.write("**Fake Reference Mel Spectrogram**")
    visualize_mel_spectrogram(fake_reference_mel_spectrogram, "Fake Reference Audio")

    # Plot Feature Differences
    st.write("**Feature Differences with Real and Fake References**")
    plot_feature_differences(real_differences, fake_differences)