from flask import Flask, request, jsonify
import librosa
import numpy as np
import torch
import tensorflow as tf
from transformers import Wav2Vec2Processor, Wav2Vec2Model, AutoFeatureExtractor, WavLMForXVector

import os

app = Flask(__name__)

# Load models lazily to optimize memory usage
wave2vec_model = None
processor = None
wavlm_model = None
feature_extractor = None
siamese_model = None
ensemble_classifier_model = None
device = torch.device("cpu")

def load_models():
    global wave2vec_model, processor, wavlm_model, feature_extractor, siamese_model, ensemble_classifier_model
    
    if wave2vec_model is None:
        wave2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-960h").to(device)
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")

    if wavlm_model is None:
        wavlm_model = WavLMForXVector.from_pretrained("microsoft/wavlm-large").to(device)
        feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-large")

    if siamese_model is None:
        siamese_model = tf.keras.models.load_model("feature_extractor_model.h5", compile=False)

    if ensemble_classifier_model is None:
        ensemble_classifier_model = tf.keras.models.load_model("ensemble_classifier.keras")

# TensorFlow GPU handling
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Helper Functions
def mel_spectrogram_gen(audio_path):
    signal, sample_rate = librosa.load(audio_path, sr=16000, duration=2)  # Use 16kHz to match model inputs
    mel_spectrogram = librosa.feature.melspectrogram(y=signal, sr=sample_rate, n_mels=128, hop_length=512)
    return librosa.power_to_db(mel_spectrogram, ref=np.max)

def pad_mel_spectrograms(mel_spectrogram, max_pad_len=87):
    pad_width = max_pad_len - mel_spectrogram.shape[1]
    return np.pad(mel_spectrogram, ((0, 0), (0, max(0, pad_width))), mode='constant')[:, :max_pad_len]

def extract_wave2vec_features(audio_file):
    load_models()
    audio_input, _ = librosa.load(audio_file, sr=16000)
    inputs = processor(audio_input, sampling_rate=16000, return_tensors="pt", padding=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = wave2vec_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

def extract_xvector_features(audio_file):
    load_models()
    audio_input, _ = librosa.load(audio_file, sr=16000)
    inputs = feature_extractor(audio_input, sampling_rate=16000, return_tensors="pt", padding=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = wavlm_model(**inputs).embeddings
    return outputs.mean(dim=1).squeeze().cpu().numpy()

def normalize_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-9)  # Avoid division by zero

def explain_prediction(audio_file):
    load_models()

    mel_spectrogram = mel_spectrogram_gen(audio_file)
    mel_spectrogram_padded = pad_mel_spectrograms(mel_spectrogram)
    mel_spectrogram_padded = normalize_data(mel_spectrogram_padded)

    # Prepare input for the Siamese model
    mel_spectrogram_padded = np.expand_dims(mel_spectrogram_padded, axis=(0, -1))

    wave2vec_features = extract_wave2vec_features(audio_file).reshape((1, -1))
    xvector_features = extract_xvector_features(audio_file).reshape((1, -1))
    siamese_features = siamese_model.predict(mel_spectrogram_padded).reshape((1, -1))

    # Combine features
    combined_features = np.concatenate([wave2vec_features, siamese_features, xvector_features], axis=1)

    prediction = ensemble_classifier_model.predict(combined_features)
    probability_fake = float(prediction[0][0]) * 100
    result = "Fake Audio" if prediction[0][0] >= 0.5 else "Real Audio"

    return {
        "result": result,
        "probability_fake": probability_fake,
    }

@app.route('/')
def home():
    return jsonify({"message": "Welcome to the Audio Deepfake Detection API!"})

@app.route('/favicon.ico')
def favicon():
    return "", 204  # Prevent unnecessary favicon requests

@app.route('/predict', methods=['POST'])
def predict():
    if 'audio_file' not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400

    audio_file = request.files['audio_file']
    audio_path = "temp_audio.wav"
    audio_file.save(audio_path)

    try:
        prediction = explain_prediction(audio_path)
        os.remove(audio_path)  # Clean up temp file
        return jsonify(prediction)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=False)  # Ensure debug is False for production
