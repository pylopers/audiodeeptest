from flask import Flask, request, jsonify
import librosa
import numpy as np
import torch
import tensorflow as tf
from transformers import Wav2Vec2Processor, Wav2Vec2Model, AutoFeatureExtractor, WavLMForXVector

import subprocess


app = Flask(__name__)

# Load models
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
def mel_spectrogram_gen(audio_path):
    signal, sample_rate = librosa.load(audio_path, sr=22050, duration=2)
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

def explain_prediction(audio_file):
    mel_spectrogram = mel_spectrogram_gen(audio_file)
    mel_spectrogram_padded = pad_mel_spectrograms(mel_spectrogram)
    mel_spectrogram_padded = normalize_data(mel_spectrogram_padded)

    mel_spectrogram_padded = np.expand_dims(mel_spectrogram_padded, axis=-1)
    mel_spectrogram_padded = np.expand_dims(mel_spectrogram_padded, axis=0)

    wave2vec_features = extract_wave2vec_features(audio_file)
    xvector_features = extract_xvector_features(audio_file)
    siamese_features = siamese_model.predict(mel_spectrogram_padded)

    wave2vec_features = wave2vec_features.reshape((1, -1))
    siamese_features = siamese_features.reshape((1, -1))
    xvector_features = xvector_features.reshape((1, -1))

    combined_features = np.concatenate([wave2vec_features, siamese_features, xvector_features], axis=1)
    prediction = ensemble_classifier_model.predict(combined_features)
    probability_fake = prediction[0][0] * 100
    result = "Fake Audio" if prediction[0][0] >= 0.5 else "Real Audio"

    return {
        "result": result,
        "probability_fake": probability_fake,
        "mel_spectrogram": mel_spectrogram.tolist(),
    }

@app.route('/')
def home():
    return jsonify({"message": "Welcome to the Audio Deepfake Detection API!"})

@app.route('/favicon.ico')
def favicon():
    return "", 204  # Return an empty response for the favicon.ico request

# Flask Routes
@app.route('/predict', methods=['POST'])
def predict():
    if 'audio_file' not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400
    audio_file = request.files['audio_file']
    audio_path = "temp_audio.wav"
    audio_file.save(audio_path)

    try:
        prediction = explain_prediction(audio_path)
        return jsonify(prediction)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
