from flask import Flask, request, jsonify
import librosa
import numpy as np
import torch
import tensorflow as tf
from transformers import Wav2Vec2Processor, Wav2Vec2Model, AutoFeatureExtractor, WavLMForXVector
import os

app = Flask(__name__)

# Global variables for models and devices
device = torch.device("cpu")
wave2vec_model = None
processor = None
wavlm_model = None
feature_extractor = None
siamese_model = None
ensemble_classifier_model = None

# TensorFlow GPU handling
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# Load models once at app startup
def initialize_models():
    global wave2vec_model, processor, wavlm_model, feature_extractor, siamese_model, ensemble_classifier_model

    # Load PyTorch models
    print("Loading Wav2Vec2 and WavLM models...")
    wave2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-960h").to(device)
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
    wavlm_model = WavLMForXVector.from_pretrained("microsoft/wavlm-large").to(device)
    feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-large")

    # Load TensorFlow models
    print("Loading TensorFlow models...")
    siamese_model = tf.keras.models.load_model("feature_extractor_model.h5", compile=False)
    ensemble_classifier_model = tf.keras.models.load_model("ensemble_classifier.keras")
    print("Models loaded successfully!")

def explain_prediction(audio_file):
    # Generate Mel spectrogram
    signal, sample_rate = librosa.load(audio_file, sr=16000, duration=2)
    mel_spectrogram = librosa.feature.melspectrogram(y=signal, sr=sample_rate, n_mels=128, hop_length=512)
    mel_spectrogram_padded = np.pad(mel_spectrogram, ((0, 0), (0, max(0, 87 - mel_spectrogram.shape[1]))), mode='constant')

    # Extract features
    inputs = processor(signal, sampling_rate=16000, return_tensors="pt", padding=True).to(device)
    wave2vec_features = wave2vec_model(**inputs).last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    xvector_features = wavlm_model(**inputs).embeddings.mean(dim=1).squeeze().cpu().numpy()
    mel_features = siamese_model.predict(np.expand_dims(mel_spectrogram_padded, axis=(0, -1)))

    combined_features = np.concatenate([wave2vec_features.reshape(1, -1), mel_features.reshape(1, -1), xvector_features.reshape(1, -1)], axis=1)
    prediction = ensemble_classifier_model.predict(combined_features)
    result = "Fake Audio" if prediction[0][0] >= 0.5 else "Real Audio"
    probability_fake = float(prediction[0][0]) * 100

    return {"result": result, "probability_fake": probability_fake}

@app.route('/')
def home():
    return jsonify({"message": "Welcome to the Audio Deepfake Detection API!"})

@app.route('/predict', methods=['POST'])
def predict():
    if 'audio_file' not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400

    audio_file = request.files['audio_file']
    audio_path = "temp_audio.wav"
    audio_file.save(audio_path)

    try:
        prediction = explain_prediction(audio_path)
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        os.remove(audio_path)

    return jsonify(prediction)

if __name__ == '__main__':
    print("Starting server and loading models...")
    initialize_models()
    app.run(host="0.0.0.0", port=5000, debug=False)
