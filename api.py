from flask import Flask, request, jsonify
from flask_cors import CORS  # ✅ Import CORS
from werkzeug.utils import secure_filename
from predict import explain_prediction  # Import model function
import os
import librosa

app = Flask(__name__)
CORS(app)  # ✅ Enable CORS for all routes

# Allowed audio formats
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac'}
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Ensure the uploads folder exists

# Maximum file size limit (e.g., 10MB)
MAX_FILE_SIZE = 10 * 1024 * 1024

# Minimum and maximum audio durations in seconds
MIN_AUDIO_DURATION = 3
MAX_AUDIO_DURATION = 30

def allowed_file(filename):
    """Check if the uploaded file has a valid extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_audio_duration(file_path):
    """Check if the audio file duration is within the allowed range."""
    try:
        duration = librosa.get_duration(filename=file_path)
        if MIN_AUDIO_DURATION <= duration <= MAX_AUDIO_DURATION:
            return True, None
        else:
            return False, f"Audio duration must be between {MIN_AUDIO_DURATION} and {MAX_AUDIO_DURATION} seconds."
    except Exception as e:
        return False, str(e)

@app.before_request
def check_request_size():
    """Check if the request size exceeds the maximum limit."""
    if request.content_length and request.content_length > MAX_FILE_SIZE:
        return jsonify({"error": "Request size exceeds the 10MB limit."}), 413

@app.route('/')
def home():
    """ Simple message to indicate the API is running """
    return jsonify({"message": "Audio Deepfake Detection API is running"}), 200

@app.route('/predict', methods=['POST'])
def predict_audio():
    # File validation
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded", "result": []}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No file uploaded", "result": []}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        # Validate audio duration
        is_valid_duration, error_message = validate_audio_duration(file_path)
        if not is_valid_duration:
            os.remove(file_path)
            return jsonify({"error": error_message, "result": []}), 400

        try:
            # Run prediction model
            prediction = explain_prediction(file_path)
            probability_fake = prediction["probability_fake"]
            result = "Fake Audio" if probability_fake >= 50 else "Real Audio"
            result_boolean = "true" if probability_fake >= 50 else "false"

            # JSON format changes for response
            response = {
                "filename": filename,
                "prediction": result,
                "probability_fake": probability_fake,
                "Deepfake_Prediction": result_boolean
            }

        except Exception as e:
            return jsonify({"error": "Error during prediction.", "details": str(e)}), 500

        finally:
            # Remove the processed file after prediction
            os.remove(file_path)

        return jsonify(response)

    return jsonify({"error": "Invalid file format", "result": []}), 400

@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file size limit errors."""
    return jsonify({"error": "File size exceeds the allowed limit."}), 413

@app.errorhandler(500)
def internal_server_error(error):
    """Handle unexpected server errors."""
    return jsonify({"error": "An internal server error occurred."}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
