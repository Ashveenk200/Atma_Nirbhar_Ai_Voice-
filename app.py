from flask import Flask, request, jsonify, render_template
import os
import tempfile
import io
import numpy as np
import pandas as pd
import librosa
import spacy
from datetime import timedelta
from contextlib import contextmanager
from transformers import pipeline
from pydub import AudioSegment
import re
import time
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import string
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as VADERSentimentAnalyzer
from flask_cors import CORS

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max upload size
CORS(app, resources={r"/*": {"origins": "http://127.0.0.1:5000"}})

# Download required NLTK data quietly
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('vader_lexicon', quiet=True)

# Load NLP tools
nlp = spacy.load("en_core_web_sm")
transformer_sentiment = pipeline("sentiment-analysis")
vader_analyzer = VADERSentimentAnalyzer()

# Load names list from name.xlsx
try:
    name_df = pd.read_excel("name.xlsx")
    name_column = name_df.columns[0]  # Assuming first column contains names
    name_list = name_df[name_column].dropna().astype(str).tolist()
    name_map = dict(zip([name.lower() for name in name_list], name_list))
    print(f"Loaded {len(name_list)} names from name.xlsx")
except Exception as e:
    print(f"Error loading name list: {e}")
    name_map = {}

# Load counselor data from MBA.xlsx
try:
    counselor_df = pd.read_excel("MBA.xlsx")
    print("Counselor DataFrame loaded successfully")
    print("Columns:", counselor_df.columns.tolist())
    # Detect score column dynamically
    score_col = None
    for col in counselor_df.columns:
        if 'score' in col.lower():
            score_col = col
            break
    if score_col:
        counselor_df[score_col] = pd.to_numeric(counselor_df[score_col], errors='coerce')
        score_range = (counselor_df[score_col].min(), counselor_df[score_col].max())
        print(f"Score range for '{score_col}': {score_range[0]} to {score_range[1]}")
    else:
        print("Warning: No score column found in MBA.xlsx")
        score_range = (0, 100)
except Exception as e:
    print(f"Error loading counselor scores: {e}")
    counselor_df = pd.DataFrame()
    score_range = (0, 100)  # Default range

# Load ASR model pipeline
try:
    asr_pipeline = pipeline("automatic-speech-recognition", model="Oriserve/Whisper-Hindi2Hinglish-Swift")
    print("ASR pipeline loaded.")
except Exception as e:
    print("ASR pipeline loading failed:", e)
    asr_pipeline = None

# Utility functions
def clean_text(text):
    """Lowercase and remove punctuation from text."""
    return text.lower().translate(str.maketrans('', '', string.punctuation))

def process_text(text):
    """Remove stopwords, stem and lemmatize text."""
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    return " ".join([
        lemmatizer.lemmatize(stemmer.stem(word))
        for word in text.split()
        if word not in stop_words
    ])

@contextmanager
def temp_audio_file(uploaded_file):
    """Context manager to save uploaded file to a temporary path and clean up."""
    temp_dir = tempfile.mkdtemp()
    ext = os.path.splitext(uploaded_file.filename)[1]
    temp_path = os.path.join(temp_dir, f"audio_temp{ext}")
    uploaded_file.save(temp_path)
    try:
        yield temp_path
    finally:
        try:
            os.remove(temp_path)
            os.rmdir(temp_dir)
        except Exception as e:
            print(f"Cleanup error for {temp_path}: {e}")

def vader_sentiment_analysis(text):
    """Perform VADER sentiment analysis and return label and scores."""
    scores = vader_analyzer.polarity_scores(text)
    label = "Positive" if scores['compound'] >= 0.05 else "Negative" if scores['compound'] <= -0.05 else "Neutral"
    return label, scores

def convert_to_numeric(text):
    """Replace percentage-like text with numeric equivalents (example placeholder)."""
    # Implement any specific conversions if needed
    return text

def extract_human_names(text):
    """Extract human names from text using spaCy and exact matching with name list."""
    doc = nlp(text)
    matched_spacy = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
    matched_direct = []
    text_lower = text.lower()
    for name_lower, name_orig in name_map.items():
        if name_lower in text_lower:
            matched_direct.append(name_orig)
    matched = list(set(matched_spacy + matched_direct))
    return matched if matched else ["No Name Found"]

def format_processing_time(seconds):
    minutes = int(seconds // 60)
    remaining_seconds = int(seconds % 60)
    return f"{minutes} minutes {remaining_seconds} seconds"

def process_audio_with_timestamps(audio_path):
    """Process audio file in chunks, transcribe, extract names, analyze sentiment."""
    if not asr_pipeline:
        print("ASR pipeline not loaded.")
        return [], "", 0, "ASR Not Loaded", {}

    try:
        print(f"Processing audio file: {audio_path}")
        audio = AudioSegment.from_file(audio_path)
        print(f"Audio duration: {len(audio)/1000} seconds")
        chunk_size = 15000  # 15 seconds in ms
        chunks = [audio[i:i + chunk_size] for i in range(0, len(audio), chunk_size)]
        results = []

        for i, chunk in enumerate(chunks):
            with io.BytesIO() as wav_buffer:
                chunk.export(wav_buffer, format="wav")
                wav_buffer.seek(0)
                audio_data, sr = librosa.load(wav_buffer, sr=16000)
                try:
                    transcription = asr_pipeline(audio_data)
                    text = clean_text(convert_to_numeric(transcription['text']))
                    results.append(text)
                    print(f"Chunk {i} transcribed: {text}")
                except Exception as e:
                    print(f"Error in chunk {i}: {e}")
                    results.append(f"[ERROR: {str(e)}]")

        full_text = " ".join(results)
        # Use the new name extraction function
        matched_names = extract_names_from_namesfound(full_text)
        sentiment_label, sentiment_scores = vader_sentiment_analysis(full_text)
        total_duration = len(audio) / 1000

        print(f"Full text: {full_text}")
        print(f"Names found: {matched_names}")
        print(f"Sentiment: {sentiment_label}, Scores: {sentiment_scores}")

        return matched_names, full_text, total_duration, sentiment_label, sentiment_scores

    except Exception as e:
        print(f"Audio processing error for {audio_path}: {e}")
        return [], "", 0, f"Error: {str(e)}", {}

def recommend_counselor(compound_score):
    """Recommend counselor based on compound sentiment score and counselor data."""
    try:
        compound_score = float(compound_score)
    except Exception:
        print("Error: Invalid compound_score provided")
        return "Invalid Score"

    if counselor_df.empty:
        print("Error: counselor_df is empty")
        return "No Data Available"

    score_col = None
    name_col = None
    for col in counselor_df.columns:
        if 'score' in col.lower():
            score_col = col
        if 'counselor' in col.lower() or 'name' in col.lower():
            name_col = col

    if not score_col or not name_col:
        print("Error: Could not find 'score' or 'counselor name' columns")
        return "Invalid Data Format"

    counselor_df[score_col] = pd.to_numeric(counselor_df[score_col], errors='coerce')
    valid_df = counselor_df.dropna(subset=[score_col])

    if valid_df.empty:
        print("Error: No valid scores after cleaning")
        return "No Valid Scores"

    min_score = valid_df[score_col].min()
    max_score = valid_df[score_col].max()

    # Scale compound_score (-1 to 1) to counselor score range
    if max_score != min_score:
        scaled_compound = min_score + (compound_score + 1) * (max_score - min_score) / 2
    else:
        scaled_compound = compound_score

    match = valid_df[valid_df[score_col] > scaled_compound]
    return ", ".join(match[name_col]) if not match.empty else "No Match Found"

def extract_names_from_namesfound(text):
    """
    Extract names from the first 20 comma-separated words of the input text,
    matching against the name.xlsx list.
    """
    words = text.split(",")  # Split into comma-separated words
    first_20_words = words[:20]  # Take the first 20 words
    text_subset = " ".join(first_20_words)  # Join them back into a string

    matched_names = []
    text_lower = text_subset.lower()

    for name_lower, name_orig in name_map.items():
        if name_lower in text_lower:
            matched_names.append(name_orig)

    return list(set(matched_names)) if matched_names else ["No Name Found"]

# Flask routes
@app.route('/')
def home():
    return render_template("index.html")

@app.route('/analyze_audio', methods=['POST'])
def analyze_audio():
    print("Received /analyze_audio request")
    uploaded_files = request.files.getlist("audio")
    print(f"Uploaded files: {[file.filename for file in uploaded_files]}")

    if not uploaded_files:
        print("No files received")
        return jsonify({"success": False, "result": "No audio files uploaded"}), 400

    all_results = []
    start_time = time.time()

    for file in uploaded_files:
        print(f"\nStarting processing for file: {file.filename}")
        try:
            with temp_audio_file(file) as temp_path:
                print(f"Temp audio path: {temp_path}")
                matched_names, full_text, total_duration, sentiment_label, sentiment_scores = process_audio_with_timestamps(temp_path)
                interest_level = "Interested" if sentiment_label == "Positive" else "Not Interested" if sentiment_label == "Negative" else "Neutral"
                processing_time_seconds = time.time() - start_time

                result = {
                    "FileName": file.filename,
                    "NamesFound": ", ".join(matched_names),
                    "NormalTranscription": full_text,
                    "ProcessingTime": format_processing_time(processing_time_seconds),
                    "PositiveScore": round(sentiment_scores.get('pos', 0) * 100, 2),
                    "NegativeScore": round(sentiment_scores.get('neg', 0) * 100, 2),
                    "NeutralScore": round(sentiment_scores.get('neu', 0) * 100, 2),
                    "CompoundScore": round(sentiment_scores.get('compound', 0) * 100, 2),
                    "Sentiment": interest_level,
                    "RecommendedCounselor": recommend_counselor(sentiment_scores.get('compound', 0))
                }
                all_results.append(result)
                print(f"Result for {file.filename}: {result}")

        except Exception as e:
            print(f"Error processing {file.filename}: {e}")
            result = {
                "FileName": file.filename,
                "Error": f"Failed to process: {str(e)}",
                "RecommendedCounselor": "Not Available"
            }
            all_results.append(result)

    print("\nFinal results:", all_results)
    return jsonify({"success": True, "results": all_results})

@app.route('/analyze_text', methods=['POST'])
def analyze_text():
    print("Received /analyze_text request")
    data = request.json
    text_input = data.get("text", "").strip()

    if not text_input:
        print("No text provided")
        return jsonify({"success": False, "result": "No text provided"}), 400

    # Extract names using the new function
    names_found = extract_names_from_namesfound(text_input)

    score = vader_analyzer.polarity_scores(text_input)
    compound = score['compound']
    sentiment = 'Positive 😊' if compound >= 0.3 else 'Negative 😞' if compound <= -0.3 else 'Neutral 😐'
    print(f"Text analysis result: Sentiment={sentiment}, Scores={score}")

    return jsonify({
        "success": True,
        "Sentiment": sentiment,
        "NormalTranscription": text_input,
        "NamesFound": ", ".join(names_found),  # Include names found in the response
        "Scores": {
            "pos": round(score['pos'] * 100, 2),
            "neg": round(score['neg'] * 100, 2),
            "neu": round(score['neu'] * 100, 2),
            "compound": round(score['compound'] * 100, 2)
        }
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)