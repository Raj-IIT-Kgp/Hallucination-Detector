from flask import Flask, request, jsonify, send_from_directory, Response
from detector.hallucination_detector import HallucinationDetector
import os

from retrieval.web_search import WebRetriever

app = Flask(__name__, static_folder='static')
store = WebRetriever()
detector = HallucinationDetector(store)

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze():
    data = request.json
    text = data.get('text', '')
    
    if not text:
        return jsonify({"error": "No text provided"}), 400

    try:
        # Run the detector pipeline
        report = detector.analyze(text)
        return jsonify(report)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/analyze_stream', methods=['POST'])
def analyze_stream_endpoint():
    data = request.json
    text = data.get('text', '')
    
    if not text:
        return jsonify({"error": "No text provided"}), 400

    def generate():
        try:
            for update in detector.analyze_stream(text):
                yield update
        except Exception as e:
            import json
            yield json.dumps({"status": "error", "message": str(e)}) + "\n"

    return Response(generate(), mimetype='application/x-ndjson')

if __name__ == '__main__':
    # Make sure static folder exists
    os.makedirs('static', exist_ok=True)
    print("Starting Hallucination Detector Web Server...")
    app.run(debug=True, host='0.0.0.0', port=8080)
