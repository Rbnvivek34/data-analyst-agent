from flask import Flask, request, jsonify
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import os
import tempfile
import mimetypes
import time
import requests
import json

app = Flask(__name__)

@app.route("/")
def home():
    return "✅ Data Analyst Agent is live. Use POST /api/ with a question file."

@app.route("/api/", methods=["POST"])
def handle_request():
    start_time = time.time()

    if "questions.txt" not in request.files:
        return jsonify({"error": "questions.txt file is required"}), 400

    question_text = request.files["questions.txt"].read().decode("utf-8")

    # Save uploaded files (except questions.txt)
    file_data = {}
    for key in request.files:
        if key == "questions.txt":
            continue
        file = request.files[key]
        suffix = os.path.splitext(file.filename)[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name
        file_data[file.filename] = tmp_path

    try:
        system_prompt = """You are a data analyst assistant. Given a task and uploaded files, do the following:

- Read and process the data files (e.g., CSVs).
- Perform any requested analysis (e.g., edge count, degree computation, shortest path).
- Generate any requested plots using Python, and convert them to base64 PNG strings (under 100 kB).
- For base64 images, ensure actual image generation with matplotlib/networkx, not dummy or placeholder strings.
- Return a structured JSON result only (no explanation).
"""

        file_summary = "\n".join([f"- {k}: {mimetypes.guess_type(k)[0] or 'Unknown type'}" for k in file_data])

        user_prompt = f"""Task:\n{question_text}

Uploaded Files:
{file_summary}

Please return ONLY the final result in valid JSON (no extra explanation)."""

        # 2. Get the AIPipe token from environment or config
        aipipe_token = os.environ.get("AIPIPE_TOKEN")
        if not aipipe_token:
            return jsonify({"error": "AIPIPE_TOKEN is not set in environment variables"}), 401

        # 3. Send request to AIPipe endpoint
        headers = {
            "Authorization": f"Bearer {aipipe_token}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "openai/gpt-4o-mini",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        }

        llm_response = requests.post(
            "https://aipipe.org/openrouter/v1/chat/completions",
            headers=headers,
            data=json.dumps(payload)
        )

        if llm_response.status_code != 200:
            return jsonify({"error": "Failed to fetch from GPT", "details": llm_response.text}), 500

        # Extract and sanitize the response
        raw_output = llm_response.json()["choices"][0]["message"]["content"]

        # Remove Markdown formatting if present
        if raw_output.strip().startswith("```"):
            raw_output = raw_output.strip()
            if raw_output.startswith("```json"):
                raw_output = raw_output[len("```json"):].strip()
            elif raw_output.startswith("```"):
                raw_output = raw_output[len("```"):].strip()
            if raw_output.endswith("```"):
                raw_output = raw_output[:-3].strip()

        # Validate that it's proper JSON
        try:
            parsed = json.loads(raw_output)
        except json.JSONDecodeError as e:
            return jsonify({"error": "Invalid JSON from LLM", "raw_output": raw_output, "details": str(e)}), 500

        # Validate base64 image size
        for img_field in ["network_graph", "degree_histogram"]:
            if img_field in parsed:
                base64_data = parsed[img_field].strip()
                if not base64_data.startswith("data:image/png;base64,"):
                    base64_data = f"data:image/png;base64,{base64_data}"
                if not is_valid_base64_image(base64_data):
                    raise ValueError(f"Invalid or too small base64 image for: {img_field}")
                parsed[img_field] = base64_data

        # Enforce time limit
        if time.time() - start_time > 170:
            return jsonify({"error": "Request took too long"}), 500

        return json.dumps(parsed), 200, {'Content-Type': 'application/json'}

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)

