from flask import Flask, request, jsonify
import os
import tempfile
import mimetypes
import time
import requests
import json
import base64
import imghdr

app = Flask(__name__)

def is_base64_image(data: str) -> bool:
    try:
        decoded = base64.b64decode(data, validate=True)
        img_type = imghdr.what(None, decoded)
        return img_type in ["png", "jpeg", "gif"]
    except Exception:
        return False

@app.route("/")
def home():
    return "Data Analyst Agent is live. Use POST /api/ with a question file."

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
        system_prompt = """You are a data analyst assistant. Given a task description and uploaded files, your job is to:
- Decide what operations (analysis, web scraping, visualization, etc.) are needed.
- Extract, clean, analyze, and visualize the data.
- Generate results in the required format (JSON, base64 plots, etc.).
- If the question asks for plots, encode them as base64 image URIs under 100,000 bytes.
- Return structured data only, like a JSON array or object, depending on the prompt.
- Do not explain your reasoning, just return the answer."""

        file_summary = "\n".join([f"- {k}: {mimetypes.guess_type(k)[0] or 'Unknown type'}" for k in file_data])

        user_prompt = f"""Task:\n{question_text}

Uploaded Files:
{file_summary}

Please return ONLY the final result in valid JSON (no extra explanation)."""

        aipipe_token = os.environ.get("AIPIPE_TOKEN")
        if not aipipe_token:
            return jsonify({"error": "AIPIPE_TOKEN is not set in environment variables"}), 401

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

        # Validate JSON
        try:
            parsed = json.loads(raw_output)
        except json.JSONDecodeError as e:
            return jsonify({"error": "Invalid JSON from LLM", "raw_output": raw_output, "details": str(e)}), 500

        # Wrap base64 image fields in data URI and check size
        max_size_bytes = 100_000
        for key, val in parsed.items():
            if isinstance(val, str):
                base64_data = val.strip()

                # Extract base64 content if already wrapped
                if base64_data.startswith("data:image/png;base64,"):
                    b64_str = base64_data[len("data:image/png;base64,"):]
                else:
                    b64_str = base64_data

                if is_base64_image(b64_str):
                    decoded_bytes = base64.b64decode(b64_str)
                    if len(decoded_bytes) > max_size_bytes:
                        return jsonify({
                            "error": f"Image field '{key}' exceeds size limit of {max_size_bytes} bytes"
                        }), 400

                    # Only wrap if not already wrapped
                    if not base64_data.startswith("data:image/png;base64,"):
                        parsed[key] = f"data:image/png;base64,{b64_str}"
                else:
                    parsed[key] = val

        # Enforce time limit
        if time.time() - start_time > 170:
            return jsonify({"error": "Request took too long"}), 500

        return json.dumps(parsed), 200, {'Content-Type': 'application/json'}

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        # Clean up temp files
        for path in file_data.values():
            try:
                os.remove(path)
            except OSError:
                pass

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
