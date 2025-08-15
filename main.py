from flask import Flask, request, jsonify
import os
import tempfile
import mimetypes
import time
import requests
import json
import base64
import filetype
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import io

app = Flask(__name__)

def is_base64_image(data: str) -> bool:
    try:
        decoded = base64.b64decode(data, validate=True)
        kind = filetype.guess(decoded)
        return kind is not None and kind.mime.startswith("image/")
    except Exception:
        return False

def encode_plot_to_base64(fig, max_size_kb=100):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img_bytes = buf.getvalue()
    buf.close()
    if len(img_bytes) > max_size_kb * 1024:
        raise ValueError("Image exceeds 100kB limit")
    return base64.b64encode(img_bytes).decode("utf-8")

def analyze_csv_and_answer(csv_path, question_text):
    df = pd.read_csv(csv_path)
    if df.shape[1] < 2:
        raise ValueError("CSV must contain at least two columns for edges.")

    G = nx.Graph()
    G.add_edges_from(df.iloc[:, :2].values)

    edge_count = G.number_of_edges()
    degrees = dict(G.degree())
    highest_degree_node = max(degrees, key=degrees.get)
    avg_degree = sum(degrees.values()) / len(degrees)
    density = nx.density(G)

    # Attempt to extract dynamic node names
    lower_q = question_text.lower()
    source_node, target_node = None, None
    for node in G.nodes():
        node_lower = node.lower()
        if f"path between {node_lower}" in lower_q:
            source_node = node
        elif f"and {node_lower}" in lower_q:
            target_node = node

    try:
        if source_node and target_node:
            shortest_path = nx.shortest_path_length(G, source=source_node, target=target_node)
        else:
            shortest_path = -1
    except Exception:
        shortest_path = -1

    # Network graph
    fig1, ax1 = plt.subplots()
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color="lightblue", edge_color="gray", node_size=1000, ax=ax1)
    network_b64 = encode_plot_to_base64(fig1)
    plt.close(fig1)

    # Degree histogram
    fig2, ax2 = plt.subplots()
    degree_values = list(degrees.values())
    unique_degrees = sorted(set(degree_values))
    counts = [degree_values.count(k) for k in unique_degrees]
    ax2.bar(unique_degrees, counts, color="green")
    ax2.set_xlabel("Degree")
    ax2.set_ylabel("Count")
    ax2.set_title("Degree Histogram")
    histogram_b64 = encode_plot_to_base64(fig2)
    plt.close(fig2)

    # Format path key dynamically
    if source_node and target_node:
        path_key = f"shortest_path_{source_node.lower()}_{target_node.lower()}"
    else:
        path_key = "shortest_path"

    return {
        "edge_count": edge_count,
        "highest_degree_node": highest_degree_node,
        "average_degree": round(avg_degree, 4),
        "density": round(density, 4),
        path_key: shortest_path,
        "network_graph": network_b64,
        "degree_histogram": histogram_b64
    }

@app.route("/")
def home():
    return "Data Analyst Agent is live. Use POST /api/ with a question file."

@app.route("/api/", methods=["POST"])
def handle_request():
    start_time = time.time()

    if "questions.txt" not in request.files:
        return jsonify({"error": "questions.txt file is required"}), 400

    question_text = request.files["questions.txt"].read().decode("utf-8")

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
        # If this is the known network graph question, handle locally
        if "edge_count" in question_text.lower() and any(f.endswith(".csv") for f in file_data):
            csv_file = next((path for name, path in file_data.items() if name.endswith(".csv")), None)
            output = analyze_csv_and_answer(csv_file, question_text)
        else:
            # Otherwise, use LLM as fallback
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

            if raw_output.strip().startswith("```"):
                raw_output = raw_output.strip()
                if raw_output.startswith("```json"):
                    raw_output = raw_output[len("```json"):].strip()
                elif raw_output.startswith("```"):
                    raw_output = raw_output[len("```"):].strip()
                if raw_output.endswith("```"):
                    raw_output = raw_output[:-3].strip()

            try:
                output = json.loads(raw_output)
            except json.JSONDecodeError as e:
                return jsonify({"error": "Invalid JSON from LLM", "raw_output": raw_output, "details": str(e)}), 500

        # Validate and wrap base64 image fields
        max_size_bytes = 100_000
        for key, val in output.items():
            if isinstance(val, str):
                b64_str = val.strip()
                #if b64_str.startswith("data:image/png;base64,"):
                    #b64_str = b64_str[len("data:image/png;base64,"):]
                if is_base64_image(b64_str):
                    decoded_bytes = base64.b64decode(b64_str)
                    if len(decoded_bytes) > max_size_bytes:
                        return jsonify({"error": f"Image field '{key}' exceeds size limit"}), 400
                    output[key] = b64_str  # ✅ Keep as raw base64 without prefix


        if time.time() - start_time > 170:
            return jsonify({"error": "Request took too long"}), 500

        return json.dumps(output), 200, {'Content-Type': 'application/json'}

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        for path in file_data.values():
            try:
                os.remove(path)
            except OSError:
                pass

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)





