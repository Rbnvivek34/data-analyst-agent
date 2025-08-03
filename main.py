from flask import Flask, request, jsonify
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import numpy as np
import os

app = Flask(__name__)

@app.route("/")
def home():
    return "✅ Data Analyst Agent is live. Use POST /api/ with a question file."

@app.route("/api/", methods=["POST"])
def handle_request():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    try:
        task = file.read().decode("utf-8").lower()
    except Exception as e:
        return jsonify({"error": "Failed to decode uploaded file"}), 400

    try:
        if "highest-grossing films" in task:
            return analyze_movies_dynamic()
        else:
            return jsonify(["Task not supported yet", "", "", ""])
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def analyze_movies_dynamic():
    url = "https://en.wikipedia.org/wiki/List_of_highest-grossing_films"
    tables = pd.read_html(url)
    df = tables[0]

    # Drop multi-level column headers if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(0)

    # Drop columns like 'Ref', 'Notes', etc.
    df = df.loc[:, ~df.columns.str.contains('Ref|Notes|Source', case=False, na=False)]

    # Handle common cases dynamically
    column_map = {
        4: ['Rank', 'Title', 'Worldwide gross', 'Year'],
        5: ['Rank', 'Title', 'Worldwide gross', 'Year', 'Peak']
    }

    if df.shape[1] in column_map:
        df.columns = column_map[df.shape[1]]
    else:
        raise ValueError(f"Unexpected number of columns: {df.shape[1]}")

    # Optional Peak column
    if 'Peak' not in df.columns:
        df['Peak'] = np.nan

    # Clean and convert
    df['Worldwide gross'] = df['Worldwide gross'].replace(r'[\$,]', '', regex=True).astype(float)
    df['Peak'] = df['Peak'].replace(r'[\$,]', '', regex=True).astype(float)
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df['Rank'] = pd.to_numeric(df['Rank'], errors='coerce')

    # Q1: Films > $2B before 2000
    q1 = int(df[(df['Worldwide gross'] > 2_000_000_000) & (df['Year'] < 2000)].shape[0])

    # Q2: First film > $1.5B
    over_1_5b = df[df['Worldwide gross'] > 1_500_000_000]
    q2 = over_1_5b.loc[over_1_5b['Year'].idxmin(), 'Title'] if not over_1_5b.empty else "N/A"

    # Q3: Correlation between Rank and Peak (fallback to Gross if Peak missing)
    if df['Peak'].notna().sum() > 2:
        q3_val = df[['Rank', 'Peak']].corr().iloc[0, 1]
    else:
        q3_val = df[['Rank', 'Worldwide gross']].corr().iloc[0, 1]
    q3 = round(float(q3_val), 6)

    # Q4: Plot
    plt.figure(figsize=(6, 4))
    sns.scatterplot(data=df, x="Rank", y="Worldwide gross")
    sns.regplot(data=df, x="Rank", y="Worldwide gross", scatter=False, color="red", linestyle="dotted")
    plt.title("Rank vs Worldwide Gross")
    plt.xlabel("Rank")
    plt.ylabel("Gross ($)")
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=80)
    plt.close()
    buf.seek(0)
    img_uri = "data:image/png;base64," + base64.b64encode(buf.read()).decode("utf-8")

    return jsonify([q1, q2, q3, img_uri])

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
