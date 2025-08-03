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
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from io import BytesIO
    import base64

    url = "https://en.wikipedia.org/wiki/List_of_highest-grossing_films"
    tables = pd.read_html(url)

    # Find the correct table by checking for 'Rank' and 'Worldwide gross' in its columns
    for table in tables:
        if 'Rank' in table.columns and 'Worldwide gross' in table.columns:
            df = table.copy()
            break
    else:
        return {"error": "Expected table not found on Wikipedia"}

    # Keep only the needed columns
    df = df[['Rank', 'Title', 'Worldwide gross', 'Year']]

    # Clean and convert
    df['Worldwide gross'] = df['Worldwide gross'].replace(r'[\$,]', '', regex=True)
    df['Worldwide gross'] = pd.to_numeric(df['Worldwide gross'], errors='coerce')
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df['Rank'] = pd.to_numeric(df['Rank'], errors='coerce')
    df = df.dropna()

    # Q1: Count films grossing > $2B before 2000
    q1 = df[(df['Worldwide gross'] > 2_000_000_000) & (df['Year'] < 2000)].shape[0]

    # Q2: First film that grossed over $1.5B
    over_1_5_billion = df[df['Worldwide gross'] > 1_500_000_000]
    earliest = over_1_5_billion.loc[over_1_5_billion['Year'].idxmin(), 'Title'] if not over_1_5_billion.empty else None

    # Q3: Correlation between Rank and Gross
    q3 = df[['Rank', 'Worldwide gross']].corr().iloc[0, 1]

    # Q4: Scatter plot
    plt.figure(figsize=(6, 4))
    sns.scatterplot(data=df, x="Rank", y="Worldwide gross")
    sns.regplot(data=df, x="Rank", y="Worldwide gross", scatter=False, color="red", linestyle='dotted')
    plt.title("Rank vs Gross")
    plt.xlabel("Rank")
    plt.ylabel("Gross ($)")
    buffer = BytesIO()
    plt.savefig(buffer, format="png", bbox_inches="tight", dpi=80)
    plt.close()
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    img_uri = f"data:image/png;base64,{img_base64}"

    return jsonify([q1, earliest, round(q3, 6), img_uri])


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)


