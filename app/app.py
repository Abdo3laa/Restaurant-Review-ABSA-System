import io
import csv
import json
import pandas as pd
from flask import Flask, request, jsonify, render_template, send_file
from absa import analyze  # your existing pipeline

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB limit


# ── helpers ──────────────────────────────────────────────────────────────────

def rows_to_absa(rows: list[str]) -> list[dict]:
    """Run ABSA on a list of sentence strings and return flat records."""
    records = []
    for sentence in rows:
        sentence = sentence.strip()
        if not sentence:
            continue
        result = analyze(sentence)
        for r in result["results"]:
            records.append({
                "sentence": sentence,
                "aspect":   r["aspect"],
                "sentiment": r["sentiment"],
            })
    return records


# ── routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze/text", methods=["POST"])
def analyze_text():
    """Single-text endpoint → returns aspects + sentiments as JSON."""
    data = request.get_json(force=True)
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "No text provided."}), 400

    result = analyze(text)
    return jsonify(result)


@app.route("/analyze/file", methods=["POST"])
def analyze_file():
    """
    File-upload endpoint.
    Form fields:
      - file        : uploaded .csv or .txt
      - has_header  : "true" / "false"   (CSV only)
      - column      : column name or index for the review text (CSV only)
    Returns JSON with records + chart data.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    f = request.files["file"]
    filename = f.filename.lower()
    has_header = request.form.get("has_header", "true").lower() == "true"
    col_choice = request.form.get("column", "").strip()  # name or 0-based index

    # ── read rows ──
    try:
        if filename.endswith(".csv"):
            content = f.read().decode("utf-8", errors="replace")
            reader = csv.reader(io.StringIO(content))

            if has_header:
                header = next(reader, [])
                # resolve column
                if col_choice == "" or col_choice == "0":
                    col_idx = 0
                elif col_choice.lstrip("-").isdigit():
                    col_idx = int(col_choice)
                elif col_choice in header:
                    col_idx = header.index(col_choice)
                else:
                    col_idx = 0
                rows = [row[col_idx] for row in reader if len(row) > col_idx]
            else:
                # no header – col_choice is a 0-based index string
                col_idx = int(col_choice) if col_choice.lstrip("-").isdigit() else 0
                rows = [row[col_idx] for row in reader if len(row) > col_idx]

        elif filename.endswith(".txt"):
            content = f.read().decode("utf-8", errors="replace")
            rows = [line for line in content.splitlines() if line.strip()]

        else:
            return jsonify({"error": "Only .csv and .txt files are supported."}), 400

    except Exception as e:
        return jsonify({"error": f"Could not read file: {str(e)}"}), 400

    if not rows:
        return jsonify({"error": "No review rows found in the file."}), 400

    # ── run ABSA ──
    records = rows_to_absa(rows)

    if not records:
        return jsonify({"error": "No aspects were extracted from the file."}), 400

    df = pd.DataFrame(records)

    # ── chart data ──
    # 1. aspect distribution (top-20)
    aspect_counts = (
        df.groupby("aspect")
          .size()
          .sort_values(ascending=False)
          .head(20)
    )

    # 2. sentiment breakdown per aspect (same top-20)
    top_aspects = aspect_counts.index.tolist()
    asc = (
        df[df["aspect"].isin(top_aspects)]
        .groupby(["aspect", "sentiment"])
        .size()
        .unstack(fill_value=0)
        .reindex(top_aspects)
    )
    for col in ["positive", "negative", "neutral"]:
        if col not in asc.columns:
            asc[col] = 0

    aspect_sentiment_data = {
        "aspects":   top_aspects,
        "positive":  asc["positive"].tolist(),
        "negative":  asc["negative"].tolist(),
        "neutral":   asc["neutral"].tolist(),
    }

    return jsonify({
        "records":              records,
        "aspect_counts":        aspect_counts.to_dict(),
        "aspect_sentiment_data": aspect_sentiment_data,
        "total_sentences":      len(rows),
        "total_aspects":        len(records),
    })


@app.route("/download/csv", methods=["POST"])
def download_csv():
    """Turn records JSON back into a downloadable CSV."""
    data = request.get_json(force=True)
    records = data.get("records", [])
    if not records:
        return jsonify({"error": "No records to export."}), 400

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=["sentence", "aspect", "sentiment"])
    writer.writeheader()
    writer.writerows(records)
    output.seek(0)

    return send_file(
        io.BytesIO(output.getvalue().encode("utf-8")),
        mimetype="text/csv",
        as_attachment=True,
        download_name="absa_results.csv",
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)