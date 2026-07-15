# ai-study-assistant
 from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from openai import OpenAI
import os
import webbrowser
import threading

load_dotenv()

app = Flask(__name__)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if GROQ_API_KEY:
    client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")
    MODEL_NAME = "llama-3.1-8b-instant"
else:
    client = OpenAI(api_key=OPENAI_API_KEY)
    MODEL_NAME = "gpt-4o-mini"

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    try:
        user_message = request.json.get("message")
        if not user_message:
            return jsonify({"error": "No message provided."}), 400

        if not (GROQ_API_KEY or OPENAI_API_KEY):
            return jsonify({"error": "No API key found. Add GROQ_API_KEY (free) or OPENAI_API_KEY."}), 500

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a helpful AI study assistant."},
                {"role": "user", "content": user_message}
            ]
        )

        reply = response.choices[0].message.content
        return jsonify({"reply": reply})
    except Exception as e:
        error_text = str(e)
        lower_error = error_text.lower()

        if "insufficient_quota" in lower_error or "quota" in lower_error or "429" in lower_error:
            return jsonify({
                "error": "OpenAI quota is exhausted. Please add billing/credits to your OpenAI account or switch to another API key/provider."
            }), 429

        print("/chat error:", type(e).__name__, error_text)
        return jsonify({"error": error_text}), 500


def open_browser():
    webbrowser.open("http://127.0.0.1:5000/")


if __name__ == "__main__":
    # ❗ IMPORTANT FIX: only run browser in main thread
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        threading.Timer(1.0, open_browser).start()

    app.run(debug=True)
