import os, re, pickle
import numpy as np
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
import nltk
from nltk.corpus import stopwords
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
import mysql.connector
import requests
from google import genai

# LOAD ENV 
load_dotenv()

# NLTK 
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

app = Flask(__name__)

# GEMINI CLIENT 
try:
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    print("Gemini AI Connected ✅")
except Exception as e:
    print("Gemini ERROR:", e)
    client = None

#  DB CONNECT 
try:
    db = mysql.connector.connect(
    host=os.getenv("DB_HOST"),
    port=int(os.getenv("DB_PORT", 26175)),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    database=os.getenv("DB_NAME")
)
    cursor = db.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id          INT AUTO_INCREMENT PRIMARY KEY,
            statement   TEXT NOT NULL,
            final_label VARCHAR(20),
            confidence  FLOAT,
            ai_verdict  VARCHAR(20),
            ai_analysis TEXT,
            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    db.commit()
    print("DB Connected ✅")
except Exception as e:
    print("DB ERROR:", e)
    db, cursor = None, None

# CONFIG 
MAX_LEN   = 200
MODEL_DIR = "models"

#  LOAD MODEL 
print("Loading model...")
model = tf.keras.models.load_model(f"{MODEL_DIR}/bilstm_model.keras")

with open(f"{MODEL_DIR}/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open(f"{MODEL_DIR}/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

print("Model loaded ✅")
print("Classes:", le.classes_)  # ['FAKE' 'TRUE']


def clean_text(text, category=""):
    # training used: category + statement combined
    combined = (str(category) + " " + str(text)).strip()
    combined = combined.lower()
    combined = re.sub(r'http\S+|www\S+', '', combined)
    combined = re.sub(r'[^a-z\s]', ' ', combined)
    combined = ' '.join(
        w for w in combined.split()
        if w not in stop_words and len(w) > 2
    )
    return combined.strip()

# ── ML PREDICT 
def ml_predict(text, category=""):
    cleaned    = clean_text(text, category)
    seq        = tokenizer.texts_to_sequences([cleaned])
    padded     = pad_sequences(seq, maxlen=MAX_LEN, padding='post')

    probs      = model.predict(padded, verbose=0)[0]
    idx        = int(np.argmax(probs))
    confidence = round(float(probs[idx]) * 100, 2)
    raw_label  = le.classes_[idx]  # 'FAKE' or 'TRUE'

    # Log all probabilities
    all_probs = {le.classes_[i]: round(float(probs[i]) * 100, 2) for i in range(len(le.classes_))}
    print(f"ML → Raw: {raw_label} | Conf: {confidence}% | All: {all_probs}")

    # Map to display label
    if confidence < 65:
        final = "SUSPICIOUS"
    elif raw_label.upper() == "TRUE":
        final = "REALISTIC"
    else:
        final = "MISLEADING"

    return raw_label, final, confidence

# ── GEMINI AI ANALYSIS — always runs, gives 4-5 line analysis
def rule_based_analysis(text, ml_label, ml_confidence):
    t = text.lower()
    red_flags  = []
    good_signs = []

    caps_words = [w for w in text.split() if w.isupper() and len(w) > 3]
    if caps_words:
        red_flags.append(f"ALL CAPS words like '{caps_words[0]}'")
    if text.count('!') >= 2:
        red_flags.append(f"{text.count('!')} exclamation marks used")
    for word in ["shocking", "secret", "exposed", "conspiracy",
                 "they don't want", "share before", "deleted",
                 "whistleblower", "truth", "reveals", "mind-control"]:
        if word in t:
            red_flags.append(f"emotional trigger word '{word}'")
    for v in ["sources say", "reportedly", "some say"]:
        if v in t:
            red_flags.append(f"vague phrase '{v}'")
    if not any(c.isdigit() for c in text):
        red_flags.append("no specific numbers or dates")

    for word in ["minister", "government", "official", "announced",
                 "according to", "percent", "crore", "parliament",
                 "supreme court", "police", "army", "isro", "rbi"]:
        if word in t:
            good_signs.append(word)

    if len(red_flags) >= 2:
        verdict = "MISLEADING"
        flag_str = "; ".join(red_flags[:3])
        analysis = (
            f"This text displays {len(red_flags)} red flags of misinformation: {flag_str}. "
            f"The writing uses emotional and sensational language designed to provoke sharing rather than inform. "
            f"No credible named sources, official agencies, or verifiable facts are present in the text. "
            f"Patterns like 'share before deleted' and 'they don't want you to know' are classic misinformation tactics. "
            f"This claim should be verified from official government websites or trusted news sources before believing."
        )
    elif len(good_signs) >= 2:
        verdict = "REALISTIC"
        sign_str = ", ".join(good_signs[:3])
        analysis = (
            f"This text contains credibility signals including: {sign_str}. "
            f"The language is factual and informative rather than emotionally manipulative. "
            f"Specific verifiable details align with real journalism standards. "
            f"No major red flags such as ALL CAPS or conspiracy language were detected. "
            f"ML model also predicts {ml_label} with {ml_confidence}% confidence."
        )
    elif ml_confidence >= 80:
        verdict = "REALISTIC" if ml_label.upper() == "TRUE" else "MISLEADING"
        analysis = (
            f"ML model classifies this as {ml_label} with high confidence ({ml_confidence}%). "
            f"Text pattern matches characteristics of {'real' if ml_label.upper() == 'TRUE' else 'fake'} news. "
            f"{'Key verifiable details and professional language support this classification.' if ml_label.upper() == 'TRUE' else 'Lack of verifiable sources raises reliability concerns.'} "
            f"{'No strong misinformation indicators were found in the text.' if ml_label.upper() == 'TRUE' else 'The overall presentation style raises concerns about reliability.'} "
            f"Cross-checking with official sources is always recommended."
        )
    else:
        verdict = "SUSPICIOUS"
        analysis = (
            f"This text does not have enough clear signals to confidently classify. "
            f"ML model gives moderate confidence of {ml_confidence}% indicating uncertainty. "
            f"Some elements appear credible but important verifiable details are missing. "
            f"The claim cannot be confirmed or denied based on text analysis alone. "
            f"Please cross-check with official sources before believing or sharing."
        )
    return verdict, analysis


def generate_ai_reason(text, ml_label, ml_confidence):
    if client:
        try:
            prompt = f"""You are a senior fake news analyst specializing in Indian news.

Analyze this news text:
\"\"\"{text[:1500]}\"\"\"

ML prediction: {ml_label} (confidence: {ml_confidence}%)
TRUE=real news, FAKE=misinformation.

Check RED FLAGS: ALL CAPS, exclamation marks!!!, emotional words (shocking/secret/exposed), vague sources, urgency phrases (share before deleted), missing names/dates.
Check CREDIBILITY: named officials, specific numbers/dates, official agencies, neutral language.

Respond EXACTLY:
VERDICT: REALISTIC
ANALYSIS: 4-5 specific sentences quoting actual words from the text.

VERDICT must be: REALISTIC, MISLEADING, or SUSPICIOUS"""

            response = client.models.generate_content(
                model="models/gemini-2.0-flash-lite",
                contents=prompt
            )
            output = response.text.strip()
            print(f"Gemini OK: {output[:100]}")

            verdict  = "SUSPICIOUS"
            analysis = ""
            lines    = output.split('\n')
            for i, line in enumerate(lines):
                line = line.strip()
                if line.startswith("VERDICT:"):
                    v = line.replace("VERDICT:", "").strip().upper()
                    if   "REALISTIC" in v: verdict = "REALISTIC"
                    elif "MISLEAD"   in v: verdict = "MISLEADING"
                    else:                  verdict = "SUSPICIOUS"
                elif line.startswith("ANALYSIS:"):
                    parts = [line.replace("ANALYSIS:", "").strip()]
                    for nl in lines[i+1:]:
                        if nl.strip(): parts.append(nl.strip())
                    analysis = " ".join(parts).strip()
            if not analysis:
                analysis = output
            return verdict, analysis

        except Exception as e:
            print("Gemini ERROR:", str(e)[:100])

    # Always fallback to rule-based
    return rule_based_analysis(text, ml_label, ml_confidence)

# ── GOOGLE FACT CHECK ─────────────────────────────────────
def get_fact_check(text):
    try:
        api_key = os.getenv("FACT_API_KEY")

        #  Safety checks
        if not api_key:
            print("FACT API KEY missing ❌")
            return []

        if not text or len(text.strip()) < 20:
            return []

        url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
        params = {
            "query": text[:200],
            "key": api_key,
            "pageSize": 3
        }

        res = requests.get(url, params=params, timeout=5)

        #  HTTP error handling
        if res.status_code != 200:
            print("FACT API ERROR:", res.status_code, res.text)
            return []

        data = res.json()
        claims = []

        if "claims" in data:
            for c in data["claims"][:2]:
                review = c.get("claimReview", [{}])[0]

                claims.append({
                    "text": c.get("text", "N/A"),
                    "rating": review.get("textualRating", "Unknown"),
                    "source": review.get("publisher", {}).get("name", "Unknown")
                })

        return claims

    except Exception as e:
        print("FACT API ERROR:", e)
        return []
# ── SAVE TO DB ────────────────────────────────────────────
def save_db(text, label, confidence, ai_verdict, ai_analysis):
    if not cursor:
        return
    try:
        cursor.execute(
            """INSERT INTO history
               (statement, final_label, confidence, ai_verdict, ai_analysis)
               VALUES (%s, %s, %s, %s, %s)""",
            (text[:2000], label, confidence, ai_verdict, ai_analysis[:2000])
        )
        db.commit()
    except Exception as e:
        print("DB INSERT ERROR:", e)

# ── ROUTES ────────────────────────────────────────────────
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/history')
def history():
    rows = []
    if cursor:
        try:
            cursor.execute(
                """SELECT id, statement, final_label, confidence,
                          ai_verdict, ai_analysis, created_at
                   FROM history ORDER BY id DESC LIMIT 100"""
            )
            rows = cursor.fetchall()
        except Exception as e:
            print("History fetch error:", e)
    return render_template('history.html', data=rows)

@app.route('/api/detect', methods=['POST'])
def detect():
    data     = request.get_json()
    text     = data.get('text', '').strip()
    category = data.get('category', '').strip()

    # Guard: empty
    if not text:
        return jsonify({
            "final_label": "SUSPICIOUS",
            "confidence":  0,
            "ml_label":    "UNCERTAIN",
            "ai_verdict":  "SUSPICIOUS",
            "ai_analysis": "No text was provided.",
            "fact_check":  []
        })

    # Guard: too short
    if len(text.split()) < 6:
        return jsonify({
            "final_label": "SUSPICIOUS",
            "label":       "SUSPICIOUS",
            "confidence":  0,
            "ml_label":    "UNCERTAIN",
            "ai_verdict":  "SUSPICIOUS",
            "ai_analysis": (
                "Input bahut chhota hai. "
                "Please ek poora news article ya kam se kam ek complete sentence paste karo "
                "taaki sahi detection ho sake."
            ),
            "reason":     "Input too short.",
            "fact_check": []
        })

    # 1. ML prediction
    ml_label, final_label, confidence = ml_predict(text, category)

    # 2. Gemini AI — always runs
    ai_verdict, ai_analysis = generate_ai_reason(text, ml_label, confidence)

    # 3. Fact check
    fact_data = get_fact_check(text)

    # 4. Combined verdict
    if final_label == "SUSPICIOUS":
        combined = ai_verdict           # ML unsure → trust Gemini
    elif ai_verdict == "SUSPICIOUS":
        combined = final_label          # Gemini unsure → trust ML
    elif final_label == ai_verdict:
        combined = final_label          # both agree → use it
    else:
        combined = "SUSPICIOUS"         # disagree → flag it

    # 5. Save to DB
    save_db(text, combined, confidence, ai_verdict, ai_analysis)

    print(f"ML: {ml_label} ({confidence}%) | Gemini: {ai_verdict} | Final: {combined}")

    return jsonify({
        "final_label": combined,
        "label":       combined,
        "confidence":  confidence,
        "ml_label":    ml_label,
        "ai_verdict":  ai_verdict,
        "ai_analysis": ai_analysis,
        "reason":      ai_analysis,
        "fact_check":  fact_data
    })

# ── RUN ───────────────────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=True)