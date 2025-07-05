from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import pandas as pd
import random
import uuid

app = Flask(__name__, template_folder="frontend", static_folder="static")
CORS(app)

score_model_path = "./Score_Evaluation_Model/scoring_distilbert_model"
score_tokenizer = DistilBertTokenizer.from_pretrained(score_model_path)
score_model = DistilBertForSequenceClassification.from_pretrained(score_model_path)
score_model.eval()

question_df = pd.read_csv("interview_questions1.csv").dropna()
question_df['role'] = question_df['role'].str.strip().str.lower().str.replace(" ", "_")

sessions = {}

def evaluate_answer(answer):
    inputs = score_tokenizer(answer, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = score_model(**inputs)
        score = outputs.logits.squeeze().item()
    score = round(max(0, min(100, score)))
    if score >= 85:
        feedback = "Excellent"
    elif score >= 60:
        feedback = "Good"
    elif score >= 35:
        feedback = "Average"
    elif score >= 15:
        feedback = "Weak"
    else:
        feedback = "Poor"
    return score, feedback

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/job_profile.html')
def job_profile():
    return render_template('job_profile.html')

@app.route('/interviewbot.html')
def interview_page():
    role = request.args.get('role', 'software_engineer').lower()
    return render_template('interviewbot.html', role=role)

@app.route("/api/interview", methods=["POST"])
def interview():
    data = request.get_json()
    user_msg = data.get("message", "").strip()
    role = data.get("role", "").strip().lower().replace(" ", "_")
    session_id = data.get("session_id")

    if not session_id:
        filtered_questions = question_df[question_df['role'] == role]['question'].tolist()
        if len(filtered_questions) == 0:
            return jsonify({"error": f"No questions found for role '{role}'."}), 400
        selected_questions = random.sample(filtered_questions, min(10, len(filtered_questions)))
        session_id = str(uuid.uuid4())
        sessions[session_id] = {
            "role": role,
            "questions": selected_questions,
            "asked": [],
            "answers": [],
            "scores": [],
            "feedbacks": [],
            "current_index": 0,
            "interview_done": False
        }
        first_question = selected_questions[0]
        sessions[session_id]["asked"].append(first_question)
        sessions[session_id]["current_index"] = 1
        return jsonify({
            "session_id": session_id,
            "question": first_question,
            "message": f"Interview started for {role.replace('_', ' ')} role."
        })

    session = sessions.get(session_id)
    if not session:
        return jsonify({"error": "Invalid session ID"}), 400

    if session["interview_done"]:
        return jsonify({
            "session_id": session_id,
            "message": "Interview completed!",
            "details": list(zip(
                session["asked"],
                session["answers"],
                session["scores"],
                session["feedbacks"]
            ))
        })

    if user_msg and session["current_index"] > 0:
        score, feedback = evaluate_answer(user_msg)
        session["answers"].append(user_msg)
        session["scores"].append(score)
        session["feedbacks"].append(feedback)
    else:
        score, feedback = None, None

    if session["current_index"] < len(session["questions"]):
        next_question = session["questions"][session["current_index"]]
        session["asked"].append(next_question)
        session["current_index"] += 1
        return jsonify({
            "session_id": session_id,
            "question": next_question,
            "score": score,
            "feedback": feedback,
            "message": f"Your last answer scored {score}/100 — {feedback}." if score is not None else ""
        })
    else:
        session["interview_done"] = True
        avg_score = sum(session["scores"]) / len(session["scores"]) if session["scores"] else 0
        return jsonify({
            "session_id": session_id,
            "score": round(avg_score, 1),
            "message": f"Interview completed! Average score: {round(avg_score, 1)}",
            "details": list(zip(
                session["asked"],
                session["answers"],
                session["scores"],
                session["feedbacks"]
            ))
        })

if __name__ == "__main__":
    app.run(debug=True)
