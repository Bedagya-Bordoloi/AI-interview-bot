from flask import Flask, request, jsonify
import uuid
import re

app = Flask(__name__)

from flask_cors import CORS
CORS(app)

role_questions = {
    "Software Engineer": [
        "Tell me about a challenging project you worked on.",
        "Describe your experience with data structures and algorithms.",
        "How do you handle technical debt in a project?",
        "Explain the concept of RESTful APIs.",
        "Describe a time you had to work with a difficult team member."
    ],
    "Data Scientist": [
        "Explain the difference between supervised and unsupervised learning.",
        "Describe a time you used machine learning to solve a real-world problem.",
        "How do you handle missing data in a dataset?",
        "Explain the concept of overfitting and how to avoid it.",
        "Describe your experience with data visualization tools."
    ],
    "Product Manager": [
        "How do you prioritize features for a product?",
        "Describe a time you had to say no to a stakeholder.",
        "How do you measure the success of a product?",
        "Explain the concept of a Minimum Viable Product (MVP).",
        "Describe a time you had to pivot on a product strategy."
    ],
    "Frontend Developer": [
        "What is the virtual DOM?",
        "Explain CSS specificity.",
        "How do you optimize website performance?",
        "Describe your experience with JavaScript frameworks.",
        "What are web accessibility best practices?"
    ],
    "Backend Developer": [
        "Explain REST vs GraphQL.",
        "What is database normalization?",
        "How do you ensure API security?",
        "Describe a time you scaled a backend system.",
        "How do you handle caching in your backend?"
    ],
    "AI Engineer": [
        "What is a neural network?",
        "How do you select a model architecture?",
        "Describe a project using deep learning.",
        "What is transfer learning?",
        "Explain how backpropagation works."
    ],
    "DevOps Engineer": [
        "What is CI/CD?",
        "Describe a time you automated infrastructure.",
        "How do you monitor system health?",
        "Explain containerization.",
        "What is infrastructure as code?"
    ],
    "UX Designer": [
        "What is user-centered design?",
        "How do you conduct user research?",
        "Describe your design process.",
        "What tools do you use for prototyping?",
        "Explain accessibility in design."
    ],
    "Cybersecurity Analyst": [
        "What is penetration testing?",
        "How do you secure a network?",
        "Describe the CIA triad.",
        "What is threat modeling?",
        "How do you respond to a security breach?"
    ],
    "Cloud Architect": [
        "What is cloud scalability?",
        "How do you design fault-tolerant systems?",
        "Describe your experience with AWS/GCP/Azure.",
        "What is serverless computing?",
        "How do you optimize cloud costs?"
    ]
}

role_keywords = {
    "Software Engineer": ["project", "code", "bug", "algorithm", "design"],
    "Data Scientist": ["data", "model", "analysis", "statistics", "predict"],
    "Product Manager": ["stakeholder", "roadmap", "kpi", "mvp", "launch"],
    "Frontend Developer": ["html", "css", "javascript", "dom", "responsive"],
    "Backend Developer": ["api", "database", "cache", "server", "scalable"],
    "AI Engineer": ["model", "neural", "train", "deep", "learning"],
    "DevOps Engineer": ["ci/cd", "pipeline", "automation", "deployment", "infrastructure"],
    "UX Designer": ["user", "prototype", "wireframe", "research", "interface"],
    "Cybersecurity Analyst": ["vulnerability", "threat", "encryption", "attack", "breach"],
    "Cloud Architect": ["cloud", "scalable", "aws", "gcp", "serverless"]
}

sessions = {}

def score_answer(role, answer):
    keywords = role_keywords.get(role, [])
    score = 0
    num_words = len(answer.strip().split())
    score += 30 if num_words >= 10 else num_words * 3
    ans_lower = answer.lower()
    for kw in keywords:
        if re.search(r'\b' + re.escape(kw) + r'\b', ans_lower):
            score += 10
    return min(score, 40)

def interviewer_response(user_input):
    user_input = user_input.lower()
    if "your name" in user_input or "who are you" in user_input or "what are you" in user_input:
        return "I'm an AI interviewer designed to simulate technical interviews."
    elif "score" in user_input or "why did i get" in user_input:
        return "Your score is based on answer length and use of key technical terms."
    elif "feedback" in user_input or "improve" in user_input or "tip" in user_input:
        return "To improve, try to provide detailed answers with technical keywords relevant to the role."
    elif "interview" in user_input:
        return "This interview was designed to mimic a real-world role-based assessment."
    elif "get a job" in user_input or "land a job" in user_input:
        return "You can apply through job boards like LinkedIn, Angellist, Internshala, or your network."
    elif "skills" in user_input:
        return "Key skills depend on the role. For tech roles, focus on data structures, projects, and communication."
    elif "opportunities" in user_input:
        return "Consider internships, freelance projects, and open source contributions to build your profile."
    elif "how are you" in user_input:
        return "I'm great, thank you! I'm here to help you prepare for interviews."
    elif "what are you doing" in user_input:
        return "I'm evaluating your interview responses and helping you improve."
    elif "aspect" in user_input or "responsibility" in user_input:
        return "Each role involves problem-solving, communication, and relevant technical or design skills."
    elif "strength" in user_input or "weakness" in user_input:
        return "Identify strengths relevant to your role and weaknesses you are actively improving."
    elif "ai" in user_input and "future" in user_input:
        return "AI is expected to greatly impact fields like healthcare, finance, transportation, and education."
    elif "resume" in user_input:
        return "Keep your resume concise, tailored to the job role, and highlight measurable achievements."
    elif "project" in user_input:
        return "Highlight your best projects with problems solved, tools used, and impact."
    elif "question" in user_input and "interviewer" in user_input:
        return "Ask about team culture, tech stack, challenges, or learning opportunities at the company."
    elif "revolutionize" in user_input or "change the industry" in user_input:
        return "AI will revolutionize the tech industry by automating tasks, personalizing services, and enhancing decision-making."
    else:
        return "AI Interviewer: That's an interesting question! I'll make a note of it for future improvements."

@app.route('/api/interview', methods=['POST'])
def interview():
    data = request.json
    msg = data.get('message', '')
    role = data.get('role')
    session_id = data.get('session_id')

    if not role or role not in role_questions:
        return jsonify({"error": "Invalid or missing role"}), 400

    if not session_id or session_id not in sessions:
        session_id = str(uuid.uuid4())
        sessions[session_id] = {
            "role": role,
            "current_q": 0,
            "answers": [],
            "score": 0,
            "phase": "interview"
        }
        question = role_questions[role][0]
        sessions[session_id]["current_q"] = 1
        return jsonify({
            "session_id": session_id,
            "question": question,
            "progress": f"1/{len(role_questions[role])}"
        })

    session = sessions[session_id]
    role = session["role"]

    if session["phase"] == "interview":
        if session["current_q"] > 0:
            session["answers"].append(msg)
            session["score"] += score_answer(role, msg)
        if session["current_q"] < len(role_questions[role]):
            question = role_questions[role][session["current_q"]]
            session["current_q"] += 1
            return jsonify({
                "session_id": session_id,
                "question": question,
                "progress": f"{session['current_q']}/{len(role_questions[role])}"
            })
        else:
            session["phase"] = "final_qa"
            # Only set this ONCE, when transitioning to final_qa
            # session["final_qa_shown"] = False
            return jsonify({
                "session_id": session_id,
                "score": min(session["score"], 100),
                "message": "Interview completed! You can now ask the AI interviewer any question."
            })
    elif session["phase"] == "final_qa":
        ai_reply = interviewer_response(msg)
        return jsonify({
            "session_id": session_id,
            "ai_response": ai_reply
        })

@app.route('/')
def home():
    return jsonify({"message": "AI Interviewer API is running"})

if __name__ == '__main__':
    app.run(debug=True)