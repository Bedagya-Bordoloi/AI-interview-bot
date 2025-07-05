# 🚀 AIVice - AI-Powered Interview Assistant 🤖

An AI-powered tool designed to simulate job interviews, providing users with realistic practice and feedback to improve their interviewing skills. It uses **JOB-SPECIFIC QUESTIONS** from custom datasets, evaluates user responses, and provides a **SCORE OUT OF 100**.

## 📖 Introduction

The **AIVice** is a **FLASK-BASED APPLICATION** that helps job seekers prepare for technical interviews. It leverages custom-made datasets of interview questions and a **PRE-TRAINED DISTILBERT MODEL** to evaluate user responses and provide constructive feedback. The bot simulates a real interview experience, asking questions tailored to the selected job role and offering a score and feedback on the user's answers. This tool aims to make the interview preparation process more efficient and effective.

## ✨ Features

*   **🎯 JOB-SPECIFIC INTERVIEWS:** Chooses questions based on the selected job role (e.g., Software Engineer, Data Scientist, etc.)
*   **💬 REALISTIC SIMULATION:** Provides a conversational interface that mimics a real interview
*   **🧠 AI-POWERED EVALUATION:** Uses a **DISTILBERT MODEL** to evaluate the quality of user responses
*   **📊 SCORING AND FEEDBACK:** Provides a score out of 100 and constructive feedback on each answer
*   **🔄 SESSION MANAGEMENT:** Maintains a session to track progress and provide a cohesive interview experience
*   **🖥️ FRONTEND INTERFACE:** User-friendly interface to easily start mock interviews
*   **📚 DATABASE OF QUESTIONS:** 5000+ questions and answers for seamless model training

## 🛠️ Technologies Used

*   **BACKEND:**
    *   Python
    *   Flask: Web framework for creating the API
    *   Transformers: Library for using pre-trained models (DistilBERT)
    *   PyTorch: Deep learning framework
    *   Pandas: Data analysis library
    *   Flask-CORS: Handles Cross-Origin Resource Sharing (CORS)
*   **FRONTEND:**
    *   HTML
    *   CSS
    *   JavaScript
*   **MODEL TRAINING:**
    *   Transformers
    *   Datasets
    *   Evaluate
    *   Scikit-learn
    *   Weights & Biases (W&B) for experiment tracking
*   **OTHER:**
    *   Beautiful Soup
    *   Requests

## 🚀 Installation

1.  **CLONE THE REPOSITORY:**
    ```bash
    git clone https://github.com/Bedagya-Bordoloi/AI-interview-bot.git
    cd AI-interview-bot
    ```

2.  **INSTALL DEPENDENCIES:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **RUN THE APPLICATION:**
    ```bash
    python app.py
    ```

4.  **ACCESS THE APPLICATION:**
    Open your web browser and go to `http://127.0.0.1:5000/`

## 📂 Code Structure

*   `app.py`: Main Flask application file
*   `Score_Evaluation_Model/`: Contains the fine-tuned DistilBERT model files
*   `frontend/`: Contains all frontend files
    *   `index.html`: Main landing page
    *   `job_profile.html`: Job profile selection page
    *   `interviewbot.html`: Interactive interview page
*   `AI Interview Question Answer Dataset Creation.ipynb`: Notebook for dataset creation
*   `AI Interview Dataset Generator.ipynb`: Generates mock interview conversations
*   `Score Evaluation Model WandB.ipynb`: Trains the score evaluation model

## 🤖 Training the Score Evaluation Model

The score evaluation model uses a **FINE-TUNED DISTILBERT MODEL** to assess answer quality. Use the `Score Evaluation Model WandB.ipynb` notebook to train your own model:

1. Prepare training data with question/answer pairs and scores
2. Run the notebook to fine-tune the model
3. Save the model in `Score_Evaluation_Model/scoring_distilbert_model/`

## 👥 Contributors

- [Krishanu](https://github.com/Krishanu)
- [Bedagya-Bordoloi](https://github.com/Bedagya-Bordoloi)

## 🔮 Future Enhancements

*   Implement more sophisticated AI models for response evaluation
*   Add support for more job roles and interview question datasets
*   Improve the user interface with real-time feedback
*   Enable personalized interview question generation
