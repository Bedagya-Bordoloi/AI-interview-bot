# AI Interview Bot

An AI-powered tool designed to simulate job interviews, providing users with realistic practice and feedback to improve their interviewing skills.  It uses job-specific questions from custom datasets, evaluates user responses, and provides a score out of 100.

## Introduction

The AI Interview Bot is a Flask-based application that helps job seekers prepare for technical interviews. It leverages custom-made datasets of interview questions and a pre-trained DistilBERT model to evaluate user responses and provide constructive feedback. The bot simulates a real interview experience, asking questions tailored to the selected job role and offering a score and feedback on the user's answers. This tool aims to make the interview preparation process more efficient and effective.

## Features

*   *Job-Specific Interviews:*  Chooses questions based on the selected job role (e.g., Software Engineer, Data Scientist, etc.).
*   *Realistic Simulation:* Provides a conversational interface that mimics a real interview.
*   *AI-Powered Evaluation:* Uses a DistilBERT model to evaluate the quality of user responses.
*   *Scoring and Feedback:*  Provides a score out of 100 and constructive feedback on each answer.
*   *Session Management:*  Maintains a session to track progress and provide a cohesive interview experience.
*   *Frontend Interface:* User-friendly interface to easily start mock interviews for their relevant job titles
*   *Database of questions:* Datasets of 5000+ questions and answers for seamless model training

## Technologies Used

*   *Backend:*
    *   Python
    *   Flask: Web framework for creating the API.
    *   Transformers: Library for using pre-trained models (DistilBERT).
    *   PyTorch: Deep learning framework.
    *   Pandas: Data analysis library for handling interview questions.
    *   Flask-CORS: Handles Cross-Origin Resource Sharing (CORS) for API requests.
*   *Frontend:*
    *   HTML
    *   CSS
    *   JavaScript
*   *Model Training & Evaluation:*
    *   Transformers
    *   Datasets
    *   Evaluate
    *   Scikit-learn
    *   Weights & Biases (W&B) for experiment tracking
*   *Other*
    *   Beautiful Soup
    *   Requests

## Installation

1.  *Clone the repository:*

    bash
    git clone https://github.com/Bedagya-Bordoloi/AI-interview-bot.git
    cd AI-interview-bot
    

2.  *Create a virtual environment (recommended):*

    bash
    python3 -m venv venv
    source venv/bin/activate  # On Linux/macOS
    venv\Scripts\activate.bat # On Windows
    

3.  *Install the required packages:*

    bash
    pip install -r requirements.txt  # If you have a requirements.txt file
    # OR install manually:
    pip install flask flask_cors transformers torch pandas scikit-learn beautifulsoup4 requests
    

4.  *Download necessary model (DistilBERT).*

    *   Ensure that the score_model_path in app.py points to the correct location of the downloaded DistilBERT model files.  This path defaults to  ./Score_Evaluation_Model/scoring_distilbert_model
    *   You can download the scoring_distilbert_model and its associated tokenizer using the training script provided in the section below or by manually downloading the model from your W&B artifacts if you trained using the provided notebook

5.  *Create Datasets:*

    The project needs a csv file called interview_questions1.csv. You can use the automated dataset creation method by running the given notebook named - AI Interview Question Answer Dataset Creation - to create one of your own.
    This CSV file contains interview questions, and ideally contains columns for role (job role) and question (the interview question). Ensure the roles are in lowercase with spaces replaced by underscores (e.g., software_engineer).

## Usage

1.  *Run the Flask application:*

    bash
    python app.py
    

2.  *Access the application:*

    Open your web browser and go to http://127.0.0.1:5000/ or the address printed in the console when you run app.py.

3.  *Navigate the Site*

    You should be able to see the main homepage. From there, click on "Get Started" to access the job profiles page and select the job title you want to have a mock interview for.

## Code Structure

*   app.py:  The main Flask application file. It handles routing, API endpoints, and the interview logic.  It also contains the code for loading the DistilBERT model and evaluating user responses.
*   Score_Evaluation_Model/: This folder is intended to contain the fine-tuned DistilBERT model files:
    *   scoring_distilbert_model/: The actual model files and the tokenizer.
*   frontend/: Contains all frontend files:
    *   index.html: Main landing page with an introduction and links to other pages.
    *   job_profile.html: Page for selecting job profiles for interviews.
    *   interviewbot.html:  The interactive interview page with the chat interface.
    *   static/:  Folder containing static assets like CSS, images, etc.
*   AI Interview Question Answer Dataset Creation.ipynb:
    *   This notebook handles the automated dataset creation of relevant questions and answers by scraping sites like Medium, GeeksForGeeks, etc.
*   AI Interview Dataset Generator.ipynb:
    *   This notebook handles the automated generation of mock interview conversations for various technical roles
*   Score Evaluation Model WandB.ipynb:
    *   This notebook contains the necessary codes to train the score evalutaion model using the huggingface datasets module

## Training the Score Evaluation Model

The score evaluation model uses a fine-tuned DistilBERT model to assess the quality of user answers. You can fine-tune this model using your own data to improve its accuracy.
To use your model in the app, ensure that the score_model_path in app.py points to the correct location.

1.  *Prepare Training Data:*  You'll need a dataset of interview question/answer pairs with associated scores (e.g., 0-100). The dataset should have columns for question, ideal_answer, user_answer, and score.
2.  **Use the Score Evaluation Model WandB.ipynb notebook:**  This notebook provides a complete workflow for fine-tuning the DistilBERT model:
    *   *Install Dependencies:* Includes the necessary pip install commands.
    *   *Load Data:*  Shows how to load your data into a Pandas DataFrame and convert it to a Hugging Face Dataset.
    *   *Tokenization:*  Uses the DistilBERT tokenizer to prepare the text data for the model.
    *   *Model Training:*  Sets up and runs the training process using the Hugging Face Trainer. Includes Weights & Biases integration for experiment tracking.
    *   *Evaluation:*  Calculates metrics like Mean Squared Error (MSE) and Root Mean Squared Error (RMSE).
    *   *Saving the Model:*  Saves the fine-tuned model and tokenizer.
    *   *Logging results with W&B:* Shows distribution of data and saves a scatter plot showing the true vs predicted scores on the validation set, alongside calculating eval metrics like MSE and RMSE.
3.  *Copy Model Files:* After training, copy the contents of the scoring_distilbert_model directory into the Score_Evaluation_Model/scoring_distilbert_model directory in your project.

## API Endpoints

*   /: Renders the main index.html page.
*   /job_profile.html: Renders the job profile selection page.
*   /interviewbot.html: Renders the interview bot interface, passing the job role as a query parameter.
*   /api/interview (POST):
    *   Receives user messages and returns interview questions, scores, and feedback.
    *   Requires a JSON payload with message, role, and session_id.
    *   Returns a JSON response containing either the next question, the interview results, or an error message.

## Frontend Details

The frontend consists of three main HTML files:

*   index.html: The landing page, providing an overview of the application and links to other pages.
*   job_profile.html:  Allows users to select their desired job role before starting the interview. It dynamically generates the list of job profiles based on the available data.
*   interviewbot.html: The main interview interface, featuring:
    *   A chat window to display questions and answers.
    *   An input field for users to type their responses.
    *   A "Send" button to submit the responses.
    *   JavaScript code to handle the chat interface, send messages to the backend, and display the results.

The CSS styles provide a visually appealing and user-friendly interface.

## Contributing

Contributions are welcome! Feel free to submit pull requests to improve the project.

## Future Enhancements

*   Implement more sophisticated AI models for response evaluation (e.g., using larger language models).
*   Add support for more job roles and interview question datasets.
*   Improve the user interface with features like real-time feedback and progress tracking.
*   Enable personalized interview question generation based on user skills and experience.
