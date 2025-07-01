from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, TrainerCallback, pipeline
from datasets import load_dataset, Dataset
from huggingface_hub import login
import re
import math
import matplotlib.pyplot as plt

# Authenticate with Hugging Face Hub
login(token="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")

# Load dataset with forced redownload
dataset = load_dataset('ikenna1234/ai_interviewer_dataset', use_auth_token=True, download_mode="force_redownload")

train_data = dataset['train']
train_texts = ["[QUESTION] " + q['question'] + " [ANSWER] " + q['answer'] + " <|endoftext|>" for q in train_data]

dataset = Dataset.from_dict({'text': train_texts})
split_dataset = dataset.train_test_split(test_size=0.1)
train_dataset = split_dataset['train']
eval_dataset = split_dataset['test']

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.add_special_tokens({'pad_token': '<|pad|>', 'eos_token': '<|endoftext|>'})
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.resize_token_embeddings(len(tokenizer))

def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)

tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_eval = eval_dataset.map(tokenize_function, batched=True)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=5,
    per_device_train_batch_size=4,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir='./logs',
    learning_rate=5e-5,
    warmup_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss"
)

eval_losses = []

class EvalLossLogger(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics and "eval_loss" in metrics:
            eval_losses.append(metrics["eval_loss"])

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    callbacks=[EvalLossLogger()]
)

trainer.train()

eval_results = trainer.evaluate()
eval_loss = eval_results["eval_loss"]
try:
    perplexity = math.exp(eval_loss)
except OverflowError:
    perplexity = float("inf")

print(f"Final Perplexity: {perplexity}")

plt.plot(eval_losses)
plt.xlabel('Epoch')
plt.ylabel('Eval Loss')
plt.title('Evaluation Loss per Epoch')
plt.savefig('eval_loss_plot.png')
plt.show()

model.save_pretrained('./fine_tuned_model')
tokenizer.save_pretrained('./fine_tuned_model')

generator = pipeline('text-generation', model=model, tokenizer=tokenizer)

def clean_response(text):
    cleaned = text.split('<|endoftext|>')[0]
    cleaned = re.sub(r'[^.!?]*$', '', cleaned.strip())
    return cleaned.strip()

prompt = "[QUESTION] Tell me about yourself. [ANSWER]"
output = generator(prompt, max_length=100, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id)
filtered_answer = clean_response(output[0]['generated_text'])
print("Sample Answer:", filtered_answer)

!pip install huggingface_hub
!huggingface-cli login
!huggingface-cli download ikenna1234/ai_interviewer_dataset --repo-type dataset

import glob
import pandas as pd
from datasets import Dataset

# Find the parquet file
parquet_paths = glob.glob('/root/.cache/huggingface/hub/datasets--ikenna1234--ai_interviewer_dataset/**/*.parquet', recursive=True)
parquet_file = parquet_paths[0]

# Load with pandas
df = pd.read_parquet(parquet_file)

# Convert to HF Dataset
hf_dataset = Dataset.from_pandas(df)

# Split train/eval
split = hf_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split['train']
eval_dataset = split['test']

# Proceed with tokenization and training using train_dataset and eval_dataset

# Install dependencies
!pip install transformers datasets matplotlib

from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, TrainerCallback, pipeline
from datasets import load_dataset, Dataset
import re
import math
import matplotlib.pyplot as plt

# Load dataset from local Parquet file
dataset = load_dataset('parquet', data_files='/root/.cache/huggingface/hub/datasets--ikenna1234--ai_interviewer_dataset/snapshots/d084a4ef400b1b93b436137d2feba8d4debb63be/train-00000-of-00001.parquet')

# Prepare text data with special tokens
train_data = dataset['train']
train_texts = ["[QUESTION] " + q['question'] + " [ANSWER] " + q['answer'] + " <|endoftext|>" for q in train_data]

# Build dataset object
dataset = Dataset.from_dict({'text': train_texts})
split_dataset = dataset.train_test_split(test_size=0.1)
train_dataset = split_dataset['train']
eval_dataset = split_dataset['test']

# Tokenizer & model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.add_special_tokens({'pad_token': '<|pad|>', 'eos_token': '<|endoftext|>'})
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.resize_token_embeddings(len(tokenizer))

def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)

# Tokenize
tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_eval = eval_dataset.map(tokenize_function, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=5,
    per_device_train_batch_size=4,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir='./logs',
    learning_rate=5e-5,
    warmup_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss"
)

# Callback to track eval loss
eval_losses = []

class EvalLossLogger(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics and "eval_loss" in metrics:
            eval_losses.append(metrics["eval_loss"])

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    callbacks=[EvalLossLogger()]
)

# Train
trainer.train()

# Evaluate
results = trainer.evaluate()
eval_loss = results["eval_loss"]
try:
    perplexity = math.exp(eval_loss)
except OverflowError:
    perplexity = float("inf")
print(f"Final Perplexity: {perplexity}")

# Plot eval loss
plt.plot(eval_losses)
plt.xlabel('Epoch')
plt.ylabel('Eval Loss')
plt.title('Evaluation Loss per Epoch')
plt.savefig('eval_loss_plot.png')
plt.show()

# Save model
model.save_pretrained('./fine_tuned_model')
tokenizer.save_pretrained('./fine_tuned_model')

# Inference
generator = pipeline('text-generation', model=model, tokenizer=tokenizer)

def clean_response(text):
    cleaned = text.split('<|endoftext|>')[0]
    cleaned = re.sub(r'[^.!?]*$', '', cleaned.strip())
    return cleaned.strip()

prompt = "[QUESTION] Tell me about yourself. [ANSWER]"
output = generator(prompt, max_length=100, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id)
filtered_answer = clean_response(output[0]['generated_text'])
print("Sample Answer:", filtered_answer)

from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, TrainerCallback, pipeline
from datasets import load_dataset, Dataset
import re
import math
import matplotlib.pyplot as plt

# Use the dataset loaded in the previous cell
dataset = hf_dataset

if len(dataset) == 0:
    raise ValueError("Loaded dataset is empty! Check your dataset source or token permissions.")

print("Sample data:", dataset[0])

# Prepare text data with special tokens
train_texts = ["[QUESTION] " + d['input'] + " [ANSWER] " + d['output'] + " <|endoftext|>" for d in dataset]

# Split dataset
split_dataset = Dataset.from_dict({'text': train_texts}).train_test_split(test_size=0.1)
train_dataset = split_dataset['train']
eval_dataset = split_dataset['test']

# Tokenizer & model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.add_special_tokens({'pad_token': '<|pad|>', 'eos_token': '<|endoftext|>'})
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.resize_token_embeddings(len(tokenizer))

def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)

tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_eval = eval_dataset.map(tokenize_function, batched=True)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=5,
    per_device_train_batch_size=4,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir='./logs',
    learning_rate=5e-5,
    warmup_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss"
)

eval_losses = []

class EvalLossLogger(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics and "eval_loss" in metrics:
            eval_losses.append(metrics["eval_loss"])

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    callbacks=[EvalLossLogger()]
)

trainer.train()

eval_results = trainer.evaluate()
eval_loss = eval_results.get("eval_loss", None)

if eval_loss is not None:
    try:
        perplexity = math.exp(eval_loss)
    except OverflowError:
        perplexity = float("inf")
    print(f"Final Perplexity: {perplexity}")
else:
    print("Eval loss not available.")

plt.plot(eval_losses)
plt.xlabel('Epoch')
plt.ylabel('Eval Loss')
plt.title('Evaluation Loss per Epoch')
plt.savefig('eval_loss_plot.png')
plt.show()

model.save_pretrained('./fine_tuned_model')
tokenizer.save_pretrained('./fine_tuned_model')

generator = pipeline('text-generation', model=model, tokenizer=tokenizer)

def clean_response(text):
    cleaned = text.split('<|endoftext|>')[0]
    cleaned = re.sub(r'[^.!?]*$', '', cleaned.strip())
    return cleaned.strip()

prompt = "[QUESTION] Tell me about yourself. [ANSWER]"
output = generator(prompt, max_length=100, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id)
filtered_answer = clean_response(output[0]['generated_text'])
print("Sample Answer:", filtered_answer)

from google.colab import userdata
userdata.get('WANDB_API_KEY')

from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, TrainerCallback, pipeline
from datasets import load_dataset, Dataset
import re
import math
import matplotlib.pyplot as plt

# Use the dataset loaded in the previous cell
dataset = hf_dataset

if len(dataset) == 0:
    raise ValueError("Loaded dataset is empty! Check your dataset source or token permissions.")

print("Sample data:", dataset[0])

# Prepare text data with special tokens
train_texts = ["[QUESTION] " + d['input'] + " [ANSWER] " + d['output'] + " <|endoftext|>" for d in dataset]

# Split dataset
split_dataset = Dataset.from_dict({'text': train_texts}).train_test_split(test_size=0.1)
train_dataset = split_dataset['train']
eval_dataset = split_dataset['test']

# Tokenizer & model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.add_special_tokens({'pad_token': '<|pad|>', 'eos_token': '<|endoftext|>'})
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.resize_token_embeddings(len(tokenizer))

def tokenize_function(examples):
    tokenized_inputs = tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)
    tokenized_inputs['labels'] = tokenized_inputs['input_ids'] # Add labels here
    return tokenized_inputs

tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_eval = eval_dataset.map(tokenize_function, batched=True)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=5,
    per_device_train_batch_size=4,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir='./logs',
    learning_rate=5e-5,
    warmup_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    report_to=["wandb"]
)

eval_losses = []

class EvalLossLogger(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics and "eval_loss" in metrics:
            eval_losses.append(metrics["eval_loss"])

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    callbacks=[EvalLossLogger()]
)

trainer.train()

eval_results = trainer.evaluate()
eval_loss = eval_results["eval_loss"]
try:
    perplexity = math.exp(eval_loss)
except OverflowError:
    perplexity = float("inf")

print(f"Final Perplexity: {perplexity}")

plt.plot(eval_losses)
plt.xlabel('Epoch')
plt.ylabel('Eval Loss')
plt.title('Evaluation Loss per Epoch')
plt.savefig('eval_loss_plot.png')
plt.show()

model.save_pretrained('./fine_tuned_model')
tokenizer.save_pretrained('./fine_tuned_model')

generator = pipeline('text-generation', model=model, tokenizer=tokenizer)

def clean_response(text):
    cleaned = text.split('<|endoftext|>')[0]
    cleaned = re.sub(r'[^.!?]*$', '', cleaned.strip())
    return cleaned.strip()

prompt = "[QUESTION] Tell me about yourself. [ANSWER]"
output = generator(prompt, max_length=100, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id)
filtered_answer = clean_response(output[0]['generated_text'])
print("Sample Answer:", filtered_answer)

from IPython.display import display, clear_output
import ipywidgets as widgets
import re

# Define the role_questions dictionary with 10 roles and their questions
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

# Define role-specific keywords
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

# Scoring system
def score_answer(role, answer):
    keywords = role_keywords.get(role, [])
    score = 0
    num_words = len(answer.strip().split())
    score += min(num_words * 3, 30)

    ans_lower = answer.lower()
    for kw in keywords:
        if re.search(r'\b' + re.escape(kw) + r'\b', ans_lower):
            score += 10

    return min(score, 40)

# Respond to follow-up question
def ai_response(user_input):
    user_input = user_input.lower()
    
    if "your name" in user_input or "who are you" in user_input:
        return "I'm an AI interviewer designed to simulate technical interviews."
    elif "what are you doing" in user_input or "purpose" in user_input:
        return "I'm here to evaluate your responses and simulate a realistic interview experience."
    elif "score" in user_input or "why did i get" in user_input:
        return "Your score is based on the length and relevance of your answers, including technical keywords."
    elif "feedback" in user_input or "how can i improve" in user_input:
        return "To improve, focus on providing structured, detailed answers using job-specific terminology."
    elif "interview" in user_input:
        return "This interview is designed to mimic a real-world role-based assessment."
    elif "tips" in user_input or "suggestion" in user_input:
        return "Be clear, concise, and use real examples from your experience. Practice common questions."
    elif "aspects" in user_input or "important in this role" in user_input:
        return "Important aspects include technical skill, problem-solving ability, and communication."
    elif "next step" in user_input or "what next" in user_input:
        return "After this, you can review your answers and continue practicing to improve your score."
    elif "how did i do" in user_input or "performance" in user_input:
        return "You did well! With more detailed answers and keyword use, your score would increase."
    elif "strength" in user_input:
        return "Your strength is shown when you give specific, thoughtful responses."
    elif "weakness" in user_input:
        return "Work on elaborating your answers and using more domain-specific terminology."
    elif "how to prepare" in user_input:
        return "Review the fundamentals of the job role, and practice answering questions aloud."
    elif "career advice" in user_input or "career path" in user_input:
        return "Explore projects, internships, or certifications aligned with your interests."
    elif "opportunities" in user_input or "what should i explore" in user_input:
        return "Look into freelance work, open source contributions, and internship roles to build experience."
    elif "resume" in user_input or "cv" in user_input:
        return "Tailor your resume for the job role, emphasizing measurable achievements."
    elif "common mistake" in user_input:
        return "A common mistake is giving generic answers. Always relate to your experience."
    elif "resources" in user_input or "where to learn" in user_input:
        return "Try using Coursera, edX, and YouTube channels like freeCodeCamp for learning."
    elif "projects" in user_input:
        return "Work on personal or open source projects related to your job preference to stand out."
    elif "mock interview" in user_input:
        return "You can try platforms like Pramp or Interviewing.io for live mock interviews."
    elif "recommendation" in user_input:
        return "I recommend exploring job boards like LinkedIn, AngelList, and GitHub Jobs."

    else:
        return "That's an interesting question! I'll make a note of it for future improvements."



# Start asking interview questions
def ask_questions(role, questions):
    current_index = 0
    total_score = {'score': 0}

    question_output = widgets.Output()
    input_box = widgets.Textarea(placeholder='Your answer...')
    submit_button = widgets.Button(description='Submit Answer')
    display_area = widgets.VBox([question_output, input_box, submit_button])
    display(display_area)

    def on_submit(b):
        nonlocal current_index
        ans = input_box.value
        total_score['score'] += score_answer(role, ans)
        current_index += 1
        input_box.value = ''

        question_output.clear_output()
        with question_output:
            if current_index < len(questions):
                print(questions[current_index])
            else:
                print(f"Interview completed. Total Score: {min(total_score['score'], 100)}/100")
                input_box.layout.display = 'none'
                submit_button.layout.display = 'none'
                followup_section()

    def followup_section():
        followup_input = widgets.Textarea(placeholder="Ask anything to the interviewer...")
        followup_button = widgets.Button(description="Ask")
        followup_output = widgets.Output()

        def on_followup_click(btn):
            followup_output.clear_output()
            with followup_output:
                response = ai_response(followup_input.value.strip())
                print(f"AI Interviewer: {response}")

        display(widgets.VBox([
            widgets.Label("Would you like to ask anything to the interviewer?"),
            followup_input, followup_button, followup_output
        ]))

        followup_button.on_click(on_followup_click)

    with question_output:
        print(questions[current_index])
    submit_button.on_click(on_submit)

# Start the interview setup
def start_interview():
    role_dropdown = widgets.Dropdown(options=list(role_questions.keys()), description='Job Role:')
    start_button = widgets.Button(description='Start Interview')
    output_box = widgets.Output()

    def on_start(b):
        output_box.clear_output()
        role = role_dropdown.value
        questions = role_questions[role]
        with output_box:
            ask_questions(role, questions)

    start_button.on_click(on_start)
    display(widgets.VBox([role_dropdown, start_button, output_box]))

start_interview()


