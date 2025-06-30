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
import random
import re

# (Keep your existing role_questions dictionary here)

def start_interview():
    role_dropdown = widgets.Dropdown(options=list(role_questions.keys()), description='Job Role:')
    start_button = widgets.Button(description='Start Interview')
    interview_container = widgets.Output()  # Container to hold the current interview

    def on_start(b):
        # Clear any previous interview
        interview_container.clear_output()

        selected_role = role_dropdown.value
        questions = role_questions[selected_role]

        with interview_container:
            ask_questions(selected_role, questions)

    start_button.on_click(on_start)

    # Display the controls and empty container
    display(widgets.VBox([role_dropdown, start_button, interview_container]))

def ask_questions(role, questions):
    current_index = 0
    question_output = widgets.Output()
    input_box = widgets.Textarea(placeholder='Your answer...')
    submit_button = widgets.Button(description='Submit Answer')
    total_score = {'score': 0}

    # Container for this interview session
    interview_session = widgets.VBox([question_output, input_box, submit_button])

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
                # Disable further submissions
                submit_button.disabled = True
                input_box.disabled = True

    with question_output:
        print(questions[current_index])

    display(interview_session)
    submit_button.on_click(on_submit)

# (Keep your existing score_answer function here)

start_interview()

model.save_pretrained('./fine_tuned_model')
tokenizer.save_pretrained('./fine_tuned_model')

!zip -r fine_tuned_model.zip fine_tuned_model

