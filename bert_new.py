import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForTokenClassification, BertConfig, AdamW
from transformers import get_linear_schedule_with_warmup
from seqeval.metrics import f1_score, accuracy_score, classification_report
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
import numpy as np
from tqdm import tqdm

# Load the NER dataset
dataset = load_dataset('conll2003')

# Tokenize the inputs
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples['tokens'], truncation=True, is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples['ner_tags']):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(label[word_idx])
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs['labels'] = labels
    return tokenized_inputs

tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)

# Define the model
num_labels = len(dataset['train'].features['ner_tags'].feature.names)
model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)

# Define the training arguments
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=8,   # batch size per device during training
    per_device_eval_batch_size=16,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
)

# Define the data collator
from transformers import DataCollatorForTokenClassification
data_collator = DataCollatorForTokenClassification(tokenizer)

# Initialize the Trainer
trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=tokenized_datasets['train'],         # training dataset
    eval_dataset=tokenized_datasets['validation'],     # evaluation dataset
    data_collator=data_collator,
    compute_metrics=lambda p: get_metrics(p, tokenized_datasets['validation']),
)

# Function to calculate metrics during evaluation
def get_metrics(p, labels):
    predictions = np.argmax(p.predictions, axis=2)
    true_predictions = [[dataset['train'].features['ner_tags'].feature.names[p_i] for p_i, l_i in zip(p_row, l_row) if l_i != -100]
                        for p_row, l_row in zip(predictions, labels)]
    true_labels = [[dataset['train'].features['ner_tags'].feature.names[l_i] for l_i in l_row if l_i != -100]
                   for l_row in labels]
    return {
        "accuracy_score": accuracy_score(true_labels, true_predictions),
        "f1_score": f1_score(true_labels, true_predictions)
    }

# Train the model
trainer.train()

# Evaluate the model
eval_results = trainer.evaluate()

# Predict entities in a sentence
def predict_entities(sentence):
    tokens = tokenizer.tokenize(sentence)
    inputs = tokenizer(sentence, return_tensors="pt")
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=2)
    return [(token, dataset['train'].features['ner_tags'].feature.names[prediction]) for token, prediction in zip(tokens, predictions[0].tolist())]

sentence = "Apple is located in California"
entities = predict_entities(sentence)
print(entities)
