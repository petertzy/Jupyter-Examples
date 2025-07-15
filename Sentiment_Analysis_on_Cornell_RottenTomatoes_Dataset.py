#!/usr/bin/env python
# coding: utf-8

# In[42]:


import os

def load_polarity_dataset(root_dir):
    texts = []
    labels = []
    for label_type in ['pos', 'neg']:
        dir_path = os.path.join(root_dir, label_type)
        for fname in os.listdir(dir_path):
            if fname.endswith('.txt'):
                file_path = os.path.join(dir_path, fname)
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
                    texts.append(text)
                    labels.append(1 if label_type == 'pos' else 0)
    return texts, labels

texts, labels = load_polarity_dataset('./polarity')

print(f'Loaded {len(texts)} samples')
print('Example text #1:', texts[0])
print('Label for example #1:', labels[0])


# In[43]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# 1. Vectorize the texts
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(texts)
y = labels

# 2. Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train the model
clf = MultinomialNB()
clf.fit(X_train, y_train)

# 4. Predict and evaluate
y_pred = clf.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print(classification_report(y_test, y_pred))


# In[39]:


import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from datasets import Dataset

# Assume texts and labels are already prepared
# texts = [...]
# labels = [...]

# 1. Split training and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

# 2. Load model name
MODEL_NAME = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# 3. Tokenize texts
train_encodings = tokenizer(
    train_texts, truncation=True, padding=True, max_length=256
)
val_encodings = tokenizer(
    val_texts, truncation=True, padding=True, max_length=256
)

# 4. Convert to Hugging Face Dataset
train_dataset = Dataset.from_dict({
    'input_ids': train_encodings['input_ids'],
    'attention_mask': train_encodings['attention_mask'],
    'labels': train_labels
})

val_dataset = Dataset.from_dict({
    'input_ids': val_encodings['input_ids'],
    'attention_mask': val_encodings['attention_mask'],
    'labels': val_labels
})

# 5. Load pretrained model
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=2
)

# 6. Manually define accuracy computation function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.tensor(logits).argmax(dim=-1)
    labels = torch.tensor(labels)
    correct = (predictions == labels).sum().item()
    total = labels.size(0)
    accuracy = correct / total
    return {"accuracy": accuracy}

# 7. Set training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy"
)

# 8. Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# 9. Train the model
trainer.train()

# 10. Evaluate the model
trainer.evaluate()


# In[ ]:




