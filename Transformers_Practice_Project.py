from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import pandas as pd
from datasets import load_dataset

# df = load_dataset("imdb")

exercise_sentences = [
    "The movie was fantastic, I really enjoyed it!",
    "I did not like the new restaurant, food was bad.",
    "The book was okay, some parts were interesting.",
    "This product exceeded my expectations!",
    "The service was disappointing and slow."
]


##Tokenization
tokensizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize(batch):
    return tokensizer(batch[exercise_sentences], padding = True, truncation = True, max_length = 120)
tokenized_text = exercise_sentences.map(tokenize, batched = True, )

##Classifier
classifier = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased')

#Training Arguments

training_args = TrainingArguments(
    output_dir= "/training_output",
    per_device_eval_batch_size= 8,
    per_device_train_batch_size=8,
    num_train_epochs=1,
    logging_steps=100,
    eval_strategy="steps"
)

trainer = Trainer(
    model=classifier,
    args=training_args,
    train_dataset=tokenized_text['train'].shuffle(seed = 42).select(range(200)),
    eval_dataset=tokenized_text['test'].shuffle(seed=42).select(range(200))
)

trainer.train() 