from transformers import GPT2Tokenizer, TFGPT2Model, DataCollatorForLanguageModeling
from datasets.load import load_dataset

import tensorflow as tf

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = TFGPT2Model.from_pretrained('gpt2')

raw_data = load_dataset("imdb")

def tokenize_function(examples):
    result = tokenizer(examples["text"])
    if tokenizer.is_fast:
        result["word_ids"] = [result.word_ids(i) for i in range(len(result["inputs_ids"]))]
    return result

tokenized_data = raw_data.map(
    tokenize_function, batched=True, remove_columns=["text", "label"]
)

def group_texts(examples):
    chunk_size = 128
    # Concatenate all texts
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    # Compute length of concatenated texts
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the last chunk if it's smaller than chunk_size
    total_length = (total_length // chunk_size) * chunk_size
    # Split by chunks of max_len
    result = {
        k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
        for k, t in concatenated_examples.items()
    }
    # Create a new labels column
    result["labels"] = result["input_ids"].copy()
    return result


grouped_data = tokenized_data.map(group_texts, batched=True)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, return_tensors="tf")

tf_train_set = model.prepare_tf_dataset(
    grouped_data["train"],
    shuffle=True,
    batch_size=16,
    collate_fn=data_collator,
)

tf_test_set = model.prepare_tf_dataset(
    grouped_data["test"],
    shuffle=False,
    batch_size=16,
    collate_fn=data_collator,
)

model = TFGPT2Model.from_pretrained("gpt2")
model.compile()