from transformers import pipeline, T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
from datasets import Dataset

# Use the text-to-text generation pipeline with a model
pipe = pipeline("text2text-generation", model="facebook/blenderbot-400M-distill")

def convert_to_chess_move(pipe, input_text):
    # Define the prompt format and instructions
    prompt = f"Convert the following natural language sentence into a chess move notation: '{input_text}'"
    # Use the pipeline to generate the output
    response = pipe(prompt)
    # Get the first result from the response
    chess_move = response[0]['generated_text']
    # Return the generated move (adjust this as per the response formatting)
    return chess_move.strip()

# Example usage
input_text = "move the knight from c4 to f6"
output_move = convert_to_chess_move(pipe, input_text)
print(f"Chess Move: {output_move}")

# ---------------------------------------- #

# Load dataset
dataset = Dataset.from_dict({
    'input': [
        'Move the knight from c4 to f6',
        'Move the queen from e4 to d5',
        'Move night from e4 to d5',
        'Move rook e4 to g3',
        'queen e8 to d5',
        'night left e4 to c5',
        'night right h2 to d2',
        'king h2 to d5',
        "go the knight from c4 to f6",
        "move the queen e4 to d5",
        "Move rook from h1 to h5",
        "bishop from f1 to c4",
        "Move the knight from b1 to c3",
        "Knight on g8 moves to f6",
        "Knight from e4 to g5",
        "Move knight from d2 to e4",
        "The knight at h6 goes to f7",
        "Move the queen from d1 to h5",
        "Queen on b3 to e6",
        "The queen at f4 goes to a7",
        "Move queen from a1 to b3",
        "The queen moves from c5 to e5",
        "Rook from a8 to a3",
        "Move the rook from h1 to h4",
        "Rook moves from c3 to c5",
        "Move rook from f2 to f5",
        "Rook on d4 to f6",
        "King from e1 to d2",
        "Move the king from h1 to g2",
        "King moves from b2 to a3",
        "Move the king from f5 to e4",
        "King from d7 to e8",
        "Move bishop from f1 to c4",
        "Bishop on e3 goes to d2",
        "Bishop moves from c1 to f4",
        "Move bishop from a2 to b3",
        "Bishop moves from g4 to e2",

        'move the knight from c4 to f6',
        'move the queen from e4 to d5',
        'move night from e4 to d5',
        'move rook e4 to g3',
        'queen e8 to d5',
        'night left e4 to c5',
        'night right h2 to d2',
        'king h2 to d5',
        "go the knight from c4 to f6",
        "move the queen e4 to d5",
        "move rook from h1 to h5",
        "bishop from f1 to c4",
        "move the knight from b1 to c3",
        "knight on g8 moves to f6",
        "knight from e4 to g5",
        "Move knight from d2 to e4",
        "The knight at h6 goes to f7",
        "move the queen from d1 to h5",
        "queen on b3 to e6",
        "the queen at f4 goes to a7",
        "move queen from a1 to b3",
        "the queen moves from c5 to e5",
        "rook from a8 to a3",
        "move the rook from h1 to h4",
        "rook moves from c3 to c5",
        "move rook from f2 to f5",
        "rook on d4 to f6",
        "ming from e1 to d2",
        "move the king from h1 to g2",
        "king moves from b2 to a3",
        "move the king from f5 to e4",
        "king from d7 to e8",
        "move bishop from f1 to c4",
        "bishop on e3 goes to d2",
        "bishop moves from c1 to f4",
        "move bishop from a2 to b3",
        "bishop moves from g4 to e2",
    ],
    'output': [
        'N(c4)f6',
        'Q(e4)d5',
        'N(c4)f6',
        'R(e4)g3',
        'Q(e8)d5',
        'N(e4)c5',
        'N(h2)d2',
        'K(h2)d5',
        "N(c4)f6",
        "Q(e4)d5",
        "R(h1)h5",
        "B(f1)c4",
        "N(b1)c3",
        "N(g8)f6",
        "N(e4)g5",
        "N(d2)e4",
        "N(h6)f7",
        "Q(d1)h5",
        "Q(b3)e6",
        "Q(f4)a7",
        "Q(a1)b3",
        "Q(c5)e5",
        "R(a8)a3",
        "R(h1)h4",
        "R(c3)c5",
        "R(f2)f5",
        "R(d4)f6",
        "K(e1)d2",
        "K(h1)g2",
        "K(b2)a3",
        "K(f5)e4",
        "K(d7)e8",
        "B(f1)c4",
        "B(e3)d2",
        "B(c1)f4",
        "B(a2)b3",
        "B(g4)e2",
        'N(c4)f6',
        'Q(e4)d5',
        'N(c4)f6',
        'R(e4)g3',
        'Q(e8)d5',
        'N(e4)c5',
        'N(h2)d2',
        'K(h2)d5',
        "N(c4)f6",
        "Q(e4)d5",
        "R(h1)h5",
        "B(f1)c4",
        "N(b1)c3",
        "N(g8)f6",
        "N(e4)g5",
        "N(d2)e4",
        "N(h6)f7",
        "Q(d1)h5",
        "Q(b3)e6",
        "Q(f4)a7",
        "Q(a1)b3",
        "Q(c5)e5",
        "R(a8)a3",
        "R(h1)h4",
        "R(c3)c5",
        "R(f2)f5",
        "R(d4)f6",
        "K(e1)d2",
        "K(h1)g2",
        "K(b2)a3",
        "K(f5)e4",
        "K(d7)e8",
        "B(f1)c4",
        "B(e3)d2",
        "B(c1)f4",
        "B(a2)b3",
        "B(g4)e2",
    ]
})

# Load pre-trained T5 model
model = T5ForConditionalGeneration.from_pretrained("t5-small")
tokenizer = T5Tokenizer.from_pretrained("t5-small")

# Tokenizing both input and output texts
input_encodings = tokenizer(dataset['input'], padding=True, truncation=True, return_tensors="pt", max_length=128)
output_encodings = tokenizer(dataset['output'], padding=True, truncation=True, return_tensors="pt", max_length=128)

# Add the decoder_input_ids (shifted version of target text) to the input encodings
# decoder_input_ids: This is typically the target text, but shifted by one token
output_ids = output_encodings['input_ids']
decoder_input_ids = output_ids[:, :-1]  # Shift the target sequence (remove the last token)
labels = output_ids[:, 1:]              # The labels are the target sequence, shifted by 1 token

# Now, we need to add decoder_input_ids and labels to the input encoding
input_encodings['decoder_input_ids'] = decoder_input_ids
input_encodings['labels'] = labels

# dataset = Dataset.from_dict(input_encodings)
dataset = Dataset.from_dict({**input_encodings, 'input': dataset['input'], 'output': dataset['output']})


# Optionally, split the dataset into train and eval sets
train_dataset = dataset.select(range(0, 9))  # For example, use first 9 samples for training
eval_dataset = dataset.select(range(9, 12))  # Use the remaining samples for evaluation

# ------------------------------------------ #

# Set up the training arguments
training_args = TrainingArguments(
    remove_unused_columns=False,
    output_dir="./results",          # output directory
    num_train_epochs=200,            # number of training epochs
    per_device_train_batch_size=8,   # batch size for training
    per_device_eval_batch_size=8,    # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir="./logs",            # directory for storing logs
    logging_steps=10,
    save_steps=500,
    save_total_limit=3,
    evaluation_strategy="no",     # evaluation strategy to adopt during training
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# Train the model
trainer.train()
tokenizer.save_pretrained("./results/checkpoint-600")

