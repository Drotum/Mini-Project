import torch
import numpy as np
import os
import pandas as pd
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification, TrainingArguments, Trainer
from datasets import DatasetDict, load_dataset, Audio
import evaluate
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter

# ----- Load metadata with debugging -----
def load_wav2vec_dataset(data_dir="wav2vec_dataset"):
    metadata_path = os.path.join(data_dir, "metadata.tsv")
    clips_path = os.path.join(data_dir, "clips")
    df = pd.read_csv(metadata_path, sep="\t")
    
    # Debug: Check label distribution
    print("Label distribution:")
    print(df['label'].value_counts())
    print(f"Unique labels: {df['label'].unique()}")
    
    df["file"] = df["file"].apply(lambda x: os.path.join(clips_path, x))
    dataset = DatasetDict({
        "train": load_dataset("csv", data_files={"train": metadata_path}, delimiter="\t")['train']
    })
    dataset["train"] = dataset["train"].add_column("path", df["file"])
    dataset = dataset.cast_column("path", Audio(sampling_rate=16000))
    # Rename to match expected keys
    dataset = dataset["train"].rename_columns({"path": "audio", "label": "label"})
    print(f"Dataset loaded: {dataset}")
    return dataset

# ----- Load dataset -----
dataset = load_wav2vec_dataset()

# Debug: Check dataset after loading
print("Dataset features:", dataset.features)
print("First few samples:")
for i in range(min(3, len(dataset))):
    print(f"Sample {i}: label={dataset[i]['label']}, audio_shape={len(dataset[i]['audio']['array'])}")

dataset = dataset.train_test_split(test_size=0.2, seed=42)

# Create proper label mapping
unique_labels = list(set(dataset['train']['label']))
label2id = {label: i for i, label in enumerate(sorted(unique_labels))}
id2label = {i: label for label, i in label2id.items()}

print(f"Label mapping: {label2id}")
print(f"ID to label: {id2label}")

# ----- Load processor and model -----
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
model = Wav2Vec2ForSequenceClassification.from_pretrained(
    "facebook/wav2vec2-base",
    num_labels=len(unique_labels),
    label2id=label2id,
    id2label=id2label,
    problem_type="single_label_classification"
)

# Freeze feature extractor to prevent overfitting
model.freeze_feature_encoder()

# ----- Improved preprocessing -----
def preprocess(batch):
    audio = batch["audio"]
    # Ensure we have audio data
    if audio["array"] is None or len(audio["array"]) == 0:
        print(f"Warning: Empty audio for file: {batch.get('file', 'unknown')}")
        return None
    
    inputs = processor(
        audio["array"],
        sampling_rate=audio["sampling_rate"],
        return_attention_mask=True,
        padding=False,
        truncation=True,
        max_length=160000,  # 10 seconds at 16kHz
        return_tensors="np"
    )
    
    # Convert string labels to integers
    label_id = label2id[batch["label"]]
    
    return {
        "input_values": inputs["input_values"][0],
        "attention_mask": inputs["attention_mask"][0],
        "label": label_id
    }

# Apply preprocessing
print("Preprocessing dataset...")
dataset = dataset.map(preprocess, remove_columns=["file"])

# Debug: Check processed dataset
print("After preprocessing:")
print("Train dataset:", dataset["train"])
print("Test dataset:", dataset["test"])

# Check for class imbalance
train_labels = dataset["train"]["label"]
print("Training label distribution:")
print(Counter(train_labels))

# ----- Calculate class weights for imbalanced data -----
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(train_labels),
    y=train_labels
)
class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}
print(f"Class weights: {class_weights_dict}")

# ----- Custom Data Collator with improved padding -----
@torch.no_grad()
def data_collator(features):
    # Filter out None values from preprocessing
    features = [f for f in features if f is not None]
    
    if len(features) == 0:
        return None
    
    # Convert to tensors
    input_values = [torch.FloatTensor(f["input_values"]) for f in features]
    attention_masks = [torch.LongTensor(f["attention_mask"]) for f in features]
    labels = torch.LongTensor([f["label"] for f in features])
    
    # Find max length in the batch
    max_len = max(x.size(0) for x in input_values)
    
    # Pad sequences
    batch_input_values = torch.zeros(len(input_values), max_len, dtype=torch.float)
    batch_attention_mask = torch.zeros(len(attention_masks), max_len, dtype=torch.long)
    
    for i, (input_val, attention_mask) in enumerate(zip(input_values, attention_masks)):
        length = input_val.size(0)
        batch_input_values[i, :length] = input_val
        batch_attention_mask[i, :length] = attention_mask
    
    return {
        "input_values": batch_input_values,
        "attention_mask": batch_attention_mask,
        "labels": labels
    }

# ----- Fixed Weighted Loss Function -----
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute weighted loss for imbalanced classes
        """
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get('logits')
        
        if labels is not None:
            # Apply class weights
            weight_tensor = torch.tensor([class_weights_dict[i] for i in range(len(class_weights_dict))], 
                                       dtype=torch.float).to(logits.device)
            loss_fct = torch.nn.CrossEntropyLoss(weight=weight_tensor)
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        else:
            # If no labels, use default loss
            loss = outputs.loss if hasattr(outputs, 'loss') else None
            if loss is None:
                loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss

# ----- Enhanced metrics -----
def compute_metrics(pred):
    preds = np.argmax(pred.predictions, axis=1)
    accuracy = evaluate.load("accuracy").compute(predictions=preds, references=pred.label_ids)
    
    # Additional metrics
    from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
    precision, recall, f1, _ = precision_recall_fscore_support(pred.label_ids, preds, average='weighted')
    
    print("Confusion Matrix:")
    cm = confusion_matrix(pred.label_ids, preds)
    print(cm)
    
    return {
        'accuracy': accuracy['accuracy'],
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# ----- Improved training arguments -----
training_args = TrainingArguments(
    output_dir="./wav2vec2-checkpoints",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,  # Lower learning rate
    per_device_train_batch_size=4,  # Smaller batch size
    per_device_eval_batch_size=4,
    num_train_epochs=15,  # More epochs
    warmup_steps=100,  # Fewer warmup steps
    logging_dir="./logs",
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="f1",  # Use F1 instead of accuracy
    greater_is_better=True,
    save_total_limit=3,
    dataloader_drop_last=False,
    gradient_accumulation_steps=2,  # Effective batch size = 4*2 = 8
    weight_decay=0.01,  # Add regularization
)

# ----- Trainer with weighted loss -----
trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    processing_class=processor,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# ----- Train -----
print("Starting training...")
trainer.train()

# ----- Save the final model -----
trainer.save_model("./final_cow_chewing_model")
print("Model saved to ./final_cow_chewing_model")