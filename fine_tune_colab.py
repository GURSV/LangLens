import pandas as pd
import os
from PIL import Image
from transformers import (
    BlipProcessor, 
    BlipForConditionalGeneration,
    TrainingArguments,
    Trainer
)
import torch
from torch.utils.data import Dataset as TorchDataset
from sklearn.model_selection import train_test_split

class CustomVQADataset(TorchDataset):
    def __init__(self, df, image_dir, processor, max_length=32):
        self.df = df
        self.image_dir = image_dir
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        question = row['question']
        answer = str(row['answer'])
        image_id = row['image_id']
        
        image_path = os.path.join(self.image_dir, f"{image_id}.png")
        image = Image.open(image_path).convert('RGB')
        
        encoding = self.processor(
            images=image,
            text=question,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        target_encoding = self.processor(
            text=answer,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        encoding['labels'] = target_encoding['input_ids'].squeeze(0)
        
        return encoding

def collate_fn(batch):
    batch_dict = {}
    keys = batch[0].keys()
    for key in keys:
        batch_dict[key] = torch.stack([item[key] for item in batch])
    return batch_dict

def train_custom_vqa_model(
    csv_path: str,
    image_dir: str,
    output_dir: str = "fine_tuned_vqa",
    test_size: float = 0.2,
    num_train_epochs: int = 5,
    batch_size: int = 4,
    learning_rate: float = 2e-5,
    max_length: int = 32
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    df = pd.read_csv(csv_path)
    print(f"Loaded dataset with {len(df)} samples")
    
    train_df, val_df = train_test_split(df, test_size=test_size, random_state=42)
    print(f"Training samples: {len(train_df)}, Validation samples: {len(val_df)}")
    
    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-vqa-base",
        ignore_mismatched_sizes=True
    ).to(device)
    
    train_dataset = CustomVQADataset(train_df, image_dir, processor, max_length=max_length)
    val_dataset = CustomVQADataset(val_df, image_dir, processor, max_length=max_length)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        logging_steps=50,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=torch.cuda.is_available(),
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        report_to=["none"],
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
    )
    
    print("Starting training...")
    trainer.train()
    
    print("Saving model and processor...")
    final_model_dir = os.path.join(output_dir, "final_model")
    model.save_pretrained(final_model_dir)
    processor.save_pretrained(final_model_dir)
    
    return model, processor

csv_path = "/content/drive/MyDrive/data_for_fine_tune.csv"
image_dir = "/content/drive/MyDrive/images"
output_dir = "/content/drive/MyDrive"

try:
    batch_size = 2
    max_length = 32
    
    model, processor = train_custom_vqa_model(
        csv_path=csv_path,
        image_dir=image_dir,
        output_dir=output_dir,
        num_train_epochs=3, 
        batch_size=batch_size,
        learning_rate=2e-5,
        max_length=max_length
    )
    print("Model training and saving completed successfully!")
except Exception as e:
    print(f"An error occurred: {str(e)}")
