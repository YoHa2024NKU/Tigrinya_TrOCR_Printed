import os
import sys
import torch
import pandas as pd
import logging
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch.nn as nn
from dataclasses import dataclass
from transformers import (
    TrOCRProcessor, 
    VisionEncoderDecoderModel, 
    Seq2SeqTrainer, 
    Seq2SeqTrainingArguments
)

# 1. SETUP LOGGING & CONSTANTS
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("TrOCR")
logging.getLogger("transformers.trainer").setLevel(logging.ERROR)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# 2. DATASET CLASS
class SimpleDataset(Dataset):
    def __init__(self, df, processor, root_dir, max_length=128):
        self.df = df.reset_index(drop=True)
        self.processor = processor
        self.root_dir = root_dir
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row['text'])
        image_path = row['image_path']
        
        try:
            image = Image.open(image_path).convert("RGB")
            # No heavy augmentation to keep CPU fast on Windows
            pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.squeeze()
            
            labels = self.processor.tokenizer(
                text, 
                padding="max_length", 
                max_length=self.max_length, 
                truncation=True, 
                return_tensors="pt"
            ).input_ids.squeeze()
            
            labels[labels == self.processor.tokenizer.pad_token_id] = -100
            return {"pixel_values": pixel_values, "labels": labels}
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            return self.__getitem__(0)

@dataclass
class DataCollator:
    processor: any
    def __call__(self, features):
        pixel_values = torch.stack([f["pixel_values"] for f in features])
        labels = torch.stack([f["labels"] for f in features])
        return {"pixel_values": pixel_values, "labels": labels}

# 3. 🔥 OPTIMIZED WORD-AWARE TRAINER 🔥
class WordAwareTrainer(Seq2SeqTrainer):
    """
    Resulted in 97% Accuracy but uses Vectorized C++ implementation
    to avoid the 10s/it slowdown.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # A. Pre-calculate weights ONCE during initialization
        tokenizer = self.processing_class
        vocab = tokenizer.get_vocab()
        vocab_size = len(tokenizer)
        
        print("⚖️  Initializing Word-Aware Weights...")
        # Start with all weights = 1.0
        self.class_weights = torch.ones(vocab_size)
        
        # Identify Space Tokens (starting with Ġ)
        space_indices = [idx for token, idx in vocab.items() if token.startswith('Ġ')]
        
        # Set their weight to 2.0 (The Logic that fixed your problem)
        self.class_weights[space_indices] = 2.0
        
        print(f"✅ Weighted {len(space_indices)} space tokens by 2.0x")

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # Ensure weights are on the correct device (GPU)
        if self.class_weights.device != logits.device:
            self.class_weights = self.class_weights.to(logits.device)

        # B. FAST Loss Calculation
        # Passing 'weight' to CrossEntropyLoss uses optimized C++ kernels
        loss_fct = nn.CrossEntropyLoss(
            weight=self.class_weights, 
            ignore_index=-100
        )
        
        logits_flat = logits.view(-1, logits.size(-1))
        labels_flat = labels.view(-1)
        
        loss = loss_fct(logits_flat, labels_flat)
        
        return (loss, outputs) if return_outputs else loss

# 4. MAIN TRAINING FUNCTION
def main():
    print("="*50)
    print("🚀 STARTING WORD-AWARE TRAINING (FAST MODE)")
    print("="*50)

    # --- CONFIGURATION ---
    DATA_ROOT = "data"
    OUTPUT_DIR = "outputs/fast_model"      #(for Handwritten)
    #OUTPUT_DIR = "outputs/fast_model_printed"
    MODEL_NAME = "microsoft/trocr-base-handwritten"
    #MODEL_NAME = "microsoft/trocr-base-printed"
    
    # BATCH SETTINGS FOR RTX 5060 (8GB VRAM)
    BATCH_SIZE = 2          # Small physical batch to prevent VRAM overflow
    GRAD_ACCUM = 4          # Accumulate to simulate Batch Size 8
    
    EPOCHS = 10
    LR = 4e-5
    
    if not torch.cuda.is_available():
        print("❌ NO GPU DETECTED. STOPPING.")
        return
    
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    set_seed(42)
    device = torch.device("cuda")

    # --- LOAD DATA ---
    print("📂 Loading Dataframes...")
    train_path = os.path.join(DATA_ROOT, "train", "train.tsv")
    dev_path = os.path.join(DATA_ROOT, "dev", "dev.tsv")
    train_df = pd.read_csv(train_path, sep='\t', encoding='utf-8')
    dev_df = pd.read_csv(dev_path, sep='\t', encoding='utf-8')
    train_df['image_path'] = train_df['image'].apply(
        lambda x: os.path.join(DATA_ROOT, "train", "images", str(x))
    )
    dev_df['image_path'] = dev_df['image'].apply(
        lambda x: os.path.join(DATA_ROOT, "dev", "images", str(x))
    )
    print(f"   Train samples: {len(train_df)}")
    print(f"   Dev samples: {len(dev_df)}")

    # --- MODEL & TOKENIZER ---
    print("🤖 Loading Model...")
    # Check for latest checkpoint
    def get_latest_checkpoint(output_dir):
        checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, d))]
        if not checkpoints:
            return None
        # Sort by checkpoint number
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))
        return os.path.join(output_dir, checkpoints[-1])

    latest_checkpoint = get_latest_checkpoint(OUTPUT_DIR)
    if latest_checkpoint:
        print(f"🔄 Resuming from checkpoint: {latest_checkpoint}")
        # Prefer loading processor from best_model if it exists
        best_model_dir = os.path.join(OUTPUT_DIR, 'best_model')
        processor_dir = best_model_dir if os.path.exists(os.path.join(best_model_dir, 'preprocessor_config.json')) else OUTPUT_DIR
        processor = TrOCRProcessor.from_pretrained(processor_dir)
        # Load model from checkpoint
        model = VisionEncoderDecoderModel.from_pretrained(latest_checkpoint)
    else:
        processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
        model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)

    # Only add new tokens if NOT resuming from checkpoint
    if not latest_checkpoint:
        print("🔤 Extending Tokenizer...")
        all_text = "".join(train_df['text'].astype(str).tolist())
        unique_chars = sorted(list(set(all_text)))
        new_tokens = [c for c in unique_chars if ord(c) > 127]
        if new_tokens:
            model.decoder.resize_token_embeddings(len(processor.tokenizer) + len(new_tokens))
            processor.tokenizer.add_tokens(new_tokens)
            print(f"   Added {len(new_tokens)} new tokens.")

    model.config.decoder_start_token_id = processor.tokenizer.bos_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = len(processor.tokenizer)
    # Move generation parameters to generation_config as required by transformers >=5.x
    model.generation_config.max_length = 128
    model.generation_config.num_beams = 1
    model.to(device)
    train_dataset = SimpleDataset(train_df, processor, DATA_ROOT)
    dev_dataset = SimpleDataset(dev_df, processor, DATA_ROOT)
    
    # --- TRAINING ARGS ---
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        num_train_epochs=EPOCHS,
        fp16=True,
        save_strategy="steps",
        save_steps=2000,
        save_total_limit=1,
        logging_steps=50,
        remove_unused_columns=False,
        report_to="none",
        dataloader_num_workers=0,     # Safer for Windows VRAM
        dataloader_pin_memory=True,
        eval_strategy="steps",   # <-- Add this
        eval_steps=2000               # <-- And this (same as save_steps is common)
    )

    # Use our Optimized Custom Trainer
    trainer = WordAwareTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,  # <-- Add this line
        data_collator=DataCollator(processor),
        processing_class=processor.tokenizer
    )

    print("\n🔥 TRAINING STARTED...")
    best_dev_loss = float('inf')
    best_model_dir = os.path.join(OUTPUT_DIR, 'best_model')
    os.makedirs(best_model_dir, exist_ok=True)

    # Determine starting epoch if resuming
    start_epoch = 0
    if latest_checkpoint:
        import json
        trainer_state_path = os.path.join(latest_checkpoint, 'trainer_state.json')
        if os.path.exists(trainer_state_path):
            with open(trainer_state_path, 'r', encoding='utf-8') as f:
                state = json.load(f)
                # Use int(ceil) to ensure we start at the next full epoch
                from math import ceil
                last_epoch = state.get('epoch', 0)
                start_epoch = int(ceil(last_epoch))
                print(f"Resuming from epoch {start_epoch}")

    for epoch in range(start_epoch, EPOCHS):
        # Resume from checkpoint if available (only on first epoch after resume)
        if epoch == start_epoch and latest_checkpoint:
            trainer.train(resume_from_checkpoint=latest_checkpoint)
        else:
            trainer.train()
        # Manual dev set evaluation
        eval_output = trainer.evaluate(eval_dataset=dev_dataset)
        dev_loss = eval_output.get('eval_loss', None)
        print(f"[Epoch {epoch+1}] Dev Loss: {dev_loss}")
        if dev_loss is not None and dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            print(f"[Epoch {epoch+1}] Saving best model (dev loss improved)")
            trainer.save_model(best_model_dir)
            processor.save_pretrained(best_model_dir)
    # Save final model as usual
    print("💾 Saving final model...")
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print("✅ Done!")

if __name__ == "__main__":
    main()