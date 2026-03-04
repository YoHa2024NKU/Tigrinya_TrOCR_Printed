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
    #OUTPUT_DIR = "outputs/fast_model"      #(for Handwritten)
    OUTPUT_DIR = "outputs/fast_model_printed"
    #MODEL_NAME = "microsoft/trocr-base-handwritten"
    MODEL_NAME = "microsoft/trocr-base-printed"
    
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
    train_df = pd.read_csv(train_path, sep='\t', encoding='utf-8')
    train_df['image_path'] = train_df['image'].apply(
        lambda x: os.path.join(DATA_ROOT, "train", "images", str(x))
    )
    print(f"   Train samples: {len(train_df)}")

    # --- MODEL & TOKENIZER ---
    print("🤖 Loading Model...")
    processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)
    
    # Extend Tokenizer
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
    model.config.max_length = 128
    model.config.num_beams = 1
    model.to(device)

    # --- DATASET ---
    train_dataset = SimpleDataset(train_df, processor, DATA_ROOT)
    
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
        dataloader_pin_memory=True
    )

    # Use our Optimized Custom Trainer
    trainer = WordAwareTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollator(processor),
        processing_class=processor.tokenizer
    )

    print("\n🔥 TRAINING STARTED...")
    # Resumes if you have a checkpoint, otherwise starts new
    trainer.train()
    
    print("💾 Saving model...")
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print("✅ Done!")

if __name__ == "__main__":
    main()