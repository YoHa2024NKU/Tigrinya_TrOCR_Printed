import torch
from torch.utils.data import Dataset
from PIL import Image, ImageEnhance, ImageFilter
import random
import os
import pandas as pd
from dataclasses import dataclass

def augment_image(image):
    """Conservative augmentation for Ethiopic OCR."""
    if random.random() > 0.5:
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(random.uniform(0.9, 1.1))
    if random.random() > 0.5:
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(random.uniform(0.9, 1.1))
    if random.random() > 0.4:
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(random.uniform(0.8, 1.2))
    if random.random() > 0.8:
        image = image.filter(ImageFilter.GaussianBlur(radius=0.3))
    return image

class RobustTigrinyaDataset(Dataset):
    def __init__(self, df, processor, max_target_length=128, split_name="train", augment=False):
        self.df = df.reset_index(drop=True)
        self.processor = processor
        self.max_target_length = max_target_length
        self.split_name = split_name
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row['text'])
        try:
            image = Image.open(row['image_path']).convert("RGB")
            if self.augment and self.split_name == "train":
                image = augment_image(image)
            
            pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.squeeze()
            labels = self.processor.tokenizer(
                text, padding="max_length", max_length=self.max_target_length, truncation=True, return_tensors="pt"
            ).input_ids.squeeze()
            labels[labels == self.processor.tokenizer.pad_token_id] = -100
            
            return {"pixel_values": pixel_values, "labels": labels}
        except Exception as e:
            # Return dummy data on failure
            print(f"Error loading {row.get('image_path', 'unknown')}: {e}")
            return self.__getitem__((idx + 1) % len(self.df))

@dataclass
class OptimizedDataCollator:
    processor: any
    def __call__(self, features):
        pixel_values = torch.stack([f["pixel_values"] for f in features])
        labels = torch.stack([f["labels"] for f in features])
        return {"pixel_values": pixel_values, "labels": labels}

def load_data(data_root):
    splits = {}
    for split in ['train', 'dev', 'test']:
        path = os.path.join(data_root, split, f"{split}.tsv")
        if os.path.exists(path):
            df = pd.read_csv(path, sep='\t', encoding='utf-8')
            df['image_path'] = df['image'].apply(lambda x: os.path.join(data_root, split, "images", str(x)))
            splits[split] = df
        else:
            splits[split] = pd.DataFrame()
    return splits['train'], splits['dev'], splits['test']