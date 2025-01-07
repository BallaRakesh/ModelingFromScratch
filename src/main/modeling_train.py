import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from transformers import ViTModel, Wav2Vec2Model, GPT2LMHeadModel, GPT2Tokenizer
from torchvision import transforms
from PIL import Image
import torchaudio
import os
import numpy as np

class MultiModalDataset(Dataset):
    def __init__(self, image_dir, audio_dir, captions_file, max_length=512):
        self.image_dir = image_dir
        self.audio_dir = audio_dir
        
        # Load captions
        self.samples = []
        with open(captions_file, 'r') as f:
            for line in f:
                img_name, audio_name, caption = line.strip().split('|')
                self.samples.append({
                    'image': os.path.join(image_dir, img_name),
                    'audio': os.path.join(audio_dir, audio_name),
                    'caption': caption
                })
        
        # Initialize transforms and tokenizer
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
        
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load and process image
        image = Image.open(sample['image']).convert('RGB')
        image = self.image_transform(image)
        
        # Load and process audio
        waveform, sample_rate = torchaudio.load(sample['audio'])
        # Resample to 16kHz if needed
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
        
        # Tokenize caption
        caption_tokens = self.tokenizer.encode(
            sample['caption'],
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'image': image,
            'audio': waveform,
            'caption': caption_tokens.squeeze(0),
            'caption_text': sample['caption']
        }

class MultiModalTransformer(pl.LightningModule):
    def __init__(self):
        super().__init__()
        
        # Load pretrained models
        self.vision_encoder = ViTModel.from_pretrained('google/vit-base-patch16-224')
        self.audio_encoder = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base')
        self.text_decoder = GPT2LMHeadModel.from_pretrained('gpt2')
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        
        # Projection layers
        self.vision_projection = nn.Linear(768, 768)  # ViT hidden size to GPT2 hidden size
        self.audio_projection = nn.Linear(768, 768)   # Wav2Vec2 hidden size to GPT2 hidden size
        
        # Cross-attention layers
        self.cross_attention = nn.MultiheadAttention(embed_dim=768, num_heads=12)
        
    def forward(self, image, audio, caption_ids=None):
        # Encode image
        vision_outputs = self.vision_encoder(image).last_hidden_state
        vision_features = self.vision_projection(vision_outputs)
        
        # Encode audio
        audio_outputs = self.audio_encoder(audio).last_hidden_state
        audio_features = self.audio_projection(audio_outputs)
        
        # Combine multimodal features using cross-attention
        combined_features, _ = self.cross_attention(
            vision_features, 
            audio_features, 
            audio_features
        )
        
        # Generate text using GPT2
        if caption_ids is not None:
            # Training mode
            outputs = self.text_decoder(
                input_ids=caption_ids,
                encoder_hidden_states=combined_features,
                labels=caption_ids,
                return_dict=True
            )
            return outputs.loss
        else:
            # Inference mode
            generated_ids = self.text_decoder.generate(
                encoder_hidden_states=combined_features,
                max_length=100,
                num_beams=5,
                no_repeat_ngram_size=2,
                early_stopping=True
            )
            return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    def training_step(self, batch, batch_idx):
        loss = self(
            batch['image'],
            batch['audio'],
            batch['caption']
        )
        self.log('train_loss', loss)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=5e-5)

# Training setup
def train_model():
    # Initialize dataset and dataloader
    dataset = MultiModalDataset(
        image_dir='path/to/images',
        audio_dir='path/to/audio',
        captions_file='path/to/captions.txt'
    )
    
    train_loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=4
    )
    
    # Initialize model and trainer
    model = MultiModalTransformer()
    trainer = pl.Trainer(
        max_epochs=10,
        accelerator='gpu',
        devices=1,
        precision=16,  # Use mixed precision training
        gradient_clip_val=1.0
    )
    
    # Train the model
    trainer.fit(model, train_loader)
    
# Example usage for inference
def generate_caption(model, image_path, audio_path):
    # Load and preprocess image
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    image = image_transform(image).unsqueeze(0)
    
    # Load and preprocess audio
    waveform, sample_rate = torchaudio.load(audio_path)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)
    waveform = waveform.unsqueeze(0)
    
    # Generate caption
    with torch.no_grad():
        caption = model(image, waveform)
    return caption
