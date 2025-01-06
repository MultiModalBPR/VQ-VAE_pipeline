import torch
from model import VQVAE2  # Import VQ-VAE model from Rosinality repo
from PIL import Image
from torchvision import transforms
import json
import os
from tqdm import tqdm

# VQ-VAE Tokenization Class
class VQVAE_Tokenizer:
    def __init__(self, checkpoint_path):
        self.vqvae = VQVAE2(channels=3, num_embeddings=512, embedding_dim=64)
        self.vqvae.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        self.vqvae.eval()
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

    def tokenize_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image).unsqueeze(0)
        with torch.no_grad():
            _, _, tokens = self.vqvae.encode(image_tensor)
        return tokens.flatten().cpu().numpy().tolist()

# Tokenize a dataset and save to JSON
def tokenize_dataset(image_dir, checkpoint_path, output_file):
    tokenizer = VQVAE_Tokenizer(checkpoint_path)
    tokenized_data = []

    for image_file in tqdm(os.listdir(image_dir)):
        image_path = os.path.join(image_dir, image_file)
        tokens = tokenizer.tokenize_image(image_path)
        tokenized_data.append({
            "image_id": image_file,
            "tokens": tokens
        })

    with open(output_file, "w") as f:
        json.dump(tokenized_data, f, indent=4)

# Example usage
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", required=True, help="Directory containing images")
    parser.add_argument("--checkpoint_path", required=True, help="Path to pre-trained VQ-VAE checkpoint")
    parser.add_argument("--output_file", required=True, help="Output file to save tokenized data")
    args = parser.parse_args()

    tokenize_dataset(args.image_dir, args.checkpoint_path, args.output_file)
