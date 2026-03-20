import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image


class DPODataset(Dataset):
    """Dataset for DPO (Direct Preference Optimization) diffusion training.

    Expects a HuggingFace dataset with columns:
        - prompt: str
        - chosen: PIL.Image (preferred image)
        - rejected: PIL.Image (dispreferred image)
    """

    def __init__(self, dataset, tokenizer, tokenizer_2, image_size=1024):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.image_size = image_size

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        prompt = item["prompt"]

        # Tokenizer 1
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

        # Tokenizer 2
        text_inputs_2 = self.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=self.tokenizer_2.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

        # Process images
        chosen_image = item["chosen"].convert("RGB").resize(
            (self.image_size, self.image_size), Image.LANCZOS
        )
        rejected_image = item["rejected"].convert("RGB").resize(
            (self.image_size, self.image_size), Image.LANCZOS
        )

        # Convert to tensors and normalize to [-1, 1]
        chosen_image = torch.tensor(np.array(chosen_image)).float() / 127.5 - 1
        rejected_image = torch.tensor(np.array(rejected_image)).float() / 127.5 - 1

        # Permute to CHW format
        chosen_image = chosen_image.permute(2, 0, 1)
        rejected_image = rejected_image.permute(2, 0, 1)

        return {
            "input_ids": text_inputs.input_ids.squeeze(0),
            "attention_mask": text_inputs.attention_mask.squeeze(0),
            "input_ids_2": text_inputs_2.input_ids.squeeze(0),
            "attention_mask_2": text_inputs_2.attention_mask.squeeze(0),
            "chosen_image": chosen_image,
            "rejected_image": rejected_image,
        }
