import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class FineTuningDataset(Dataset):
    def __init__(self, data_path, tokenizer):
        self.data_dir = data_path
        self.df = pd.read_csv(os.path.join(data_path, "index.csv"))
        self.tokenizer = tokenizer

        self.train_tranforms = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_path = os.path.join(self.data_dir, "images", self.df.iloc[idx]["image"])
        image = Image.open(image_path).convert("RGB")
        image = self.train_tranforms(image)

        input_ids = self.tokenizer(
            self.df.iloc[idx]["short_prompt"],
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )["input_ids"][0]

        return {"pixel_values": image, "input_ids": input_ids}