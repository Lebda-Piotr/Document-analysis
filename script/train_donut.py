import torch
from transformers import (
    VisionEncoderDecoderModel,
    DonutProcessor,
    MBart50Tokenizer,
    TrainingArguments,
    Trainer
)
from torch.utils.data import Dataset
import json
from PIL import Image

class DonutDataset(Dataset):
    def __init__(self, annotations_path, processor):
        with open(annotations_path) as f:
            self.data = json.load(f)
        self.processor = processor
        self.tokenizer = MBart50Tokenizer.from_pretrained("facebook/mbart-large-50")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(item["image_path"]).convert("RGB")
        
        # Przetwarzanie obrazu
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze()
        
        # Przygotowanie tekstu
        prompt = f"<s_{item['language']}><s_dokument>"
        text = json.dumps(item["entities"])
        labels = self.tokenizer(
            prompt + text + "</s>",
            add_special_tokens=False,
            max_length=512,
            padding="max_length",
            return_tensors="pt"
        ).input_ids.squeeze()
        
        return {
            "pixel_values": pixel_values,
            "labels": labels
        }

# Inicjalizacja modelu
processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base")

# Dostosowanie tokenizera
tokenizer = MBart50Tokenizer.from_pretrained("facebook/mbart-large-50")
tokenizer.add_special_tokens([f"<s_{lang}>" for lang in ["de", "da"]] + ["</s>"])

# Konfiguracja treningu
training_args = TrainingArguments(
    output_dir="./models/trained",
    num_train_epochs=15,
    per_device_train_batch_size=2,
    learning_rate=2.5e-5,
    fp16=True,
    logging_dir="./logs",
    save_strategy="epoch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=DonutDataset("data/annotations.json", processor),
)

# Rozpocznij trening
trainer.train()

# Zapisz model
model.save_pretrained("./models/trained")
processor.save_pretrained("./models/trained")