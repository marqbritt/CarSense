import torch
from torch.utils.data import Dataset, DataLoader
from TorchCRF import CRF
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from transformers import BertTokenizer

# entity labels

entity_labels = [
    'O',  # Outside
    'B-brand', 'I-brand',
    'B-model', 'I-model',
    'B-year', 'I-year',
    'B-mileage', 'I-mileage',
    'B-engine', 'I-engine',
    'B-engine_size', 'I-engine_size',
    'B-transmission', 'I-transmission',
    'B-automatic_transmission', 'I-automatic_transmission',
    'B-drivetrain', 'I-drivetrain',
    'B-min_mpg', 'I-min_mpg',
    'B-max_mpg', 'I-max_mpg',
    'B-damaged', 'I-damaged',
    'B-first_owner', 'I-first_owner',
    'B-personal_use_only', 'I-personal_use_only',
    'B-turbo', 'I-turbo',
    'B-alloy_wheels', 'I-alloy_wheels',
    'B-adaptive_cruise_control', 'I-adaptive_cruise_control',
    'B-navigation_system', 'I-navigation_system',
    'B-power_liftgate', 'I-power_liftgate',
    'B-backup_camera', 'I-backup_camera',
    'B-keyless_start', 'I-keyless_start',
    'B-remote_start', 'I-remote_start',
    'B-sunroof_moonroof', 'I-sunroof_moonroof',
    'B-automatic_emergency_braking', 'I-automatic_emergency_braking',
    'B-stability_control', 'I-stability_control',
    'B-leather_seats', 'I-leather_seats',
    'B-memory_seat', 'I-memory_seat',
    'B-third_row_seating', 'I-third_row_seating',
    'B-apple_carplay_android_auto', 'I-apple_carplay_android_auto',
    'B-bluetooth', 'I-bluetooth',
    'B-usb_port', 'I-usb_port',
    'B-heated_seats', 'I-heated_seats',
    'B-interior_color', 'I-interior_color',
    'B-exterior_color', 'I-exterior_color',
    'B-price', 'I-price'
]


# Define your CustomDataset class
class CustomDataset(Dataset):
    def __init__(self, texts, tokenizer):
        self.texts = texts
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        tokenized_text = self.tokenizer.encode_plus(text, add_special_tokens=True, return_tensors='pt',
                                                    padding='max_length', truncation=True, max_length=128)
        return tokenized_text['input_ids'].squeeze()


# Define your CarNERModel class
class CarNERModel(nn.Module):
    def __init__(self, num_tags, hidden_dim, vocab_size, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.bilstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=2, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_dim * 2, num_tags)
        self.crf = CRF(num_tags)

    def forward(self, input_ids):
        embedded = self.embedding(input_ids)
        output, _ = self.bilstm(embedded)
        output = self.linear(output)
        return output


# Function to train the NER model
def train_ner_model(model, train_loader, val_loader, num_epochs, learning_rate):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        for batch_input_ids in train_loader:
            batch_input_ids = batch_input_ids.to(device)
            optimizer.zero_grad()
            emissions = model(batch_input_ids)
            loss = model.crf(emissions, torch.zeros_like(batch_input_ids))
            loss = -loss
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Average Train Loss: {avg_train_loss:.4f}")

        model.eval()
        with torch.no_grad():
            total_val_loss = 0.0
            for batch_input_ids in val_loader:
                batch_input_ids = batch_input_ids.to(device)
                emissions = model(batch_input_ids)
                loss = model.crf(emissions, torch.zeros_like(batch_input_ids))
                loss = -loss
                total_val_loss += loss.item()
            avg_val_loss = total_val_loss / len(val_loader)
            print(f"Epoch [{epoch + 1}/{num_epochs}], Average Validation Loss: {avg_val_loss:.4f}")

    return model


if __name__ == "__main__":
    # Initialize tokenizer and device
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load datasets
    questions_df = pd.read_csv('path')
    cars_df = pd.read_csv('path').fillna('')

    # Prepare data loaders
    questions_dataset = CustomDataset(questions_df['Pattern'], tokenizer)
    questions_loader = DataLoader(questions_dataset, batch_size=32, shuffle=True)
    cars_dataset = CustomDataset(cars_df.iloc[:, 0], tokenizer)
    cars_loader = DataLoader(cars_dataset, batch_size=32, shuffle=True)

    # Train the model
    model = CarNERModel(num_tags=len(entity_labels), hidden_dim=768, vocab_size=tokenizer.vocab_size,
                        embedding_dim=300).to(device)
    model = train_ner_model(model, questions_loader, cars_loader, num_epochs=10, learning_rate=1e-4)

    # Save the trained model
    print("Saving Model...")
    torch.save(model.state_dict(), 'useNER_model.pth')
    print("Successfully saved model!")
