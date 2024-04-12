import os
import torch
from torch.utils.data import DataLoader
import pandas as pd
from transformers import BertTokenizer
from model import CarNERModel, CustomDataset, entity_labels

# Initialize the tokenizer and device
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")
# Assuming entity_labels is defined somewhere accessible, like a separate config file or directly below
index_to_label = {i: label for i, label in enumerate(entity_labels)}

# Load the pre-trained model with correct vocabulary size and embedding dimension
try:
    model_path = '/Users/marq/Documents/useNER/useNER_model.pth'
    model = CarNERModel(num_tags=len(entity_labels), hidden_dim=768, vocab_size=tokenizer.vocab_size, embedding_dim=300)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
except FileNotFoundError:
    print(f"Model file not found at '{model_path}'. Please check the path and try again.")
    exit()
except Exception as e:
    print(f"An error occurred: {e}")
    exit()

# Correct the path to the cars dataset
cars_df_path = ('/Users/marq/Desktop/car_data.csv')
try:
    cars_df = pd.read_csv(cars_df_path)
except FileNotFoundError:
    print(f"Cars dataset file not found at '{cars_df_path}'. Please check the path and try again")
    exit()
except Exception as e:
    print(f"An error occurred while loading the cars dataset: {e}")
    exit()

def extract_entities(question):
    # Tokenize the question
    tokens = tokenizer.tokenize(question)

    # Convert tokens to IDs
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids_tensor = torch.tensor([input_ids]).to(device)

    # Use your NER model to predict entity labels
    with torch.no_grad():
        outputs = model(input_ids_tensor)
    predicted_labels = torch.argmax(outputs, dim=2).squeeze().tolist()

    # Build a dictionary of extracted entities and their types
    entities = {}
    current_entity = ""
    current_type = ""
    for i, label_id in enumerate(predicted_labels):
        label = index_to_label[label_id]
        if label.startswith("B-"):
            if current_entity:
                entities[current_entity] = current_type
            current_type = label[2:]
            current_entity = tokens[i]
        elif label.startswith("I-"):
            current_entity += " " + tokens[i]
    if current_entity:
        entities[current_entity] = current_type

    # Extract entity values
    extracted_values = {
        "brand": [],
        "model": [],
        "price": [],
        "transmission": [],
        "engine": [],
        "year": [],
        "mileage": [],
        "engine_size": [],
        "automatic_transmission": [],
        "fuel_type": [],
        "drivetrain": [],
        "min_mpg": [],
        "max_mpg": [],
        "automatic_emergency_breaking": [],
        "stability_control": [],
        "navigation_system": [],
        "exterior_color": [],
        "interior_color": [],
    }

    for entity, entity_type in entities.items():
        if entity_type == "brand":
            extracted_values["brand"].append(entity)
        elif entity_type == "model":
            extracted_values["model"].append(entity)
        elif entity_type == "price":
            try:
                extracted_values["price"].append(float(entity))
            except ValueError:
                pass
        elif entity_type == "transmission":
            extracted_values["transmission"].append(entity)
        elif entity_type == "engine":
            extracted_values["engine"].append(entity)
        elif entity_type == "year":
            extracted_values["year"].append(entity)
        elif entity_type == "mileage":
            extracted_values["mileage"].append(entity)
        elif entity_type == "engine_size":
            extracted_values["engine_size"].append(entity)
        elif entity_type == "automatic_transmission":
            extracted_values["automatic_transmission"].append(entity)
        elif entity_type == "fuel_type":
            extracted_values["fuel_type"].append(entity)
        elif entity_type == "drivetrain":
            extracted_values["drivetrain"].append(entity)
        elif entity_type == "min_mpg":
            extracted_values["min_mpg"].append(entity)
        elif entity_type == "max_mpg":
            extracted_values["max_mpg"].append(entity)
        elif entity_type == "automatic_emergency_breaking":
            extracted_values["automatic_emergency_breaking"].append(entity)
        elif entity_type == "stability_control":
            extracted_values["stability_control"].append(entity)
        elif entity_type == "navigation_system":
            extracted_values["navigation_system"].append(entity)
        elif entity_type == "exterior_color":
            extracted_values["exterior_color"].append(entity)
        elif entity_type == "interior_color":
            extracted_values["interior_color"].append(entity)

    return extracted_values

# Define the function to answer questions
def answer_question(question):
    # Use your NER model to extract entities from the question
    extracted_values = extract_entities(question)

    # Normalize the cars_df for case-insensitive matching
    cars_df_normalized = cars_df.copy()
    cars_df_normalized = cars_df_normalized.apply(lambda x: x.astype(str).str.lower() if x.dtype == "object" else x)

    # Apply all filters simultaneously
    filtered_cars = cars_df_normalized.copy()
    for entity_type, values in extracted_values.items():
        if not values:
            continue
        normalized_values = [str(value).lower() for value in values]
        column_name = entity_type.replace(" ", "_").lower()
        if column_name in filtered_cars.columns:
            if entity_type in ["brand", "model", "transmission", "engine", "engine_size", "fuel_type", "drivetrain", "interior_color"]:
                filtered_cars = filtered_cars[filtered_cars[column_name].isin(normalized_values)]
            elif entity_type == "exterior_color":
                filtered_cars = filtered_cars[filtered_cars[column_name].str.contains('|'.join(normalized_values), case=False)]
            elif entity_type == "price":
                # Price as a range
                min_value, max_value = min(map(float, normalized_values)), max(map(float, normalized_values))
                filtered_cars = filtered_cars[(filtered_cars[column_name].astype(float) >= min_value) & (filtered_cars[column_name].astype(float) <= max_value)]
            elif entity_type in ["min_mpg", "max_mpg", "year"]:
                # Handle numeric values appropriately
                if entity_type == "year":
                    # Year might require exact matches
                    filtered_cars = filtered_cars[filtered_cars[column_name].isin(normalized_values)]
                else:
                    # MPG values as a range
                    min_value, max_value = min(map(float, normalized_values)), max(map(float, normalized_values))
                    filtered_cars = filtered_cars[(filtered_cars[column_name].astype(float) >= min_value) & (filtered_cars[column_name].astype(float) <= max_value)]

    # Generate the response based on the matching cars
    if not filtered_cars.empty:
        matching_cars_str = [str(car) for _, car in filtered_cars.head(3).iterrows()]
        if len(filtered_cars) > 5:
            matching_cars_str.append("(and more)")
        response = f"Based on your query, the relevant car(s) are:\n{', '.join(matching_cars_str)}"
    else:
        response = "I'm sorry, I couldn't find any matching cars in the dataset."

    return response

# Prompt the user to ask a question
def clear_screen():
    _ = os.system('clear' if os.name == 'posix' else 'cls')
if __name__ == "__main__":
    while True:
        question = input("Ask a question about cars (type 'exit' to quit or retry to clear screen):").strip()
        if question.lower() == 'exit':
            break
        elif question.lower() == 'retry':
            clear_screen()
            continue
        answer = answer_question(question)
        print(answer)
        print()
