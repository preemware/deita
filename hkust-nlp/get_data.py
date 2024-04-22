from datasets import load_dataset
import json

# Specify the dataset name and version
dataset_name = "hkust-nlp/deita-10k-v0"

# Load the dataset
dataset = load_dataset(dataset_name)

# Convert the train split to a list of dictionaries
train_data = [example for example in dataset["train"]]

# Save the train data as a JSON file
with open("train_data.json", "w") as file:
    json.dump(train_data, file, indent=4)

print("Dataset downloaded and saved as train_data.json")
