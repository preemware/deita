from datasets import load_dataset
import json

def dataset_to_json(dataset, json_file_path):
    """
    Export a dataset to JSON format.

    Parameters:
        dataset: The dataset to export.
        json_file_path: Path to the output JSON file.
    """
    dataset_list = [example for example in dataset]  # Convert dataset to list
    with open(json_file_path, 'w') as outfile:
        json.dump(dataset_list, outfile)  # Dump the entire list to a file

# Load the ShareGPT dataset from the Hugging Face datasets hub
# Replace 'your_dataset_name_here' with the actual dataset name
dataset = load_dataset("hkust-nlp/deita-10k-v0")

# Assuming the dataset has a split, e.g., 'train'. Use the appropriate split.
# If the dataset doesn't have splits, you can omit the `split` parameter.
sharegpt_dataset = dataset["train"]

# Path to the output JSON file
json_file_path = 'deita-10k-v0.json'

# Export the dataset to JSON
dataset_to_json(sharegpt_dataset, json_file_path)

print(f"Dataset exported to JSON format at: {json_file_path}")
