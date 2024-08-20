import re
import pandas as pd

def parse_mmlu_results(file_path):
    # Read the content of the file
    with open(file_path, 'r') as file:
        raw_output = file.read()

    # Define a regular expression pattern to match the task name and accuracy
    task_pattern = r'MMLU Task Accuracy \(task=(.*?)\): (\d+\.\d+)'
    
    # Find all matches for the pattern in the raw output
    matches = re.findall(task_pattern, raw_output)
    
    # Create a dictionary to store the task name and accuracy
    results = {}
    for match in matches:
        task_name, accuracy = match
        results[task_name] = float(accuracy)
    
    # Convert the list of dictionaries to a pandas DataFrame
    results = pd.DataFrame.from_dict(results, orient='index', columns=["accuracy"])
    
    # Return the parsed results
    return results

# Example usage
file_path = 'mmlu_results/wo_defender.txt'  # Replace with the path to your txt file
results = parse_mmlu_results(file_path)
results = results.sort_index()

supercategories = [
    "STEM",
    "STEM",
    "STEM",
    "Other",
    "Other",
    "STEM",
    "STEM",
    "STEM",
    "STEM",
    "Other",
    "STEM",
    "STEM",
    "STEM",
    "Social Sciences",
    "STEM",
    "STEM",
    "Humanities",
    "Other",
    "STEM",
    "STEM",
    "STEM",
    "Humanities",
    "Social Sciences",
    "Social Sciences",
    "Social Sciences",
    "STEM",
    "Social Sciences",
    "STEM",
    "Social Sciences",
    "STEM",
    "Humanities",
    "Humanities",
    "Other",
    "Social Sciences",
    "Humanities",
    "Humanities",
    "Humanities",
    "STEM",
    "Other",
    "Other",
    "Other",
    "Other",
    "Humanities",
    "Humanities",
    "Other",
    "Humanities",
    "Humanities",
    "Other",
    "Humanities",
    "Other",
    "Social Sciences",
    "Social Sciences",
    "Social Sciences",
    "Social Sciences",
    "Social Sciences",
    "Other",
    "Humanities",
]

results["supercategory"] = supercategories
results.to_csv("mmlu_results/wo_defender.csv")
# Calculate the average accuracy for each supercategory
average_accuracy_per_supercategory = results.groupby("supercategory")["accuracy"].mean()

