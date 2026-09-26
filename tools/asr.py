import json
import os

def read_json_file(file_path):
    """
    Read and parse a JSON file from the given path
    
    Args:
        file_path (str): Path to the JSON file
        
    Returns:
        dict/list: The parsed JSON data
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {file_path}")
        return None
    except Exception as e:
        print(f"Error reading JSON file: {e}")
        return None
    
path = 'results/Exh/llama2_7b_125000/test_cases/logs_supplemented.json'
    
data = read_json_file(path)

number_success_standard = 0
number_success_contextual = 0

number_total_standard = 0
number_total_contextual = 0


# First, filter out items with 'Copyright' category
for item in data:
    if data[item][0]['FunctionalCategory'] != 'copyright':
        if data[item][0]['FunctionalCategory'] == 'standard':
            number_total_standard += 1
            if data[item][0]['label'] == 1:
                number_success_standard += 1
        elif data[item][0]['FunctionalCategory'] == 'contextual':
            number_total_contextual += 1
            if data[item][0]['label'] == 1:
                number_success_contextual += 1

# Calculate ASR percentages with proper handling of potential division by zero
standard_asr = (number_success_standard / number_total_standard * 100) if number_total_standard > 0 else 0
contextual_asr = (number_success_contextual / number_total_contextual * 100) if number_total_contextual > 0 else 0
total_success = number_success_standard + number_success_contextual
total_attempts = number_total_standard + number_total_contextual
overall_asr = (total_success / total_attempts * 100) if total_attempts > 0 else 0

print(f"Standard ASR: {standard_asr:.2f}% ({number_success_standard}/{number_total_standard})")
print(f"Contextual ASR: {contextual_asr:.2f}% ({number_success_contextual}/{number_total_contextual})")
print(f"Overall ASR: {overall_asr:.2f}% ({total_success}/{total_attempts})")