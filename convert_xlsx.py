import json
import pandas as pd

# # Function to convert JSONL to XLSX
# def jsonl_to_xlsx(jsonl_file_path, xlsx_file_path):
#     data = []
    
#     # Read the .jsonl file
#     with open(jsonl_file_path, 'r', encoding='utf-8') as f:
#         for line in f:
#             data.append(json.loads(line))  # Parse each JSON object and add to the list
    
#     # Convert to a DataFrame
#     df = pd.DataFrame(data)
    
#     # Save as an Excel .xlsx file
#     df.to_excel(xlsx_file_path, index=False)

#     print(f"Conversion complete! File saved as {xlsx_file_path}")

# # File paths
# jsonl_file = 'datasets/dummy_qa_dataset.jsonl'  # Path to your input JSONL file
# xlsx_file = 'datasets/dummy_qa_dataset.xlsx'    # Path to your output XLSX file

# # Run the conversion
# jsonl_to_xlsx(jsonl_file, xlsx_file)


# load a jsonl file and change the "User" field to "user_question" and "Main information to provide" to "fact"
import json

def rename_fields(jsonl_file_path, new_jsonl_file_path):
    data = []
    
    # Read the .jsonl file
    with open(jsonl_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            json_data = json.loads(line)
            new_json_data = {"user_question": json_data["User"], "fact": json_data["Main information to provide"]}
            data.append(new_json_data)  # Parse each JSON object and add to the list
    
    # Save as a new JSONL file
    with open(new_jsonl_file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

    print(f"Conversion complete! File saved as {new_jsonl_file_path}")


# File paths
jsonl_file = 'datasets/dummy_qa_dataset.jsonl'  # Path to your input JSONL file
new_jsonl_file = 'datasets/dummy_qa_dataset_new.jsonl'    # Path to your output JSONL file

# Run the conversion
rename_fields(jsonl_file, new_jsonl_file)