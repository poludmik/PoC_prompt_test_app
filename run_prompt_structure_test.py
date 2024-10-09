import json
import html  # Import the html module to escape HTML tags
import tqdm
import markdown  # Import the markdown library to convert markdown to HTML
from LLMCalls import LLMCalls
from Prompts import evaluate_prompt_differences


# Load JSONL data from a file
jsonl_file_path = "datasets/output_prompt_structure.jsonl"  # replace with your JSONL file path
output_data = []
response_lengths = {"prompt1": [], "prompt2": [], "prompt3": []}  # For storing lengths of responses per format

# Reading JSONL data
with open(jsonl_file_path, 'r') as f:
    for line in tqdm.tqdm(f):
        json_data = json.loads(line)
        instance_responses = {}
        for key, prompt in json_data.items():
            # Using real LLM call to get the responses
            # response = "dummy response for key: " + key
            response = LLMCalls.call_openai_chat_completion("gpt-4o", prompt)
            instance_responses[key] = {"prompt": prompt, "response": response}
        output_data.append(instance_responses)

# Collect prompts and responses for LLM Judge analysis
judge_analysis_text = evaluate_prompt_differences
for instance in output_data:
    for key, data in instance.items():
        judge_analysis_text += f"Prompt {key}:\n{data['prompt']}\nResponse {key}:\n{data['response']}\n\n"

# Call the OpenAI API to get a summary from an LLM judge
def get_llm_judge_summary(judge_analysis_text):
    try:
        response = LLMCalls.call_openai_chat_completion("gpt-4o", judge_analysis_text, "You are an LLM judge that notices causal patterns in prompts and responses.")
        return response
    except Exception as e:
        return f"Error: {e}"

# Generate the summary
llm_judge_summary = get_llm_judge_summary(judge_analysis_text)

# Convert the judge's summary from Markdown to HTML
llm_judge_summary_html = markdown.markdown(llm_judge_summary)

# Create HTML content with different colored prompts and escaping HTML tags
html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LLM Prompt Comparison</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
        }
        .container {
            display: flex;
            flex-wrap: wrap;
        }
        .column {
            flex: 1;
            padding: 10px;
        }
        .box {
            border: 1px solid #ddd;
            padding: 15px;
            margin-bottom: 20px;
            border-radius: 5px;
        }
        .prompt {
            font-weight: bold;
        }
        .response {
            color: #555;
        }
        .prompt1 {
            background-color: #ffe6e6; /* Light red */
        }
        .prompt2 {
            background-color: #e6ffe6; /* Light green */
        }
        .prompt3 {
            background-color: #e6f0ff; /* Light blue */
        }
        .response-box {
            background-color: #f9f9f9;
        }
        .box div {
            white-space: pre-wrap; /* Allows white spaces and newlines to be rendered */
        }
    </style>
</head>
<body>

<h1>LLM Prompt Structure Comparison</h1>
<div class="container">
"""

# Add each JSON instance to the HTML, replace newlines with <br> tags, and escape HTML tags
color_classes = ["prompt1", "prompt2", "prompt3"]

for instance in output_data:
    html_content += "<div class='container'>"
    for idx, (key, data) in enumerate(instance.items()):
        color_class = color_classes[idx % 3]  # Rotate between prompt1, prompt2, and prompt3
        # Escape HTML tags in prompt and response, and replace newlines with <br>
        prompt_html = html.escape(data['prompt']).replace("\n", "<br>")
        response_html = html.escape(data['response']).replace("\n", "<br>")
        
        # Add the word count to the response_lengths dictionary
        response_length = len(data['response'].split())  # Calculate word count
        response_lengths[color_class].append(response_length)  # Append length to respective prompt format
        
        # Add prompt and response in separate rows
        html_content += f"""
        <div class="column">
            <div class="box {color_class}">
                <div class="prompt">Prompt {key}:</div>
                <div>{prompt_html}</div>
            </div>
            <div class="box response-box">
                <div class="response">Response:</div>
                <div>{response_html}</div>
            </div>
        </div>
        """
    html_content += "</div>"

# Add the LLM Judge's summary (now converted from Markdown to HTML) at the bottom of the HTML
html_content += """
<h2>LLM Judge's Summary:</h2>
""" + llm_judge_summary_html + """
"""

# Calculate the average response length for each prompt format
average_lengths = {format_name: (sum(lengths) / len(lengths)) if lengths else 0 for format_name, lengths in response_lengths.items()}

# Add the average lengths to the HTML
html_content += """
<h2>Average Response Lengths per Prompt Format:</h2>
<ul>
"""
for format_name, avg_length in average_lengths.items():
    html_content += f"<li>{format_name}: {avg_length:.2f} words</li>"

html_content += """
</ul>
"""

# Close HTML
html_content += """
</div>
</body>
</html>
"""

# Write HTML content to file
with open('prompt_comparison_with_judge_summary.html', 'w') as html_file:
    html_file.write(html_content)

print("HTML file 'prompt_comparison_with_judge_summary.html' created successfully.")
