import streamlit as st
import os
import json
import random
from Prompts import *
from LLMCalls import LLMCalls
import matplotlib.pyplot as plt
import datetime

class ConfigurableTestModule:
    def __init__(self, test_set_path, prompt_to_test, judge_prompt, model_name):
        self.test_set_path = test_set_path
        self.model_name = model_name
        self.prompt_to_test = prompt_to_test
        self.judge_prompt = judge_prompt

        # read jsonl file
        self.test_set = []
        with open(test_set_path, 'r') as f:
            for line in f:
                self.test_set.append(json.loads(line))

        self.test_set = self.test_set[:20]

    def test_configurable_answers(self, progress_bar=None):
        """
        Tests the model on the test set with dynamically extracted metrics from the judge prompt.
        """
        total_tests = len(self.test_set)
        all_responses = []
        metrics_data = {}

        # Extract metrics from judge prompt
        system_message = "You are an assistant that helps identify metrics to evaluate LLM responses."
        user_message = extract_metrics_from_the_judge_prompt.format(judge_prompt=self.judge_prompt)
        response = LLMCalls.call_openai_chat_completion(self.model_name, user_message, system_message)

        # Parse the extracted metrics (find backticks and extract JSON object, convert to list)
        backticks = response.split("```")
        json_object = backticks[1].strip()
        metrics = json.loads(json_object) # a list of strings
        print(f"\033[93mMetrics: \033[0m{metrics}")

        st.write("**Metrics to be extracted via a judge LLM:**")
        # Display the extracted metrics as a list (joined by commas between [])
        metric_names_string = f"[{', '.join(metrics)}]"
        st.write(metric_names_string)

        # Initialize the metrics data dictionary
        for metric in metrics:
            metrics_data[metric] = []

        for test_idx, test in enumerate(self.test_set):
            system_message = "You are a helpful assistant."
            # format the prompt with the fields from the test set (could be dynamic)(replace {value} with the actual value)
            formatted_prompt = self.prompt_to_test
            for field in test:
                print(f"Field: {field}")
                formatted_prompt = formatted_prompt.replace("{"+field+"}", test[field])
            print(f"\033[94mPrompt: \033[0m{formatted_prompt}")

            user_message = formatted_prompt           
            response = LLMCalls.call_openai_chat_completion(self.model_name, user_message, system_message)
            print(f"\033[92mResponse: \033[0m{response}")

            # evaluate the response via the judge prompt
            print(f"\033[94mJudge Prompt: \033[0m{self.judge_prompt}")
            middle_of_judge_prompt = self.judge_prompt.replace("{LLM_response}", response)
            # also try to replace the fields in the judge prompt
            for field in test:
                middle_of_judge_prompt = middle_of_judge_prompt.replace("{"+field+"}", test[field])

            evaluation_prompt = start_of_eval_prompt + middle_of_judge_prompt + end_of_eval_prompt.format(metric_names=metric_names_string)
            print(f"\033[94mEvaluation Prompt: \033[0m{evaluation_prompt}")
            
            system_message = "You are an LLM judge that evaluates the response based on the metrics extracted."
            evaluation_response = LLMCalls.call_openai_chat_completion("gpt-4o", evaluation_prompt, system_message)

            # Parse the evaluation response to extract the metrics
            backticks = evaluation_response.split("```")
            json_object = backticks[1].strip()
            # now, it's a dictionary with the metrics
            evaluation = json.loads(json_object)
            print(f"\033[93mEvaluation: \033[0m{evaluation}")

            # Append the metrics to the metrics data dictionary
            for metric in metrics:
                try:
                    metrics_data[metric].append(evaluation[metric])
                except Exception as e:
                    print(f"Error: {e}")

            # Update progress bar
            progress = (test_idx + 1) / total_tests
            if progress_bar:
                progress_bar.progress(progress)
            
            # Append the response to the list of all responses
            all_responses.append(response)

        # Average the metrics
        averages = {}
        for metric, values in metrics_data.items():
            # if the values are numbers, calculate the average
            if all(isinstance(value, (int, float)) for value in values):
                averages[metric] = sum(values) / len(values)
            elif all(isinstance(value, str) for value in values):
                # if the values are strings, calculate the frequency of each value
                value_counts = {value: values.count(value) / len(values) for value in set(values)}
                averages[metric] = value_counts

        # Plot the metrics dynamically as histograms or as pie plots for string values and save the image
        num_metrics = len(metrics_data) + 1  # Adding 1 for response length
        num_rows = (num_metrics + 1) // 2
        fig, axs = plt.subplots(num_rows, 2, figsize=(14, 6 * num_rows))
        axs = axs.flatten()

        for idx, (metric, values) in enumerate(metrics_data.items()):
            if all(isinstance(value, (int, float)) for value in values):
                axs[idx].hist(values, bins=10, color='skyblue', edgecolor='black')
                axs[idx].set_title(metric, fontsize=12)
                axs[idx].set_xlabel('Score', fontsize=10)
                axs[idx].set_ylabel('Frequency', fontsize=10)
                axs[idx].grid(True)
            elif all(isinstance(value, str) for value in values):
                value_counts = {value: values.count(value) for value in set(values)}
                axs[idx].pie(value_counts.values(), labels=value_counts.keys(), autopct='%1.1f%%', colors=plt.cm.Paired.colors)
                axs[idx].set_title(metric, fontsize=12)

        # Plot the response length
        response_lengths = [len(response.split()) for response in all_responses]
        axs[-1].hist(response_lengths, bins=10, color='lightgreen', edgecolor='black')
        axs[-1].set_title("Response Length", fontsize=12)
        axs[-1].set_xlabel('Length (words)', fontsize=10)
        axs[-1].set_ylabel('Frequency', fontsize=10)
        axs[-1].grid(True)
        
        # Adjust layout and save
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        image_name = f"plots/metrics_{time}.png"
        plt.savefig(image_name)
        plt.show()

        return image_name, averages


# Initialize session state to persist results
if 'random_responses' not in st.session_state:
    st.session_state['random_responses'] = []
if 'averages' not in st.session_state:
    st.session_state['averages'] = {}
if 'result_image' not in st.session_state:
    st.session_state['result_image'] = ""

st.set_page_config(layout="wide")

# Streamlit app UI
st.title("Configurable Prompt Test :bar_chart:")
st.write("Enter a prompt to test the configurable metrics through corresponding evaluations.")

# Input field for the prompt name
prompt_name = st.text_input("Enter a name for this prompt:", "My new prompt")

# Layout for prompt and judge prompt side by side
col1, col2 = st.columns(2)

# Input field for the prompt text
with col1:
    prompt_input = st.text_area(
        "Enter your prompt:", 
        "Provide a sarcastic response to the following question: \n\"{user_question}\"\nbut the answer must contain the following information:\n\"{fact}\".",
        height=400  # Adjust the height for a larger text area
    )

# Input field for the judge prompt
with col2:
    st.info('The judge prompt must specify the metrics and their value ranges to assess, using the same fields as in the first prompt, plus the {LLM_response} field.', icon="ℹ️")
    judge_prompt = st.text_area(
        "Enter your judge prompt:",
        sample_eval_prompt,
        height=400  # Adjust the height for a larger text area
    )

if st.button("Run Configurable Test"):
    if prompt_input and prompt_name and judge_prompt:
        if "{LLM_response}" not in judge_prompt:
            st.error("The judge prompt must include the '{LLM_response}' field.")
        else:
            # Create a progress bar
            progress_bar = st.progress(0)
            
            # Run the configurable test and get the resulting image, averages, and random responses
            with st.spinner("Running the configurable test..."):
                configurable_test_module = ConfigurableTestModule("datasets/dummy_qa_dataset_new.jsonl", prompt_input, judge_prompt, "gpt-4o")
                result_image, averages = configurable_test_module.test_configurable_answers(progress_bar)
            
            # Store results in session state to persist them across layout changes
            st.session_state['result_image'] = result_image
            st.session_state['averages'] = averages
    
            # Indicate completion
            st.success("Configurable test completed!")

# Display the image 1.5x bigger if available in session state
if st.session_state['result_image']:
    st.image(st.session_state['result_image'], caption="Generated Metrics", use_column_width=False, width=900)

# Display averages if available in session state
if st.session_state['averages']:
    st.subheader("Average Metrics")
    print(st.session_state['averages'])
    metric_display = ""
    for key, value in st.session_state['averages'].items():
        if isinstance(value, dict):
            metric_display += f"**{key.capitalize()}**:<br>"
            for subkey, subvalue in value.items():
                metric_display += f"&nbsp;&nbsp;- **{subkey.capitalize()}**: {subvalue:.2f}<br>"
        elif isinstance(value, (int, float)):
            metric_display += f"**{key.capitalize()}**: {value:.2f}<br>"
    st.markdown(metric_display, unsafe_allow_html=True)
