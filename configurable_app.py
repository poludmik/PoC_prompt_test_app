import streamlit as st
from streamlit_extras.stylable_container import stylable_container
import os
import json
import random
from Prompts import end_of_eval_prompt, extract_metrics_from_the_judge_prompt, sample_eval_prompt, start_of_eval_prompt
from LLMCalls import LLMCalls
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime
from collections import Counter
import nltk
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')

class ConfigurableTestModule:
    def __init__(self, test_set_path, prompt_to_test, judge_prompt, tested_model_name, judge_model_name):
        self.test_set_path = test_set_path
        if not os.path.exists(test_set_path):
            raise FileNotFoundError(f"Test set file not found at: {test_set_path}")

        self.tested_model_name = tested_model_name
        self.judge_model_name = judge_model_name
        self.prompt_to_test = prompt_to_test
        self.judge_prompt = judge_prompt

        self.test_set = []
        if test_set_path.endswith('.jsonl'):
            with open(test_set_path, 'r') as f:
                for line in f:
                    self.test_set.append(json.loads(line))
        elif test_set_path.endswith('.xlsx'):
            df = pd.read_excel(test_set_path)
            self.test_set = df.to_dict(orient='records')
            print(f"Test set: {self.test_set}")

        for feature in self.test_set[0].keys():
            if "{" + feature + "}" not in prompt_to_test:
                raise ValueError("Field {" + feature + "} isn't included in the prompt to test.")
            if "{" + feature + "}" not in judge_prompt:
                raise ValueError("Field {" + feature + "} isn't included in the judge prompt.")

        self.test_set = self.test_set[:5]

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
        response = LLMCalls.call_openai_chat_completion(self.judge_model_name, user_message, system_message)

        # Parse the extracted metrics (find backticks and extract JSON object, convert to list)
        backticks = response.split("```")
        json_object = backticks[1].strip()
        metrics = json.loads(json_object)  # a list of strings
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
            formatted_prompt = self.prompt_to_test
            for field in test:
                print(f"Field: {field}")
                formatted_prompt = formatted_prompt.replace("{"+field+"}", test[field])
            print(f"\033[94mPrompt: \033[0m{formatted_prompt}")

            user_message = formatted_prompt
            response = LLMCalls.call_openai_chat_completion(self.tested_model_name, user_message, system_message)
            print(f"\033[92mResponse: \033[0m{response}")

            # evaluate the response via the judge prompt
            print(f"\033[94mJudge Prompt: \033[0m{self.judge_prompt}")
            mid = self.judge_prompt.replace("{LLM_response}", response)
            # also try to replace the fields in the judge prompt
            for field in test:
                mid = mid.replace("{"+field+"}", test[field])

            evaluation_prompt = start_of_eval_prompt + mid + end_of_eval_prompt.format(metric_names=metric_names_string)
            print(f"\033[94mEvaluation Prompt: \033[0m{evaluation_prompt}")

            system_message = "You are an LLM judge that evaluates the response based on the metrics extracted."
            evaluation_response = LLMCalls.call_openai_chat_completion(self.judge_model_name,
                                                                       evaluation_prompt,
                                                                       system_message)

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
                averages[metric] = sum(values) / len(values) if len(values) > 0 else 0
            elif all(isinstance(value, str) for value in values):
                # if the values are strings, calculate the frequency of each value
                value_counts = {value: values.count(value) / len(values) for value in set(values)}
                averages[metric] = value_counts

        # Plot the metrics dynamically as histograms or as pie plots for string values and save the image
        num_metrics = len(metrics_data) + 7  # Adding 1 for response length
        num_rows = (num_metrics + 1) // 2
        fig, axs = plt.subplots(num_rows, 2, figsize=(14, 6 * num_rows))
        axs = axs.flatten()

        for idx, (metric, values) in enumerate(metrics_data.items()):
            # test boolean values first
            if all(isinstance(value, (bool, np.bool)) for value in values):
                print(f"Plotting pie chart for {metric}")
                print(f"Values: {values}")
                print(f"Type: {type(values[0])}" if values else "No values")
                print("1")
                value_counts = {value: values.count(value) for value in set(values)}
                axs[idx].pie(value_counts.values(), labels=value_counts.keys(), autopct='%1.1f%%',
                             colors=plt.cm.Paired.colors)
                axs[idx].set_title(metric, fontsize=12)
            elif all(isinstance(value, (int, float)) for value in values):
                print(f"Plotting histogram for {metric}")
                print(f"Values: {values}")
                print(f"Type: {type(values[0])}" if values else "No values")
                print("2")
                axs[idx].hist(values, bins=10, color='skyblue', edgecolor='black')
                axs[idx].set_title(metric, fontsize=12)
                axs[idx].set_xlabel('Score', fontsize=10)
                axs[idx].set_ylabel('Frequency', fontsize=10)
                axs[idx].grid(True)
            elif all(isinstance(value, str) for value in values):
                print(f"Plotting histogram for {metric}")
                print(f"Values: {values}")
                print(f"Type: {type(values[0])}" if values else "No values")
                print("3")
                value_counts = {value: values.count(value) for value in set(values)}
                axs[idx].pie(value_counts.values(), labels=value_counts.keys(), autopct='%1.1f%%',
                             colors=plt.cm.Paired.colors)
                axs[idx].set_title(metric, fontsize=12)

        # Plot the response length
        response_lengths = [len(response.split()) for response in all_responses]
        axs[-1].hist(response_lengths, bins=10, color='lightgreen', edgecolor='black')
        axs[-1].set_title("Response Length", fontsize=12)
        axs[-1].set_xlabel('Length (words)', fontsize=10)
        axs[-1].set_ylabel('Frequency', fontsize=10)
        axs[-1].grid(True)

        # Plot the number of short words
        short_words = [len([word for word in response.split() if len(word) <= 5]) for response in all_responses]
        axs[-2].hist(short_words, bins=10, color='lightcoral', edgecolor='black')
        axs[-2].set_title("Number of Short Words", fontsize=12)
        axs[-2].set_xlabel('Count', fontsize=10)
        axs[-2].set_ylabel('Frequency', fontsize=10)
        axs[-2].grid(True)

        # Plot the number of long words
        long_words = [len([word for word in response.split() if len(word) > 10]) for response in all_responses]
        axs[-3].hist(long_words, bins=10, color='lightcoral', edgecolor='black')
        axs[-3].set_title("Number of Long Words", fontsize=12)
        axs[-3].set_xlabel('Count', fontsize=10)
        axs[-3].set_ylabel('Frequency', fontsize=10)
        axs[-3].grid(True)

        # Plot the number of short sentences
        short_sentences = [len([word for word in response.split() if '.' in word]) for response in all_responses]
        axs[-4].hist(short_sentences, bins=10, color='lightcoral', edgecolor='black')
        axs[-4].set_title("Number of Short Sentences", fontsize=12)
        axs[-4].set_xlabel('Count', fontsize=10)
        axs[-4].set_ylabel('Frequency', fontsize=10)
        axs[-4].grid(True)

        # Plot the number of long sentences
        long_sentences = [len([word for word in response.split() if '.' in word and len(word) > 10]) for response in all_responses]
        axs[-5].hist(long_sentences, bins=10, color='lightcoral', edgecolor='black')
        axs[-5].set_title("Number of Long Sentences", fontsize=12)
        axs[-5].set_xlabel('Count', fontsize=10)
        axs[-5].set_ylabel('Frequency', fontsize=10)
        axs[-5].grid(True)

        # Plot the number of punctuation marks
        punctuation_marks = [len([word for word in response if word in [',', '.', '!', '?']]) for response in all_responses]
        axs[-6].hist(punctuation_marks, bins=10, color='lightcoral', edgecolor='black')
        axs[-6].set_title("Number of Punctuation Marks", fontsize=12)
        axs[-6].set_xlabel('Count', fontsize=10)
        axs[-6].set_ylabel('Frequency', fontsize=10)
        axs[-6].grid(True)

        # Collect all PoS tags and average. How many nouns, verbs, adjectives, adverbs, etc. is on average in the responses?
        all_pos_tags = []
        for response in all_responses:
            tokens = nltk.word_tokenize(response)
            pos_tags = nltk.pos_tag(tokens) # list of tuples (word, PoS tag)
            counts = Counter(tag for word, tag in pos_tags)
            all_pos_tags.append(counts)
        # Average the PoS tags
        pos_tags_averages = {}
        for tag in counts.keys():
            pos_tags_averages[tag] = sum(tag_counts[tag] for tag_counts in all_pos_tags) / len(all_pos_tags)
        
        # Plot the PoS tags (dict)
        axs[-7].bar(pos_tags_averages.keys(), pos_tags_averages.values(), color='lightcoral', edgecolor='black')
        axs[-7].set_title("Average PoS Tags", fontsize=12)
        axs[-7].set_xlabel('PoS Tag', fontsize=10)
        axs[-7].set_ylabel('Average Count', fontsize=10)
        axs[-7].grid(True)

        averages["Response Length"] = sum(response_lengths) / len(response_lengths)
        averages["Number of Short Words"] = sum(short_words) / len(short_words)
        averages["Number of Long Words"] = sum(long_words) / len(long_words)
        averages["Number of Short Sentences"] = sum(short_sentences) / len(short_sentences)
        averages["Number of Long Sentences"] = sum(long_sentences) / len(long_sentences)
        averages["Number of Punctuation Marks"] = sum(punctuation_marks) / len(punctuation_marks)
        averages["Average PoS Tags"] = pos_tags_averages

        # Adjust layout and save
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        image_name = f"plots/metrics_{time}.png"
        plt.savefig(image_name)
        plt.show()

        return image_name, averages, all_responses


# Initialize session state to persist results
if 'random_responses' not in st.session_state:
    st.session_state['random_responses'] = []
if 'averages' not in st.session_state:
    st.session_state['averages'] = {}
if 'result_image' not in st.session_state:
    st.session_state['result_image'] = ""

st.set_page_config(layout="wide")

# Streamlit app UI
st.title("Test the prompt on a set of examples :bar_chart:")
st.write("Select a dataset, models, and prompts to test the 'left' prompt on a set of examples. Judge model will evaluate the responses based on the criteria provided. The program will call the 'left' LLM with it's test prompt and the judge LLM with it's answer on every dataset instance.")

with stylable_container(
        key="container_with_border",
        css_styles="""
            {
                border: 1px solid rgba(49, 51, 63, 0.2);
                border-radius: 0.5rem;
                padding: calc(1em - 1px)
            }
            """,
                        ):
    # Dataset selection and upload
    st.subheader("Dataset Selection")

    # Select existing dataset
    uploaded_datasets = [f for f in os.listdir("datasets") if f.endswith('.xlsx') or f.endswith('.jsonl')]
    selected_dataset = st.selectbox("Select an existing dataset:", uploaded_datasets, index=0)

    # Display available fields from the selected dataset
    if selected_dataset.endswith('.jsonl'):
        with open(f"datasets/{selected_dataset}", 'r') as f:
            first_line = json.loads(f.readline())
            available_fields = list(first_line.keys())
    elif selected_dataset.endswith('.xlsx'):
        df = pd.read_excel(f"datasets/{selected_dataset}")
        available_fields = list(df.columns)

    st.write(":exclamation:**Fields that need to be present in both prompts:**  " +
             ", ".join(["{" + field + "}" for field in available_fields]))

    # preview the first two rows of the dataset
    if selected_dataset.endswith('.jsonl'):
        df = pd.read_json(f"datasets/{selected_dataset}", lines=True)
    elif selected_dataset.endswith('.xlsx'):
        df = pd.read_excel(f"datasets/{selected_dataset}")

    st.dataframe(df.head(min(2, len(df))), hide_index=True)

    # Upload new dataset
    uploaded_file = st.file_uploader("**Upload a new dataset (XLSX or JSONL)**:", type=["xlsx", "jsonl"])

    if uploaded_file:
        new_dataset_path = os.path.join("datasets", uploaded_file.name)
        with open(new_dataset_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"Uploaded {uploaded_file.name} successfully!")
        uploaded_datasets.append(uploaded_file.name)
        selected_dataset = uploaded_file.name

# st.divider()
st.subheader("Model Selection and Prompts")

model_options_test = ["gpt-4o-mini", "gpt-4o"]
model_options_judge = ["gpt-4o", "gpt-4o-mini"]

# Dropdown for the tested LLM model
col1, col2 = st.columns(2)
with col1:
    tested_model = st.selectbox("**Select the LLM model to be tested:**", model_options_test, index=0)

# Dropdown for the judge LLM model
with col2:
    judge_model = st.selectbox("**Select the judge LLM model:**", model_options_judge, index=0)

# Layout for prompt and judge prompt side by side
col1, col2 = st.columns(2)

# Input field for the prompt text
with col1:
    prompt_input = st.text_area(
        "**Enter your prompt for the tested model:**",
        "Provide a response to the following question: \n\"{user_question}\"\nbut the answer must contain the following information:\n\"{fact}\".\nYou are a little sarcastic and extroverted by 40% - not too friendly.",
        height=460  # Adjust the height for a larger text area
    )

# Input field for the judge prompt
with col2:
    judge_prompt = st.text_area(
        "**Enter the middle of the judge prompt:**",
        sample_eval_prompt,
        height=460  # Adjust the height for a larger text area
    )
    st.info("""Thus 'middle' judge prompt must specify the metrics and their value types to assess, e.g.:
            int from 0 to 28,
            string ["good", "bad", "neutral"],
            boolean (true, false).
            The prompt will be formatted with the same fields as the tested prompt / the fields in the dataset.
            Besides the fields that are in the tested prompt, the {LLM_response} field must be added.
            This judge prompt will be appended by instructions for the evaluator.""",
            icon="ℹ️")

if st.button("Run Configurable Test"):
    if prompt_input and judge_prompt:
        if "{LLM_response}" not in judge_prompt:
            st.error("The judge prompt must also include the '{LLM_response}' field.")
        else:
            # Create a progress bar
            progress_bar = st.progress(0)

            # Run the configurable test and get the resulting image, averages, and random responses
            with st.spinner("Running the configurable test..."):
                configurable_test_module = ConfigurableTestModule(f"datasets/{selected_dataset}",
                                                                  prompt_input, judge_prompt,
                                                                  tested_model, judge_model)
                result_image, averages, responses = configurable_test_module.test_configurable_answers(progress_bar)

            # Store results in session state to persist them across layout changes
            st.session_state['result_image'] = result_image
            st.session_state['averages'] = averages
            st.session_state['random_responses'] = responses

            # Indicate completion
            st.success("Configurable test completed!")

# Display the image 1.5x bigger if available in session state
if st.session_state['result_image']:
    st.image(st.session_state['result_image'], caption="Generated Metrics", use_column_width=False, width=900)

with stylable_container(
        key="container_with_border",
        css_styles="""
            {
                border: 1px solid rgba(49, 51, 63, 0.2);
                border-radius: 0.5rem;
                padding: calc(1em - 1px)
            }
            """,
                    ):
    # Display averages if available in session state
    if st.session_state['averages']:
        st.subheader("Average Metrics:")
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

with stylable_container(
        key="container_with_border",
        css_styles="""
            {
                border: 1px solid rgba(49, 51, 63, 0.2);
                border-radius: 0.5rem;
                padding: calc(1em - 1px)
            }
            """,
                    ):
    # Display random responses if available in session state
    if st.session_state['random_responses']:
        st.subheader("Randomly Selected Responses:")
        # select 3 random responses from the list
        random_responses = random.sample(st.session_state['random_responses'], 3)
        for idx, response in enumerate(random_responses):
            st.write(response)
            st.divider()
