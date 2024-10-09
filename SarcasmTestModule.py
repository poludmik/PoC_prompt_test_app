import datetime
import json
import matplotlib.pyplot as plt
from LLMCalls import LLMCalls
from Prompts import evaluate_sarcasm_prompt
import random

class SarcasmTestModule:

    def __init__(self, test_set_path, prompts, model_name):
        self.test_set_path = test_set_path
        self.model_name = model_name
        self.prompts = prompts

        # read jsonl file
        self.test_set = []
        with open(test_set_path, 'r') as f:
            for line in f:
                self.test_set.append(json.loads(line))
        # take first 10 samples for testing
        # self.test_set = self.test_set[:10]

    def test_sarcasm_answers(self, progress_bar=None):
        """
        Tests the model on the test set with multiple metrics and updates the progress bar.
        1. Factuality (0-10)
        2. Sarcasm level (0-10)
        3. Humor type (witty, offensive, humorous)
        4. Response length in words
        """
        factuality = []
        sarcasm_lvl = []
        humor = []
        response_length = []
        all_responses = []
        total_tests = len(self.test_set)  # Number of tests to be performed
        
        for prompt_name, prompt in self.prompts.items():
            print(f"\033[94mPrompt: \033[0m{prompt}")
            
            for test_idx, test in enumerate(self.test_set):
                system_message = "You are a helpful assistant."
                user_message = prompt.format(user_question=test["User"], fact=test["Main information to provide"])

                response = LLMCalls.call_openai_chat_completion(self.model_name, user_message, system_message)
                print(f"\033[92mResponse: \033[0m{response}")
                all_responses.append(response)
                
                n_words = len(response.split())
                print(f"Number of words: {n_words}")

                # Evaluate the response
                evaluation_prompt = evaluate_sarcasm_prompt.format(user_question=test["User"], LLM_response=response, fact=test["Main information to provide"])
                evaluation_response = LLMCalls.call_openai_chat_completion("gpt-4o", evaluation_prompt, system_message)
                backticks = evaluation_response.split("```")
                json_object = backticks[1].strip()
                evaluation = json.loads(json_object)
                print(f"\033[93mEvaluation: \033[0m{evaluation}")
                
                factuality.append(evaluation["factuality"])
                sarcasm_lvl.append(evaluation["sarcasm_lvl"])
                humor.append(evaluation["humor"])
                response_length.append(n_words)

                # Update progress bar
                progress = (test_idx + 1) / total_tests
                if progress_bar:
                    progress_bar.progress(progress)  # Update the progress bar

            # Get random responses for display
            random_responses = random.sample(all_responses, 3)

            humor_summary = {
                "witty": humor.count("witty") / len(humor),
                "offensive": humor.count("offensive") / len(humor),
                "humorous": humor.count("humorous") / len(humor)
            }
            
            # Calculate average metrics
            avg_factuality = sum(factuality) / len(factuality)
            avg_sarcasm_lvl = sum(sarcasm_lvl) / len(sarcasm_lvl)
            avg_response_length = sum(response_length) / len(response_length)

            averages = {
                "factuality": avg_factuality,
                "sarcasm_level": avg_sarcasm_lvl,
                "response_length": avg_response_length,
                "humor_summary": humor_summary
            }

            # Generate the plots
            factuality_histogram = {i: factuality.count(i) for i in range(11)}
            sarcasm_lvl_histogram = {i: sarcasm_lvl.count(i) for i in range(11)}
            max_length = max(response_length)
            response_length_histogram = {i: sum(1 for length in response_length if i <= length < i + 10) for i in range(0, max_length, 10)}

            # Call the new function to create the plots
            image_name = self.create_metrics_histogram(prompt_name, factuality_histogram, sarcasm_lvl_histogram, response_length_histogram, humor_summary)

        return image_name, averages, random_responses
    
    def create_metrics_histogram(self, prompt_name, factuality_histogram, sarcasm_lvl_histogram, response_length_histogram, humor_summary):
        """
        Creates and saves a 2x2 plot with histograms for factuality, sarcasm level, response length, and humor summary.
        """
        fig, axs = plt.subplots(2, 2, figsize=(10, 8))
        fig.suptitle(f'{prompt_name} - Metrics Histograms', fontsize=16)

        # Factuality plot
        axs[0, 0].bar(factuality_histogram.keys(), factuality_histogram.values(), color='skyblue', edgecolor='black', width=0.7)
        axs[0, 0].set_title('Factuality', fontsize=12)
        axs[0, 0].set_xlabel('Score', fontsize=10)
        axs[0, 0].set_ylabel('Frequency', fontsize=10)
        axs[0, 0].grid(True)

        # Sarcasm level plot
        axs[0, 1].bar(sarcasm_lvl_histogram.keys(), sarcasm_lvl_histogram.values(), color='salmon', edgecolor='black', width=0.7)
        axs[0, 1].set_title('Sarcasm Level', fontsize=12)
        axs[0, 1].set_xlabel('Score', fontsize=10)
        axs[0, 1].set_ylabel('Frequency', fontsize=10)
        axs[0, 1].grid(True)

        # Response length plot
        axs[1, 0].bar(response_length_histogram.keys(), response_length_histogram.values(), color='lightgreen', edgecolor='black', width=8)
        axs[1, 0].set_title('Response Length', fontsize=12)
        axs[1, 0].set_xlabel('Length (words)', fontsize=10)
        axs[1, 0].set_ylabel('Frequency', fontsize=10)
        axs[1, 0].grid(True)

        # Humor summary plot
        humor_types = ['witty', 'offensive', 'humorous']
        humor_vals = [humor_summary['witty'], humor_summary['offensive'], humor_summary['humorous']]
        axs[1, 1].bar(humor_types, humor_vals, color='orange', edgecolor='black', width=0.5)
        axs[1, 1].set_title('Humor Summary', fontsize=12)
        axs[1, 1].set_xlabel('Type', fontsize=10)
        axs[1, 1].set_ylabel('Proportion', fontsize=10)
        axs[1, 1].grid(True)

        # Adjust layout and save
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        image_name = f"plots/{prompt_name}_metrics_" + time + ".png"
        plt.savefig(image_name)
        plt.show()

        return image_name


def main():
    test_set_path = "datasets/dummy_qa_dataset.jsonl"
    prompts = [
        "Write a sarcastic response to the following question: {user_question} but the answer must contain the following information {fact}.",
        "Respond with the maximum degree of sarcasm to the following question: {user_question} but the answer must contain the following information {fact}."
    ]
    
    prompt_dict_with_names = {"prompt_1": prompts[0], "prompt_2": prompts[1]}
    
    model_name = "gpt-4o-mini"
    sarcasm_test_module = SarcasmTestModule(test_set_path, prompt_dict_with_names, model_name)
    sarcasm_test_module.test_sarcasm_answers()

if __name__ == "__main__":
    main()
