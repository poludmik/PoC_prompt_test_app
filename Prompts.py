start_of_eval_prompt = """You are a language model tasked with evaluating an assistant's response to a user's question based on specific criteria.\n"""

sample_eval_prompt = """**User's Question:** "{user_question}"

**Assistant's Response:** "{LLM_response}"

**Correct Factual Information the assistant had to provide in it's response:** "{fact}"

**Evaluation Criteria:**
1. **Factuality (int, 0 to 10):** Does the assistant's response contain the correct factual information and how easily is it understandable?
2. **Sarcasm Level (int, 0 to 10):** On a scale from 0 to 10, rate the level of sarcasm in the assistant's response.
3. **Humor (string):** Is the sarcastic response witty, offensive, or humorous?
"""

end_of_eval_prompt = """
**Your Task:**
- **Step 1:** Provide a brief reasoning analyzing the assistant's response based on the criteria above.
- **Step 2:** Generate a JSON object containing the metric fields exactly as specified, enclosed between three backticks (```).
Here's an example output:
<concise reasoning here>
```
{{ "metric_1": your_value (for example integer from 0 to 10), "metric_2": your_evaluation_value_2, ..., "metric_n": could be e.g. a string }}
```
Name the metrics in the JSON object as follows:
{metric_names}

**Please ensure that:**
- The reasoning is concise and directly relates to the criteria.
- The JSON object is valid and includes only the specified fields.
- Do not include any additional text except for the reasoning and the JSON object between triple backticks.

Begin your evaluation now.
"""


evaluate_prompt_differences = """You are a language model tasked with evaluating the differences between three prompts and their corresponding responses.
Answer with a short analysis of how the structure of each prompt affects the respective generated responses based on response quality and relevance of each prompt.
Try to identify the features that are different across the responses and how they correlate with the prompts.
Provide a couple of examples from the generated responses to support your hypothesis. Say what prompt format fits best and for what.
"""

extract_metrics_from_the_judge_prompt = """You are an assistant that extracts the metric names from the judge prompt. Return a list object between backticks containing the extracted metric names. 
Example output:
```
["factuality", "humor level", "extroversion"]
```

The judge prompt is as follows:
{judge_prompt}
"""