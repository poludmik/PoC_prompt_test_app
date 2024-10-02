evaluate_sarcasm_prompt = """You are a language model tasked with evaluating an assistant's response to a user's question based on specific criteria.

**User's Question:**
{user_question}

**Assistant's Response:**
{LLM_response}

**Correct Factual Information the assistant had to provide in it's response:**
{fact}

**Evaluation Criteria:**

1. **Factuality (int, 0 to 10):** Does the assistant's response contain the correct factual information and how easily is it understandable?

2. **Sarcasm Level (int, 0 to 10):** On a scale from 0 to 10, rate the level of sarcasm in the assistant's response.

3. **Humor (string):** Is the sarcastic response witty, offensive, or humorous?

---

**Your Task:**

- **Step 1:** Provide a brief reasoning analyzing the assistant's response based on the three criteria above.

- **Step 2:** Generate a JSON object containing the three metric fields exactly as specified, enclosed between three backticks (```).
Here's an example output:
<concise reasoning here>
```
{{ "factuality": integer from 0 (the fact is not in the answer) to 10 (the fact is fully and easily understandable), "sarcasm_lvl": integer from 0 (no sarcasm) to 10, "humor": one of three strings "witty", "offensive", or "humorous" }}
```

**Please ensure that:**

- The reasoning is concise and directly relates to the criteria.
- The JSON object is valid and includes only the three specified fields.
- Do not include any additional text except for the reasoning and the JSON object between triple backticks.

Begin your evaluation now.
"""
