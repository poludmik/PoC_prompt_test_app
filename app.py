import streamlit as st
from SarcasmTestModule import SarcasmTestModule
import os


# Define the function to run the sarcasm test and get the resulting image, averages, and responses
def run_sarcasm_test(prompt_input, prompt_name, progress_bar):
    # Define dummy test set and prompts
    test_set_path = "datasets/dummy_qa_dataset.jsonl"
    prompts = {prompt_name: prompt_input}
    
    model_name = "gpt-4o-mini"
    
    sarcasm_test_module = SarcasmTestModule(test_set_path, prompts, model_name)
    image_name, averages, random_responses = sarcasm_test_module.test_sarcasm_answers(progress_bar)  # Pass progress bar to the test function
    
    # Assuming that the last image saved by sarcasm test will be the result
    image_dir = "plots"
    images = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith(".png")]
    latest_image = max(images, key=os.path.getctime)  # Get the latest image created

    return latest_image, averages, random_responses

# Initialize session state to persist results
if 'random_responses' not in st.session_state:
    st.session_state['random_responses'] = []
if 'averages' not in st.session_state:
    st.session_state['averages'] = {}
if 'result_image' not in st.session_state:
    st.session_state['result_image'] = ""

st.set_page_config(layout="centered")

# Streamlit app UI
st.title("Sarcasm Prompt Test :bar_chart:")
st.write("Enter a prompt to test the sarcasm levels through corresponding metrics.")

# Input field for the prompt name
prompt_name = st.text_input("Enter a name for this prompt:", "My new prompt")

# Input field for the prompt text (making it larger by default)
prompt_input = st.text_area(
    "Enter your prompt:", 
    "Write a sarcastic response to the following question: \n\"{user_question}\"\nbut the answer must contain the following information:\n\"{fact}\".",
    height=250  # Adjust the height for a larger text area
)

if st.button("Run Sarcasm Test"):
    if prompt_input and prompt_name:
        # Create a progress bar
        progress_bar = st.progress(0)
        
        # Run the sarcasm test and get the resulting image, averages, and random responses
        with st.spinner("Running the sarcasm test..."):
            result_image, averages, random_responses = run_sarcasm_test(prompt_input, prompt_name, progress_bar)
        
        # Store results in session state to persist them across layout changes
        st.session_state['result_image'] = result_image
        st.session_state['averages'] = averages
        st.session_state['random_responses'] = random_responses

        # Indicate completion
        st.success("Sarcasm test completed!")

# Display the image 1.5x bigger if available in session state
if st.session_state['result_image']:
    st.image(st.session_state['result_image'], caption="Generated Metrics", use_column_width=False, width=900)

# Display averages if available in session state
if st.session_state['averages']:
    st.subheader("Average Metrics")
    st.write(f"**Factuality (0-10)**: {st.session_state['averages']['factuality']:.2f}")
    st.write(f"**Sarcasm Level (0-10)**: {st.session_state['averages']['sarcasm_level']:.2f}")
    st.write(f"**Response Length (words)**: {st.session_state['averages']['response_length']:.2f}")
    st.write(f"**Humor Types Summary**: {st.session_state['averages']['humor_summary']}")

# Display random responses if available in session state
if st.session_state['random_responses']:
    st.sidebar.subheader("Random LLM Responses")
    for i, response in enumerate(st.session_state['random_responses'], 1):
        st.sidebar.write(f"**Response {i}:** {response}\n")
