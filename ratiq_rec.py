pip install -q librosa gtts streamlit
%%capture
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from unsloth import FastLanguageModel
def setup_model():
base_model_name = "unsloth/Meta-Llama-3.1-8B-bnb-4bit"
adapter_name = "Ratiq/DecathlonFT"
model_path = "./Llama2-Model/"
print("Loading base model and adapter separately...")
# Load the base model
base_model = AutoModelForCausalLM.from_pretrained(
base_model_name,
device_map="auto",
trust_remote_code=True,
load_in_4bit=True
)
# Load the adapter
model = PeftModel.from_pretrained(base_model, adapter_name)
# Save the model and adapter
model.save_pretrained(model_path)
print("Model and adapter saved.")
9
# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.save_pretrained(model_path)
print("Tokenizer saved.")
print("Model setup complete.")
setup_model()
%%writefile gif2.py
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
# Define a specific prompt with more detailed and humanlike responses
ismo_prompt = """You are an expert bicycle advisor for Decathlon, and a customer is considering a particular,bicycle model. Please provide a detailed and humanlike response that includes:
1. A comprehensive and engaging description of the key features of the Rockrider ST100, explaining its functionality, and why it stands out.
2. Recommend 2 other similar bicycle models from Decathlon's collection that offer comparable features. Explain how they are similar to the Rockrider ST100.
3. For each of the similar models, give a detailed comparison, highlighting the following:
- Key similarities that make them good alternatives.
- Important differences that might influence the customer’s choice.
- Advantages or disadvantages of choosing one model over the other.
4. Summarize your response with a conclusion on how the Rockrider ST100 fits into Decathlon's broader emphasizing its unique value for the customer.
Be friendly, conversational, and make sure to keep your explanations clear and informative for someone who,may not be an expert in bicycles.
{0}
10
Response:"""
@st.cache_resource
def load_model():
try:
model_name = "/content/Llama2-Model" # Your fine-tuned model path
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
return model, tokenizer
except Exception as e:
st.error(f"Error loading model: {str(e)}")
return None, None
def generate_response(model, tokenizer, prompt):
inputs = tokenizer([ismo_prompt.format(prompt)], return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=300)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
# Remove the input prompt from the response
response = response.replace(ismo_prompt.format(prompt), "").strip()
return response
def main():
st.set_page_config(layout="wide")
st.title("Decathlon AI")
model, tokenizer = load_model()
if model is None or tokenizer is None:
11
st.error("Failed to load the model. Please check your model path or internet connection.")
return
static_image_path = "/content/1200px-Decathlon_Logo.png" # Path to your static image file
# Create a two-column layout
col1, col2 = st.columns([3, 1])
with col1:
# Initialize chat history
if "messages" not in st.session_state:
st.session_state.messages = []
# Display only AI responses from chat history
for message in st.session_state.messages:
if message["role"] == "assistant":
with st.chat_message(message["role"]):
st.markdown(message["content"])
# React to user input
if prompt := st.chat_input("Ask a question about Decathlon Cycle's:"):
# Add user message to chat history (for context, but not displayed)
st.session_state.messages.append({"role": "user", "content": prompt})
with st.spinner("Generating response..."):
response = generate_response(model, tokenizer, prompt)
# Display
with st.chat_message("assistant"):
st.markdown(response)
12
# Add assistant response to chat history
st.session_state.messages.append({"role": "assistant", "content": response})
with col2:
# Display the static image by default
st.image(static_image_path, width=200)
if __name__ == "__main__":
main()