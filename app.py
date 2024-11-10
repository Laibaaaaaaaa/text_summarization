import streamlit as st
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Load model and tokenizer
model = T5ForConditionalGeneration.from_pretrained('t5-small')
tokenizer = T5Tokenizer.from_pretrained('t5-small')

# Summarization function
def summarize(text):
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(inputs, max_length=150, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Set Streamlit page config
st.set_page_config(page_title="Text Summarization Tool", page_icon="📝", layout="wide")

# Add custom CSS for the app styling
st.markdown("""
    <style>
        body {
            background-color: #f4f4f9;
            font-family: 'Arial', sans-serif;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            font-size: 16px;
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        .stTextArea textarea {
            font-size: 16px;
            padding: 10px;
            border-radius: 8px;
            border: 1px solid #ddd;
        }
        .stTextArea textarea:focus {
            outline-color: #4CAF50;
        }
        .stMarkdown {
            color: #333;
        }
    </style>
    """, unsafe_allow_html=True)

# Streamlit app layout
st.title("📝 Text Summarization Tool")
st.markdown("""
    This tool automatically summarizes long articles or documents into concise summaries.
    Enter some text and let the tool do the rest!
""")

user_input = st.text_area("Enter the text you want to summarize", "", height=200)

col1, col2 = st.columns([1, 3])

# Column 1: Add a 'Clear' button
with col1:
    clear_button = st.button("Clear", key="clear")

# Column 2: Summarize button and output
with col2:
    if st.button("Summarize"):
        if user_input:
            with st.spinner('Summarizing...'):
                summary = summarize(user_input)
                st.subheader("Summary:")
                st.write(summary)
        else:
            st.write("Please enter text to summarize.")

    if clear_button:
        st.experimental_rerun()  # Clear input on button click

