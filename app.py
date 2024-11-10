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

# Streamlit app layout
st.title("Text Summarization Tool")
user_input = st.text_area("Enter the text you want to summarize", "", height=200)

if st.button("Summarize"):
    if user_input:
        summary = summarize(user_input)
        st.subheader("Summary:")
        st.write(summary)
    else:
        st.write("Please enter text to summarize.")
