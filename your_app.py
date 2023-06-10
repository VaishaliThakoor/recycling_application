import streamlit as st

st.title("Simple Streamlit App")

# Text input
user_input = st.text_input("Enter some text")

# Echo the entered text
st.write("You entered:", user_input)

