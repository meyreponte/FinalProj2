import streamlit as st

# Title of the app
st.title('My First Streamlit App')

# A simple text output
st.write('Hello, Streamlit!')

# Taking input from the user
user_input = st.text_input("Enter your name", '')

# Displaying input back to the user
if user_input:
    st.write(f'Hello {user_input}!')

# Button to click
if st.button('Say Hello'):
    st.write('Hello there!')

# A simple slider
age = st.slider('How old are you?', 0, 130, 25)
st.write("I'm ", age, 'years old')
