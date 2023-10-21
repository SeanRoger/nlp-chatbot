import streamlit as st
import pandas as pd
import time
from io import StringIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests

# Initialize 'messages' and 'last_active_time' in session_state if not present
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'last_active_time' not in st.session_state:
    st.session_state.last_active_time = time.time()

# Load data from URL
url = 'https://raw.githubusercontent.com/SeanRoger/nlp-chatbot/main/chatbot_dataset.csv'

try:
    # Fetch data from the URL
    response = requests.get(url)
    response.raise_for_status()
    df = pd.read_csv(StringIO(response.text))
except requests.exceptions.HTTPError as http_err:
    st.error(f'HTTP error occurred: {http_err}')
except Exception as err:
    st.error(f'An error occurred: {err}')
else:
    # Set up title and icons
    st.title("Megiddo Bot")
    user_icon = "🙂"
    bot_icon = "🤖"

    # Display previous messages
    for message in st.session_state.messages:
        if message["role"] == "User":
            st.text(f"{user_icon} User:")
            st.text_area("User", value=message["content"], height=100)
        elif message["role"] == "Megiddo Bot":
            st.text(f"{bot_icon} Megiddo Bot:")
            st.text_area("Megiddo Bot", value=message["content"], height=100)

    # Prompt for new question after 10 seconds
    if time.time() - st.session_state.last_active_time > 10:
        st.text("Please type your next question.")

    # Accept user input
    user_input = st.text_input("User Input", key="user_input")
    if user_input:
        st.session_state.last_active_time = time.time()
        st.session_state.messages = []
        st.session_state.messages.append({"role": "User", "content": user_input})

        try:
            # Process user input
            vectorizer = TfidfVectorizer()
            all_data = list(df['Query']) + [user_input]
            tfidf_matrix = vectorizer.fit_transform(all_data)
            document_vectors = tfidf_matrix[:-1]
            query_vector = tfidf_matrix[-1]
            similarity_scores = cosine_similarity(query_vector, document_vectors)
            sorted_indexes = similarity_scores.argsort()[0][-1:]
            response = df.iloc[sorted_indexes[0]]['Response']
            category = df.iloc[sorted_indexes[0]]['Category']

            # Display bot response and category
            st.text(f"{bot_icon} Megiddo Bot:")
            st.text_area("Megiddo Bot", value=response, height=100)
            st.text_area("Category", value=category, height=100)
            st.session_state.messages.append({"role": "Megiddo Bot", "content": response})

        except KeyError:
            st.error("Invalid data format in the CSV file. Please check the format and try again.")

    # Display the bar graph if DataFrame is present
    if 'df' in locals():
        st.title("Bar Graph")
        category_counts = df['Category'].value_counts()
        st.bar_chart(category_counts)

# Note: No external libraries required for displaying the bar chart in this version.
