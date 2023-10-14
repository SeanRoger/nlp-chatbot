import streamlit as st
import pandas as pd
import time
from io import StringIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests

# Initialize 'messages' in session_state if not present
if 'messages' not in st.session_state:
    st.session_state.messages = []

url = 'chatbot_dataset.csv'

try:
    response = requests.get(url)
    response.raise_for_status()
    df = pd.read_csv(StringIO(response.text))
except requests.exceptions.HTTPError as http_err:
    st.error(f'HTTP error occurred: {http_err}')
except Exception as err:
    st.error(f'An error occurred: {err}')
else:
    st.title("Megiddo Bot")

    # User and bot icons
    user_icon = "🙂"
    bot_icon = "🤖"

    for message in st.session_state.messages:
        if message["role"] == "User":
            st.text(f"{user_icon} User:")
            st.text_area("User", value=message["content"], height=100)
        elif message["role"] == "Megiddo Bot":
            st.text(f"{bot_icon} Megiddo Bot:")
            st.text_area("Megiddo Bot", value=message["content"], height=100)

    if prompt := st.text_input("User Input"):
        st.session_state.messages.append({"role": "User", "content": prompt})
        
        try:
            vectorizer = TfidfVectorizer()
            all_data = list(df['Query']) + [prompt]
            tfidf_matrix = vectorizer.fit_transform(all_data)
            document_vectors = tfidf_matrix[:-1]
            query_vector = tfidf_matrix[-1]
            similarity_scores = cosine_similarity(query_vector, document_vectors)
            sorted_indexes = similarity_scores.argsort()[0][-1:]
            response = df.iloc[sorted_indexes[0]]['Response']
            category = df.iloc[sorted_indexes[0]]['Category']
            
            st.text(f"{bot_icon} Megiddo Bot:")
            st.text_area("Megiddo Bot", value=response, height=100)
            st.text_area("Category", value=category, height=100)
            st.session_state.messages.append({"role": "Megiddo Bot", "content": response})

            # Introduce a delay of 10 seconds before prompting the user for the next question
            time.sleep(10)
            st.text("Please type your next question.")
        except KeyError:
            st.error("Invalid data format in the CSV file. Please check the format and try again.")
