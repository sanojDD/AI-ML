import streamlit as st
import pickle

# Load the vectorizer and model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Streamlit UI
st.set_page_config(page_title="SMS Spam Classifier", layout="centered")
st.title("📩 SMS Spam Classifier")
st.write("Enter an SMS message below to check if it's spam or not.")

# Input box
input_sms = st.text_area("Enter your message here:")

# Prediction button
if st.button("Classify"):
    if input_sms.strip() == "":
        st.warning("Please enter a message.")
    else:
        # Preprocess
        transformed_sms = tfidf.transform([input_sms])
        result = model.predict(transformed_sms)[0]

        # Output
        if result == 1:
            st.error("🚫 Spam Message")
        else:
            st.success("✅ Not Spam")
