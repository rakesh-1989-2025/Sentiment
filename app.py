import streamlit as st
import joblib
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

joblib.dump(vectorizer, 'vectorizer.pkl')
st.title("Sentiment Analysis App")
st.write("This app predicts the sentiment of a given text.")    

text = st.text_area("Enter text here:")

if st.button("Predict"):
    if text:
        # Preprocess the text
        text_vectorized = vectorizer.transform([text])
        
        # Make prediction
        prediction = model.predict(text_vectorized)
        
        # Display the result
        if prediction == 1:
            st.success("Positive sentiment")
        else:
            st.error("Negative sentiment")
    else:
        st.warning("Please enter some text to analyze.")
    
