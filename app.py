from tensorflow.keras.models import load_model 
from preprocessing import preprocess_reviews
import streamlit as st
import pickle

st.title("Product Review Analysis")

model = load_model('models/LSTM.h5')
token = pickle.load((open('token.pkl', 'rb')))

review = st.text_area("Enter the review")
review = str(review)
max_len = 130


if st.button("Find Rating "):
    review_encoded = preprocess_reviews(review, max_len,token)
    result = model.predict(review_encoded)
    
    if result >= 0.8:
        st.header('*****')
    elif result>0.65:
        st.header('****')
    elif result>0.5:
        st.header('***')
    elif result>0.3:
        st.header('**')
    else:
        st.header('*')
        
        
        
        



