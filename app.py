import streamlit as st
import pandas as pd
import pickle
import string
from nltk.corpus import stopwords
import nltk
from PIL import Image
from nltk.stem.porter import PorterStemmer

image = Image.open(r'C:\Users\pc\ML\Projects\Recc Book\myimage.jpg')

ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    
    text = y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    text = y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
    
            
    return " ".join(y)

tfidf = pd.read_pickle(r'C:\Users\pc\ML\Projects\Spam project\vectorizer.pkl')
model = pd.read_pickle(r'C:\Users\pc\ML\Projects\Spam project\model.pkl')

st.title("Email/SMS Spam Classifier")

st.sidebar.write('Built By -')
st.sidebar.title('Rishabh Vyas')
st.sidebar.image(image,caption='Machine Learning Engineer',width=160)
st.sidebar.write('E-mail - rishabhvyas472@gmail.com')

input_sms = st.text_area("Enter the message")

if st.button('Predict'):

    # 1. preprocess
    transformed_sms = transform_text(input_sms)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms]).toarray()
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")