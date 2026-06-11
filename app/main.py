import streamlit as st
import joblib
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
import nltk
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()


tfidf = joblib.load('models/tfidf_vectorizer.joblib')
lr_model = joblib.load('models/logistic_regression.joblib')

def clean_tweet(text):

    text = text.lower()
    #(pattern , replacement , text)
    #meaning of r'http\S+: r''->treats backslashes literally , \S -> any non whitespace character , + means one or more of the previous thing
    #http\S+ means: "find http followed by one or more non-space characters
    text = re.sub(r'http\S+' , '' , text)
    text = re.sub(r'@\S+' , '' , text)
    text = re.sub(r'#' , '' , text)
    text = re.sub(r'&\S+;', '', text)
    text = re.sub(r'\d', '', text)
    text = re.sub(r'[^a-z\s]', '', text)

    words = text.split()
    words = [word for word in words if word not in stop_words]
    text = ' '.join(words)

    words = text.split()
    words = [stemmer.stem(word) for word in words]
    text = ' '.join(words)


    return text




st.title("Twitter Sentiment Classifier")

tweet = st.text_input("Enter your tweet here to classify its sentiment")
predict = st.button("Predict")

if predict:

    cleaned_text = clean_tweet(tweet)
    vectorized = tfidf.transform([cleaned_text])
    result = lr_model.predict(vectorized)[0]

    if result==1:
        st.write("Positive Tweet")
    else:
        st.write("Negative Tweet")



