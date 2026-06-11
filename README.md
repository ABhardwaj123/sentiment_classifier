### Twitter Sentiment Classifier

A Machine Learning project that classifies the sentiment of a tweet as Positive or Negative using NLP techniques and classical ML models.


## Dataset
Sentiment140 — ~1.5M real tweets labeled as positive or negative.


### Tech Stack
ML: scikit-learn, pandas, numpy , nltk
Frontend: Streamlit


### Model Comparision between Naive Bayes and Logistic Regression

Model           Accuracy            Precision            Recall              F1

Naive Bayes        75.1%              75%                 75%               75%

Logistic Reg.       76.9%             75.8%               78.8%             77.3%



### What I did
- Analyzed class distribution of tweets , their length and frequencies
- Built a text-cleaning pipleline using regex and NLTK
- Converted text into vectorized form using TF-IDF
- Trained and compared Naive Bayes and Logistic Regression
- Saved the better model using joblib and built a UI using streamlit
