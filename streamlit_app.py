import streamlit as st
import pickle
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.stem import  PorterStemmer 
import re

model = pickle.load(open('pickle_files/model.pkl','rb')) 
vectorizer = pickle.load(open('pickle_files/tfidf_vectorizer.pkl','rb')) 

nltk.download('stopwords')

sw = set(stopwords.words('english'))
def text_preprocessing(text):
    txt = TextBlob(text)
    result = txt.correct()
    removed_special_characters = re.sub("[^a-zA-Z]", " ", str(result))
    tokens = removed_special_characters.lower().split()
    stemmer = PorterStemmer()
    
    cleaned = []
    stemmed = []
    
    for token in tokens:
        if token not in sw:
            cleaned.append(token)
            
    for token in cleaned:
        token = stemmer.stem(token)
        stemmed.append(token)

    return " ".join(stemmed)

def text_classification(text):
    if len(text) < 1:
        st.write("  ")
    else:
        with st.spinner("Classification in progress..."):
            cleaned_review = text_preprocessing(text)
            process = vectorizer.transform([cleaned_review]).toarray()
            prediction = model.predict(process)
            p = ''.join(str(i) for i in prediction)
        
            if p == 'True':
                st.success("The review entered is by a real user.")
            if p == 'False':
                st.error("The review entered is by a fake user.")

def main():
    st.title("Amazon Fake Reviews Model")
    
    st.subheader("About")
    if st.checkbox("About Classifer"):
        st.markdown('**Model:** Stochastic Gradient Descent')
        st.markdown('**Vectorizer:** Count')
        st.markdown('**Test-Train splitting:** 30% - 70%')
        st.markdown('**Stemmer:** PorterStemmer')
        
    if st.checkbox("Evaluation Results"):
        st.markdown('**Accuracy:** 87.6%')
        st.markdown('**Precision:** 89.7%')
        st.markdown('**Recall:** 85.0%')
        st.markdown('**F-1 Score:** 87.2%')

    st.subheader("Fake Review Classifier")
    review = st.text_area("Enter Review: ")
    if st.button("Check"):
        text_classification(review)
       
main()