import streamlit as st
import pickle
import string
import nltk

nltk.download('punkt')
nltk.download('stopwords')

ps = nltk.stem.PorterStemmer()

try:
    with open('model (1).pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('vectorizer (1).pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
except FileNotFoundError:
    st.error("Error: The 'model (1).pkl' or 'vectorizer (1).pkl' file was not found.")
    st.stop()

def process_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = [i for i in text if i.isalnum()]
    y = [i for i in y if i not in nltk.corpus.stopwords.words('english') and i not in string.punctuation]
    y = [ps.stem(i) for i in y]

    return " ".join(y)

st.title("SMS Spam Classifier")
st.write("Enter an SMS message to check if it is spam or ham.")

input_sms = st.text_area("Enter the SMS message:")

if st.button("Predict"):
    if input_sms:
        transformed_sms = process_text(input_sms)
        vectorized_sms = vectorizer.transform([transformed_sms])
        prediction = model.predict(vectorized_sms)[0]

        if prediction == 'spam':
            st.error("This is a spam message.")
        else:
            st.success("This is not a spam message (ham).")
    else:
        st.warning("Please enter a message to classify.")
