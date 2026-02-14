import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import joblib
import string

punc = string.punctuation

def preprocess_text(sentences):
    cleaned_sentences = []
    for sentence in sentences:
        text = sentence.lower()
        tokens = word_tokenize(text)
        punctuation_filter = [word for word in tokens if word not in punc]
        eng_stopwords = stopwords.words("english")
        filtered_tokens = [word for word in punctuation_filter if word not in eng_stopwords]
        wnet = WordNetLemmatizer()
        lemmatized_words = []
        for word in filtered_tokens:
            lemmatized_words.append(wnet.lemmatize(word,"v"))
        cleaned_text = " ".join(lemmatized_words)
        cleaned_sentences.append(cleaned_text) 
    
    return cleaned_sentences

def predict_intent(user_input):
    vectorizer = joblib.load("tfidf.pkl")
    logistic = joblib.load("intent_clf_model.pkl")

    processed = preprocess_text([user_input])
    user_vector = vectorizer.transform(processed)
    prediction = logistic.predict(user_vector)
    # print(prediction)
    return prediction[0]