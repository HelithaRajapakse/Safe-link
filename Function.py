import string
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import pandas as pd

# Initialize NLTK and download necessary resources
nltk.download('punkt')
nltk.download('stopwords')

# Load stopwords and punctuation
stop_words = set(stopwords.words('english'))
punctuation = string.punctuation

# Preprocess SMS content, preserving URLs
def preprocess(sms):
    # Replace URLs with a placeholder
    sms = re.sub(r'http\S+', 'URL_PLACEHOLDER', sms)
    remove_punct = "".join([char.lower() for char in sms if char not in punctuation])
    tokenize = word_tokenize(remove_punct)
    remove_stopwords = [word for word in tokenize if word not in stop_words]
    return ' '.join(remove_stopwords)

# Check if a link is secure and safe
def is_secure_link(link):
    parsed_url = urlparse(link)
    if parsed_url.scheme == 'https':
        try:
            response = requests.get(link)
            if response.status_code == 200:
                return True, "Secure"
            else:
                return False, "Not Secure (HTTP Status {})".format(response.status_code)
        except requests.exceptions.RequestException as e:
            return False, "Not Secure (Error: {})".format(e)
    return False, "Not a secure link"

# Function to predict if a message is spam or not
def predict_spam(message, tfidf_vectorizer, naive_bayes):
    processed_message = preprocess(message)
    input_tfidf = tfidf_vectorizer.transform([processed_message])
    prediction = naive_bayes.predict(input_tfidf)[0]
    return prediction

# Load local data
file_path = '/Users/helitharajapaksha/Desktop/SmsProject/venv/Dataset.txt'
data = pd.read_csv(file_path, sep='\t', header=None, names=["label", "sms"])

# Module Training using TF-IDF
tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(data['sms'])
y = data['label']

# Train a Naive Bayes classifier
naive_bayes = MultinomialNB()
naive_bayes.fit(X_tfidf, y)

# User input for prediction
user_input = input("Please type a message to check if it's original or fake:\n")
secure_link_found = False
for word in user_input.split():
    if word.startswith('http') or word.startswith('www'):
        secure_link_found, safe_link = is_secure_link(word)
        break

# Predict user input
prediction = predict_spam(user_input, tfidf_vectorizer, naive_bayes)

# Output the prediction result with probabilities and secure link information
if prediction == 'ham':
    print("This message is original.")
else:
    print("This message is fake.")

if secure_link_found:
    print("Safe-link: {}".format(safe_link))
else:
    print("No secure link found.")
