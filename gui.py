import tkinter as tk
from tkinter import ttk, messagebox
import string
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import requests
from urllib.parse import urlparse
from sklearn.exceptions import NotFittedError
import pandas as pd
from PIL import Image, ImageTk

# Download NLTK stopwords
nltk.download('stopwords')

class SpamDetectorApp:
    def __init__(self, master):
        self.master = master
        master.title("Welcome to Safe Link")

        # Load and display gif image
        img_path = "/Users/helitharajapaksha/Desktop/SmsProject/venv/Safe-Link.jpg"
        img = Image.open(img_path)
        img = ImageTk.PhotoImage(img)
        panel = tk.Label(master, image=img)
        panel.image = img  # Keep a reference to the image to prevent garbage collection
        panel.pack()

        self.label = tk.Label(master, text="Enter an SMS:", font=("Arial", 14, "bold"))
        self.label.pack(pady=10)  # Increase spacing

        self.entry = tk.Entry(master, width=50, font=("Arial", 12))  # Increase font size
        self.entry.pack(pady=10)  # Increase spacing

        # Create a styled button
        style = ttk.Style()
        style.configure("TButton",
                        foreground="black",
                        background="lightgray",
                        font=("Arial", 12, "bold"),
                        padding=10,
                        )

        self.detect_button = ttk.Button(master, text="Detect Spam", command=self.detect_spam, style="TButton")
        self.detect_button.pack(pady=10)  # Increase spacing

        # New label to display copied text
        self.copied_text_label = tk.Label(master, text="", font=("Arial", 12))
        self.copied_text_label.pack(pady=10)  # Increase spacing

        # Initialize TF-IDF vectorizer and Naive Bayes classifier
        self.tfidf_vectorizer = TfidfVectorizer()
        self.naive_bayes = MultinomialNB()

        # Load your data and train the model
        self.load_and_train_model()

        # Footer
        footer_label = tk.Label(master, text="This app created by Helitha Rajapakse, Danusha Dewmin, and Buddhini", font=("Arial", 10))
        footer_label.pack(side=tk.BOTTOM)

        # Add button hover events
        self.detect_button.bind("<Enter>", self.on_button_hover)
        self.detect_button.bind("<Leave>", self.on_button_leave)

    def preprocess(self, sms):
        sms = re.sub(r'http\S+', 'URL_PLACEHOLDER', sms)
        remove_punct = "".join([char.lower() for char in sms if char not in string.punctuation])
        tokenize = word_tokenize(remove_punct)
        remove_stopwords = [word for word in tokenize if word not in stopwords.words('english')]
        return ' '.join(remove_stopwords)

    def is_secure_link(self, link):
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

    def train_model(self, data):
        X_tfidf = self.tfidf_vectorizer.fit_transform(data['sms'])
        y = data['label']
        self.naive_bayes.fit(X_tfidf, y)

    def load_and_train_model(self):
        # Replace 'Dataset.txt' with the actual filename or path to your dataset
        file_path = "/Users/helitharajapaksha/Desktop/SmsProject/venv/Dataset.txt"

        try:
            # Load your data
            data = pd.read_csv(file_path, sep='\t', header=None, names=["label", "sms"])

            # Train the model with your data
            self.train_model(data)
        except FileNotFoundError:
            messagebox.showerror("Error", "Dataset file not found. Please check the file path.")

    def predict_spam(self, message):
        processed_message = self.preprocess(message)
        try:
            input_tfidf = self.tfidf_vectorizer.transform([processed_message])
        except NotFittedError:
            messagebox.showwarning("Training Required", "Please train the model before detecting spam.")
            return None

        prediction = self.naive_bayes.predict(input_tfidf)[0]
        return prediction

    def detect_spam(self):
        user_input = self.entry.get()

        # Ensure that tfidf_vectorizer is fitted before using it
        if not hasattr(self.tfidf_vectorizer, 'vocabulary_'):
            messagebox.showwarning("Training Required", "Please train the model before detecting spam.")
            return

        secure_link_found = False
        for word in user_input.split():
            if word.startswith('http') or word.startswith('www'):
                secure_link_found, safe_link = self.is_secure_link(word)
                break

        prediction = self.predict_spam(user_input)

        if prediction is not None:
            if prediction == 'ham':
                messagebox.showinfo("Spam Detection", "This message is original.")
            else:
                messagebox.showwarning("Spam Detection", "This message is fake.")

            if secure_link_found:
                messagebox.showinfo("Link Security", "Safe-link: {}".format(safe_link))
            else:
                messagebox.showinfo("Link Security", "No secure link found.")

            # Update the copied text label
            self.copied_text_label.config(text=f"Copied Text: {user_input}")

    def on_button_hover(self, event):
        self.detect_button.configure(background="lightblue")

    def on_button_leave(self, event):
        self.detect_button.configure(background="lightgray")

# Create an instance of Tkinter
root = tk.Tk()

# Create an instance of your SpamDetectorApp
app = SpamDetectorApp(root)

# Run the Tkinter event loop
root.mainloop()
