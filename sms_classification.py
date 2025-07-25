# sms_classifier.py
# This script loads SMS data, performs analysis, and trains a model to classify
# messages as 'promotional' or 'service'.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
import random
from wordcloud import WordCloud, STOPWORDS

# Tools from scikit-learn for building the model
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# NLTK tools for text processing
from nltk.corpus import stopwords

def download_nltk_data():
    """Downloads necessary NLTK data if not already present."""
    try:
        stopwords.words('english')
    except LookupError:
        print("Downloading NLTK stopwords...")
        nltk.download('stopwords')
        print("Download complete.")

def load_and_clean_data(filepath):
    """Loads data from a CSV file and performs initial cleaning."""
    try:
        df = pd.read_csv(filepath, on_bad_lines='skip')
    except FileNotFoundError:
        print(f"Error: '{filepath}' not found.")
        return None

    if df.shape[1] < 2:
        print("Error: The CSV file does not have at least two columns.")
        return None

    df = df.iloc[:, [0, 1]]
    df.columns = ['sms', 'category']

    df.dropna(subset=['sms', 'category'], inplace=True)
    df['sms'] = df['sms'].astype(str)
    df['category'] = df['category'].str.lower().str.strip()
    df = df[df['category'].isin(['promotional', 'service'])]

    print("--- Data successfully loaded and cleaned. ---")
    print(f"Original message counts:\n{df['category'].value_counts()}")
    return df

def augment_service_messages(df):
    """
    Balances the dataset by generating new service messages from templates,
    ensuring a wide variety of service message types are created.
    """
    print("\n--- Augmenting service messages using template generation... ---")
    
    promo_count = df['category'].value_counts().get('promotional', 0)
    service_count = df['category'].value_counts().get('service', 0)

    if promo_count <= service_count:
        print("No augmentation needed as service messages are not the minority class.")
        return df

    # Templates for generating new, diverse service messages
    templates = {
        'logistics': [
            "Your {item} with tracking number {number} has been {status}.",
            "Update on your order {number}: it is now {status}.",
            "Your delivery of item {number} is scheduled for today.",
            "The parcel {number} has been dispatched from our warehouse."
        ],
        'billing': [
            "Your bill of Rs. {amount} for your {service} is now due.",
            "Payment reminder: Rs. {amount} is due for your {service} account.",
            "Thank you for your payment of Rs. {amount} towards your bill."
        ],
        'security': [
            "Your verification code is {pin}. Do not share it with anyone.",
            "A new login to your account was detected from a new device.",
            "Your password has been successfully reset. If this wasn't you, contact support."
        ],
        'appointments': [
            "Your appointment with {person} is confirmed for {date}.",
            "Reminder: You have a booking for {service} on {date}.",
        ]
    }
    
    # Vocabulary to fill in the templates
    vocab = {
        'item': ['package', 'order', 'parcel', 'shipment'],
        'number': lambda: str(random.randint(1000000, 9999999)),
        'status': ['shipped', 'dispatched', 'out for delivery', 'delivered'],
        'amount': lambda: str(random.randint(500, 5000)),
        'service': ['internet', 'electricity', 'mobile plan', 'subscription'],
        'pin': lambda: str(random.randint(1000, 9999)),
        'person': ['Dr. Sharma', 'the consultant', 'the service center'],
        'date': ['tomorrow at 10 AM', 'Friday, 3 PM', '15th of this month']
    }

    augmented_texts = []
    num_to_generate = promo_count - service_count

    for _ in range(num_to_generate):
        category = random.choice(list(templates.keys()))
        template = random.choice(templates[category])
        
        filled_sms = template
        # Fill placeholders with random words from the vocab
        for placeholder in re.findall(r'{(.*?)}', template):
            value_options = vocab.get(placeholder)
            if callable(value_options):
                # If it's a function (like for numbers), call it
                replacement = value_options()
            else:
                # Otherwise, pick a random word from the list
                replacement = random.choice(value_options)
            filled_sms = filled_sms.replace(f"{{{placeholder}}}", replacement, 1)
        
        augmented_texts.append({'sms': filled_sms, 'category': 'service'})

    if augmented_texts:
        augmented_df = pd.DataFrame(augmented_texts)
        df = pd.concat([df, augmented_df], ignore_index=True)

    print(f"--- Augmentation complete. ---")
    print(f"New message counts:\n{df['category'].value_counts()}")
    return df

def perform_eda(df):
    """Performs and displays Exploratory Data Analysis."""
    print("\n--- Performing Exploratory Data Analysis (EDA) on augmented data ---")
    
    plt.figure(figsize=(10, 6))
    sns.countplot(x='category', data=df, palette={'promotional': '#e74c3c', 'service': '#3498db'}, order=['promotional', 'service'])
    plt.title('Distribution of SMS Categories After Augmentation', fontsize=16)
    plt.show()

def preprocess_text(df):
    """Cleans and preprocesses the SMS text for modeling."""
    print("\n--- Preprocessing text with improved logic... ---")
    corpus = []
    english_stopwords = set(stopwords.words('english'))

    for i in range(len(df)):
        review = re.sub(r'[^a-zA-Z\u0900-\u097F]+', ' ', df['sms'].iloc[i])
        review = review.lower().split()
        review = [word for word in review if not word in english_stopwords]
        review = ' '.join(review)
        corpus.append(review)
    
    print("Text preprocessing complete.")
    return corpus

def train_and_evaluate_model(X, y):
    """Splits data, trains the model with cross-validation, and evaluates it."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    print(f"\n--- Data splitting complete. ---")
    
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ('clf', LinearSVC(class_weight='balanced', max_iter=2000, dual=True)),
    ])
    
    parameters = {
        'clf__C': [0.1, 1, 10],
    }

    grid_search = GridSearchCV(pipeline, parameters, cv=5, n_jobs=-1, verbose=1)

    print("\n--- Starting model training with template-based augmentation... ---")
    grid_search.fit(X_train, y_train)

    print("\n--- Training complete. ---")
    print(f"Best C parameter found: {grid_search.best_params_['clf__C']}")

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    print(f"\n--- Final Model Evaluation on Test Data ---")
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    print("\n--- Classification Report ---")
    print(class_report)

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Greens',
                xticklabels=['Promotional', 'Service'], yticklabels=['Promotional', 'Service'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix for Best Model')
    plt.show()

    return best_model

def interactive_prediction_loop(model):
    """Starts an interactive loop to test SMS messages."""
    print("\n--- Starting Interactive Prediction Mode ---")
    print("Enter an SMS message to classify, or type 'exit' or 'quit' to end.")

    def predict_sms(sms_text):
        # The model pipeline handles the entire transformation
        prediction = model.predict([sms_text])
        return prediction[0]

    while True:
        # Get input from the user
        user_input = input("\nEnter SMS message: ")
        
        # Check if the user wants to exit
        if user_input.lower() in ['exit', 'quit']:
            print("Exiting interactive mode.")
            break
        
        # Ensure the input is not empty
        if not user_input.strip():
            print("Please enter a message.")
            continue

        # Get the prediction and print it
        prediction = predict_sms(user_input)
        print(f"-> Prediction: '{prediction.upper()}'")


def main():
    """Main function to run the entire pipeline."""
    download_nltk_data()
    
    data_file = 'Word_Classification - Sheet1.csv'
    
    df = load_and_clean_data(data_file)
    if df is not None:
        df_augmented = augment_service_messages(df)
        
        perform_eda(df_augmented)
        corpus = preprocess_text(df_augmented)
        model = train_and_evaluate_model(corpus, df_augmented['category'])
        
        # Start the interactive loop instead of the fixed test
        interactive_prediction_loop(model)

if __name__ == '__main__':
    main()
