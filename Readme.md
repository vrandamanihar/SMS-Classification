SMS Spam Classifier: Promotional vs. Service
This project contains a machine learning model designed to classify SMS messages into two categories: Promotional (spam, advertisements, offers) or Service (updates, OTPs, delivery notifications).

The model is built to handle real-world challenges, such as datasets with mixed languages (English and Hindi) and a significant imbalance between the number of promotional and service messages.

The Problem
In today's world, our SMS inboxes are flooded with messages. It's crucial to distinguish between:

Promotional SMS: Often considered spam, these are marketing messages, offers, and discounts that can be safely ignored or deleted.

Service SMS: These are important, time-sensitive messages like One-Time Passwords (OTPs), transaction alerts, delivery status updates, and appointment reminders.

Manually sorting these is tedious. This project automates the process using Natural Language Processing (NLP) and machine learning.

Key Features
Multi-language Processing: The model is designed to understand and process text containing both English and Hindi characters.

Advanced Data Augmentation: To combat the common issue of having few service message examples, the script uses a template-based system to generate new, realistic service messages (for logistics, billing, security, etc.). This ensures the model learns a wide variety of service-related patterns.

Contextual Understanding: Instead of just looking at single words, the model analyzes pairs of words (n-grams). This helps it understand context, recognizing that "tracking number" is a service phrase, while "limited time offer" is promotional.

Focused Learning on Service Messages: The model uses a class_weight='balanced' setting, which forces it to "pay more attention" to the less common service messages. This makes it less likely to misclassify an important update as spam.

Interactive Prediction Mode: Once trained, the script launches a loop that allows you to enter any SMS message and get an instant classification.

How It Works: The Machine Learning Pipeline
The sms_classifier.py script follows a complete machine learning pipeline:

Load and Clean Data: The script starts by loading the Word_Classification - Sheet1.csv file, cleans it by removing empty rows, and standardizes the category labels.

Data Augmentation: It detects if there are fewer service messages than promotional ones. If so, it automatically generates new, diverse service messages from a set of predefined templates to create a balanced and robust training dataset.

Exploratory Data Analysis (EDA): It generates plots to visualize the distribution of message categories, giving insights into the data's composition after augmentation.

Text Preprocessing:

Removes punctuation and numbers while preserving all English and Hindi letters.

Converts all text to lowercase.

Removes common, low-value English "stopwords" (e.g., 'the', 'a', 'is').

Feature Extraction (TF-IDF): The cleaned text is converted into numerical vectors using TfidfVectorizer. This process gives higher scores to words and phrases that are important for distinguishing between the two categories.

Model Training:

A Linear Support Vector Machine (LinearSVC), a powerful and effective classifier for text, is used.

GridSearchCV is employed to automatically test different model settings and find the best-performing configuration through 5-fold cross-validation, which prevents overfitting.

Evaluation: The best model is evaluated on a held-out test set, and its performance is displayed with an accuracy score and a detailed classification report.

Interactive Prediction: After training, the script enters a loop where you can input any SMS and see the model's predicted category in real-time.

How to Use This Project
Prerequisites
You need Python 3 installed, along with the following libraries:

pandas

scikit-learn

nltk

matplotlib

seaborn

wordcloud

Setup
Clone the repository:

git clone <your-repository-url>
cd <your-repository-directory>

Install the required libraries:
You can install them all using pip:

pip install pandas scikit-learn nltk matplotlib seaborn wordcloud

Place your dataset:
Make sure your dataset file, named Word_Classification - Sheet1.csv, is in the same directory as the sms_classifier.py script.

Running the Script
To run the entire pipeline (including training, evaluation, and the interactive mode), simply execute the following command in your terminal:

python sms_classifier.py

The script will first perform all the training and evaluation steps, displaying plots and reports. Finally, it will prompt you to enter SMS messages for classification.

--- Starting Interactive Prediction Mode ---
Enter an SMS message to classify, or type 'exit' or 'quit' to end.

Enter SMS message: Your package with tracking number 991243 is out for delivery
-> Prediction: 'SERVICE'

Enter SMS message: exit
Exiting interactive mode.
