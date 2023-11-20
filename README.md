# Natural-Language-Processing-Project
**Toxicity tweets**

This dataset has a collection of Tweets. It's labeled as Toxic - 1, Non-toxic - 0. The code written here uses various NLP libraries to remove toxicity from the tweets.

**Prerequisites**

- Ensure that the following Python libraries are installed
- Required libraries: sci-kit-learn, numpy, matplotlib, pandas, nltk
- Make sure to download the NLTK libraries
- import nltk
- nltk.download('stopwords')
- nltk.download('wordnet')
- nltk.stem import WordNetLemmatizer

**Dataset**

The code reads data from a CSV file and contains tweets and their toxicity labels.

**Cleaning of text**

The tweet text is processed through the following steps:
Removing the punctuation and digits from the text.
Removing the stopwords from the text.
Lemmatize words to convert them to their base form.

**Bag of Words Representation**

The cleaned tweet text is converted into a Bag of Words (BoW) representation using the CountVectorizer from sci-kit-learn.

**TF-IDF Representation**

The cleaned tweet text is also converted into a TF-IDF (Term Frequency-Inverse Document Frequency) representation using the TfidfVectorizer from sci-kit learn.

**Model Training and Testing**

The code built earlier evaluates the performance using different classification models like,
- Decision Tree Classifier
- Random Forest Classifier
- Naive Bayes Classifier (Gaussian Naive Bayes)
- K-Nearest Neighbors (KNN) Classifier
- Support Vector Machine (SVM) Classifier

**Coding approach**

1) Read the file using the file path link created.
2) Make sure the required Python libraries are installed and imported.
3) Preprocess the data as mentioned above.
4) Train and test the model using the different classification models mentioned above.
5) Print the following evaluation metrics for each model
 - Precision
 - Recall
 - F1 - Score
 - Confusion Matrix
 - ROC-AUC Score     


