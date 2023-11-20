# Kannada MNIST
This project involves using a classification model called PCA for dimensionality reduction on the given dataset. It aims to analyze the performance of different classifiers with varying sizes of pca
components and thereby find the AUC-ROC scores.

**Libraries Needed**

Make sure to install and import the following Python libraries 
- Pytesseract,
- Panda,
- Sci-kit-learn,
- Numpy,
- Matplotlib

**Processing the data**

The code uses the Kannada MNIST dataset which consists of handwritten digits in the Kannada script. The dataset is divided into train data and test data.
The data is loaded using the numpy library to read data from the NPZ file given.

**Principal Component Analysis**

The next step would be to perform a principal component analysis in order to reduce the dimensionality of the data given. PCA is performed on different sizes like 10,15,20,25,30.
PCA is fit transformed in the training data and transformed in the test data given.

**Classification Models**

The following classification models are used in order to find out the best dimensionality reduction done using different PCA sizes
- Decision Tree Classifier
- Random Forest Classifier
- Naive Bayes Classifier (Gaussian Naive Bayes)
- K-Nearest Neighbors (KNN) Classifier
- Support Vector Machine (SVM) Classifier
  
**Coding Approach**
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
