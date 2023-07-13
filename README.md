# Intel-Unnati-Project-Fake-News-Detection-Using-Python-and-Machine-Learning
As a part of the Intel Unnati Industrial Training.This is our project, we have used several machine learning (ML) and deep learning (DL) models to detect fake news. Here is a summary of the models and their corresponding accuracies:
In our project, we have three main files:

a. `ModelsAccuracyTable.xlsx`:
   - This Excel file contains the accuracy values of various models used for fake news detection. It provides a summary of the model performances, including the ML and DL models we employed.

b. `FakeNewsClassifierUsingLSTM.ipynb`:
   - This Python script focuses on implementing the LSTM deep learning model for fake news detection. The LSTM model leverages its recurrent architecture to capture sequential dependencies in textual data, making it suitable for analyzing news articles.

c. `FakeNewsDetection using ML.ipynb`:
   - This Python script serves as the main file for our fake news detection using ML techniques. It encompasses the implementation of multiple ML models such as Logistic Regression, Multinomial Naive Bayes, Support Vector Classifier, Decision Tree Classifier, Random Forest Classifier, Ensemble Learning, Passive Aggressive Classifier, and AdaBoost Classifier. These models have been trained and evaluated for the task of detecting fake news.
     
1. LSTM (DL Model): Accuracy - 0.9528
   - Long Short-Term Memory (LSTM) is a type of recurrent neural network (RNN) commonly used for sequence classification tasks like text analysis.

2. Logistic Regression (ML Model): Accuracy - 0.9865
   - Logistic Regression is a linear classification model that applies the logistic function to predict binary outcomes.

3. Multinomial Naive Bayes (ML Model): Accuracy - 0.9385
   - Multinomial Naive Bayes is a probabilistic classifier based on the Bayes' theorem, suitable for text classification tasks.

4. Support Vector Classifier (ML Model): Accuracy - 0.9936
   - Support Vector Classifier (SVC) is a supervised learning model that separates classes using hyperplanes in high-dimensional space.

5. Decision Tree Classifier (ML Model): Accuracy - 0.9957
   - Decision Tree Classifier builds a tree-like model of decisions and their possible consequences, enabling classification based on feature values.

6. Random Forest Classifier (ML Model): Accuracy - 0.9902
   - Random Forest Classifier is an ensemble learning method that combines multiple decision trees to make predictions.

7. Ensemble Learning (ML Model): Accuracy - 0.9957
   - Ensemble Learning combines multiple ML models to improve predictive performance and make more accurate predictions.

8. Passive Aggressive Classifier (ML Model): Accuracy - 0.9943
   - Passive Aggressive Classifier is an online learning algorithm for binary classification that adapts to misclassified samples aggressively.

9. AdaBoost Classifier (ML Model): Accuracy - 0.9943
   - AdaBoost Classifier is an ensemble learning method that iteratively combines weak classifiers to create a strong classifier.

These models have been trained and evaluated on our dataset for fake news detection. Each model has achieved a certain level of accuracy, which indicates its performance in classifying news articles as real or fake.
