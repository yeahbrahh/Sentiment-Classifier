# Sentiment-Classifier

A binary sentiment classifier.

## Objective:
The goal of this project is to build and evaluate a **sentiment classifier** using the _Amazon Health & Personal Care_ reviews dataset. By sorting reviews into positive or negative outcomes, the model will help surface customer satisfaction trends and highlight key product strengths and weaknesses at scale.

### Labels:

- **1–2 stars** = Negative (0)

- **4–5 stars** = Positive (1)

- **3 stars** = Dropped entirely (noise)

- **Sentiment**: (0 or 1)

### Features:
- **title** : review titles; 100 character limit 

- **text** : standard text features extracted from the review body.
- **verified_purchase** : true or false

### Addressing Class Imbalance:  
Amazon reviews historically skew positive. We will analyze the exact class distribution early and address any imbalance using class-weighted model parameters. Built-in class weight adjustments will be utilized to penalize misclassifications of the minority class.

## Methodology:

- ### TF-IDF + Linear SVM:  

We use TF-IDF to convert the review text into numbers. Because text data creates massive, high-dimensional tables, Linear SVMs handle this efficiently, train quickly, and provide a strong baseline model.

- ### Imbalance Handling:

We'll build imbalance handling (like adjusting class weights) right into the pipeline.

- ### Feature Inspection: 
We'll look at the model's coefficients to see what specific words are driving the positive/negative predictions and check the results.

## Metrics:
- Accuracy, Precision, Recall, and F1-Score.
- Confusion matrices to visualize false positives and false negatives.
- We will review the Linear SVM's coefficient weights to pinpoint key predictive words and phrases, confirming that the model captures true semantic patterns instead of artifacts.

## Tech Stack:
- **Python 3.14.7**
- **Pandas**: Used for cleaning up the data, filtering out neutral reviews, and organizing everything into DataFrames so it's ready to use.
- **Scikit-Learn**: Handles the core machine learning work.
- **Matplotlib**: Used for drawing charts and graphs to visualize things like the balance of positive vs. negative reviews and the model's results.
