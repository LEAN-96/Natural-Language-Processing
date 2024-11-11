# NLP Text Classification

## 1. Project Overview

This project involves the implementation of **Natural Language Processing (NLP)** techniques to solve a **text classification** problem. The notebook demonstrates how to preprocess text data, build machine learning models, and evaluate their performance. Specifically, it covers the use of traditional machine learning algorithms for classifying text into predefined categories.


### Dataset
We use a subset of Yelp reviews that contains:
- **Columns**: `business_id`, `date`, `review_id`, `stars`, `text`, `type`, `user_id`, `cool`, `useful`, `funny`.
- **Target**: The `stars` column is used to define sentiment. Reviews with 4 or 5 stars are classified as **positive**, and those with 1 or 2 stars as **negative**.

Before using the data for machine learning, the following preprocessing steps were carried out:

1. **Text Cleaning**: Lowercasing, removing punctuations, stopwords, and special characters.
2. **Tokenization**: Splitting the text into individual words (tokens).
3. **Feature Extraction**: Using **TF-IDF (Term Frequency-Inverse Document Frequency)** to convert text into numerical features.
4. **Label Encoding**: Assigning binary labels (1 for positive, 0 for negative) based on the star rating.

### Machine Learning Methods
This project uses two machine learning algorithms to classify Yelp reviews:

1. **Multinomial Naive Bayes (MultinomialNB)**:  
   MultinomialNB is a probabilistic algorithm well-suited for text classification problems. It assumes that the features (words) are distributed according to a multinomial distribution. This model is efficient and often performs well for tasks like spam detection and sentiment analysis. The algorithm calculates the probability of each word in a document belonging to a certain class (positive or negative) based on the frequency of words in the training data. It then predicts the class with the highest probability.

2. **TF-IDF Transformer with Logistic Regression**:  
   The **TF-IDF Transformer** converts a collection of raw documents into a matrix of TF-IDF (Term Frequency-Inverse Document Frequency) features. TF-IDF helps highlight important words in the corpus by scaling down the influence of commonly used words and emphasizing rarer, more meaningful words. After transforming the text data into TF-IDF features, these are fed into a **Logistic Regression** classifier. Logistic Regression models the probability of a review being positive or negative based on the features. It is a linear model that predicts binary outcomes.

### Notebook Overview
The notebook is structured as follows:

1. **Data Loading and Preprocessing**: 
   - The dataset is loaded from a file (e.g., CSV) and prepared by cleaning the text and converting it into numerical features using vectorization techniques like TF-IDF.
   
2. **Model Building**:
   - Two key models are defined: Logistic Regression and Naive Bayes. These models are chosen due to their effectiveness in handling high-dimensional text data.

3. **Model Training**:
   - The models are trained on the preprocessed dataset using default parameters. Cross-validation is employed to ensure generalization.

4. **Evaluation**:
   - The models are evaluated using accuracy, precision, recall, and F1-score metrics. A confusion matrix is also generated to provide insights into misclassifications.

5. **Visualization**:
   - Users can expect visualizations such as accuracy plots over epochs (if applicable), confusion matrices, and bar charts showing feature importance or class distribution.

## 2. Requirements

### Running Locally

To run the notebook locally, follow these steps:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/LEAN-96/Natural-Language-Processing.git
    cd natural-language-processing
    ```

2. **Set up a virtual environment**:
    Using `venv`:
    ```bash
    python3 -m venv env
    source env/bin/activate  # On Windows use `env\Scripts\activate`
    ```

    Or using `conda`:
    ```bash
    conda create --name ml-env python=3.8
    conda activate ml-env
    ```

3. **Install project dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Launch Jupyter Notebook**:
    ```bash
    jupyter notebook
    ```
    Open the notebook (`3-Nlp.ipynb`) in the Jupyter interface to run it interactively.

### Running Online via MyBinder

To run this notebook online without installing anything locally, use MyBinder:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/LEAN-96/Natural-Language-Processing.git/HEAD?labpath=notebooks)

Once MyBinder loads:
1. Navigate to your notebook (`3-Nlp.ipynb`) in the file browser on the left.
2. Click on the notebook to open it.
3. Run all cells using "Run All" (`Shift + Enter` for individual cells).

By using MyBinder, you can explore the notebook without installing any software packages locally.

## 3. Reproducing Results

To reproduce the results from this project:

1. Open the notebook (`3-Nlp.ipynb`) using Jupyter (locally or via MyBinder).
2. Execute all cells sequentially by selecting them and pressing `Shift + Enter`.
3. Ensure that all cells execute without errors.
4. Observe output results directly within the notebook interface.

### Interpreting Results:

This section provides an interpretation of the results obtained from two different machine learning models used for classifying Yelp reviews: **Multinomial Naive Bayes (MultinomialNB)** and **TF-IDF with a Logistic Regression**. Both models were evaluated using the **confusion matrix** and the **classification report**, which includes key metrics such as precision, recall, F1-score, and accuracy.

1. **Confusion Matrix**

The confusion matrix provides insights into the number of correct and incorrect predictions:
- **True Positives (TP)**: Correctly classified positive reviews.
- **True Negatives (TN)**: Correctly classified negative reviews.
- **False Positives (FP)**: Negative reviews incorrectly classified as positive.
- **False Negatives (FN)**: Positive reviews incorrectly classified as negative.

#### MultinomialNB Confusion Matrix:

[[159 69]

[ 22 976]]

- **159** true negatives (class 1, negative reviews).
- **976** true positives (class 5, positive reviews).
- **69** false positives (negative reviews predicted as positive).
- **22** false negatives (positive reviews predicted as negative).

#### TF-IDF Confusion Matrix:

[[ 0 228]

[ 0 998]]

- **228** true negatives (class 1, negative reviews).
- **998** true positives (class 5, positive reviews).
- **0** false positives.
- **0** false negatives.

2. **Classification Report Metrics**

The classification report provides a detailed breakdown of the model’s performance for each class (1 and 5), using the following metrics:
- **Precision**: The proportion of true positive predictions out of all positive predictions made by the model.
- **Recall**: The proportion of actual positives correctly identified by the model.
- **F1-score**: The harmonic mean of precision and recall, providing a balanced measure of performance.
- **Support**: The number of true instances of each class in the dataset.


#### Multinomial Naive Bayes (MultinomialNB):
- **Accuracy**: 93% overall, indicating that the model performs well in classifying both positive and negative reviews.
- **Recall for Negative Reviews (Class 1)**: 70%, which means that it missed some negative reviews.
- **Precision for Positive Reviews (Class 5)**: 93%, meaning most positive predictions were correct.
- **Balanced Performance**: The model provides a balanced performance across both classes, with a slight drop in recall for negative reviews.

#### TF-IDF Model:
- **Accuracy**: 81%, lower than the MultinomialNB model.
- **Perfect Recall (100%)**: The TF-IDF model achieved 100% recall for both positive and negative reviews, meaning it correctly identified all true instances. However, this comes at the cost of precision.
- **Precision**: 81%, meaning the model made more incorrect positive predictions.
- **Overfitting or Bias**: The TF-IDF model may be overfitting to the training data, as it perfectly recalled all instances but sacrificed precision.

## Conclusion

In summary, **Multinomial Naive Bayes** provides better overall performance with a higher accuracy score and more balanced precision-recall metrics, making it the preferable choice for this Yelp review classification task.