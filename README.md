# NLP Text Classification

## 1. Project Overview

This project involves the implementation of **Natural Language Processing (NLP)** techniques to solve a **text classification** problem. The notebook demonstrates how to preprocess text data, build machine learning models, and evaluate their performance. Specifically, it covers the use of traditional machine learning algorithms for classifying text into predefined categories.

The main objective is to guide users through the process of transforming raw text data into a format suitable for machine learning models, training models on this data, and evaluating their performance using appropriate metrics.

### Dataset
The dataset used in this project consists of text samples labeled into different categories. It is sourced from [mention dataset source], containing approximately [dataset size] samples. Each sample includes a text field and a corresponding label indicating its category.

Before using the data for machine learning, the following preprocessing steps were carried out:
- **Text cleaning**: Removal of punctuation, stopwords, and other irrelevant tokens.
- **Tokenization**: Splitting text into individual words or tokens.
- **Vectorization**: Converting text data into numerical format using methods like TF-IDF or Bag-of-Words.

### Machine Learning Methods
The following machine learning methods are applied in the notebook:

- **Logistic Regression**: A linear model used for binary or multi-class classification tasks. It is applied here to predict the category of a given text sample based on its features.
- **Naive Bayes**: A probabilistic classifier based on Bayes' theorem with strong independence assumptions between features. It is particularly effective for text classification tasks.

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

- **Accuracy Metrics**: These metrics show how well each model performs on both training and testing datasets.
- **Confusion Matrix**: This matrix helps visualize correct vs incorrect predictions across different categories.
- **Feature Analysis or Graphs**: Visualizations such as bar charts may be included to show which words or features are most important for classification.