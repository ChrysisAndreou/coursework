# Text Pre-processing and Analysis for Twitter Sentiment

## Project Overview
This project provides a comprehensive pipeline for text pre-processing, analysis, and sentiment classification of Twitter data. The primary objective is to demonstrate the importance and impact of various Natural Language Processing (NLP) pre-processing steps on vocabulary size, token characteristics, and model performance.

The script uses the `tweet_eval` sentiment dataset, focusing on positive and negative tweets. It applies a series of normalization techniques, analyzes the resulting token distributions, and trains a Multinomial Naive Bayes classifier to predict sentiment.

## Features
- **Comprehensive Text Pre-processing Pipeline:** Implements a series of steps tailored for noisy Twitter data, including:
    - Contraction Expansion
    - Twitter-specific Tokenization (`twokenize`)
    - Lowercasing & Punctuation Removal (while preserving emojis)
    - URL and Email Replacement
    - Stopword Removal
    - Lemmatization
- **In-depth Token Analysis:**
    - Tracks vocabulary size reduction at each pre-processing stage.
    - Compares token frequencies between positive and negative sentiment tweets.
    - Calculates and visualizes "token uniqueness" scores to identify sentiment-specific vocabulary.
    - Analyzes the prevalence of non-dictionary words (e.g., slang, hashtags) that carry strong sentiment signals.
- **Sentiment Classification:**
    - Trains and evaluates a Multinagomial Naive Bayes classifier as a baseline model.
    - Compares the performance of models using raw token counts (CountVectorizer) versus TF-IDF features.
    - Experiments with removing sentiment-neutral ("delta") tokens to assess their impact on classification.
- **Visualization:** Generates plots to visualize vocabulary reduction and token frequency distributions, providing clear insights into the data.

## Dataset
The project uses the **`tweet_eval` (sentiment)** dataset from the `datasets` library. It filters the dataset to include only tweets with **positive** and **negative** labels, excluding the neutral class as per the assignment requirements.

## Methodology
The project follows a structured NLP pipeline:
1.  **Data Loading and Filtering:** The `tweet_eval` dataset is loaded, and neutral tweets are removed.
2.  **Pre-processing:** Each tweet undergoes the following sequential transformations:
    1.  Contraction Expansion (e.g., "can't" -> "cannot")
    2.  HTML Unescaping & Tokenization
    3.  Lowercasing
    4.  URL & Email Replacement with placeholders (`URL`, `EMAIL`)
    5.  Punctuation Removal
    6.  Stopword Removal
    7.  Lemmatization (chosen over stemming to preserve word meaning)
3.  **Statistical Analysis:** After each step, statistics on token count and vocabulary size are recorded to measure the impact of the transformation.
4.  **Token Characterization:** The final processed tokens are analyzed to find:
    - Top N most frequent tokens overall, and for each sentiment.
    - Common vs. unique tokens between positive and negative vocabularies.
    - Tokens with high "uniqueness" scores, indicating a strong correlation with a specific sentiment.
5.  **Classification:**
    - The cleaned tokens are reassembled into processed tweet strings.
    - The data is split into training (80%) and testing (20%) sets.
    - N-gram features (uni-grams, bi-grams, and tri-grams) are extracted using `CountVectorizer`.
    - A Multinomial Naive Bayes model is trained and its performance is evaluated using a classification report.

## Key Findings
- **Vocabulary Reduction:** The pre-processing pipeline successfully reduced the overall vocabulary size by **over 25%** (from 44,232 to 32,915 unique tokens), making the data more efficient for modeling.
- **Baseline Model Performance:** The Multinomial Naive Bayes classifier trained on n-gram count features achieved a strong baseline **accuracy of 82%**.
- **TF-IDF vs. Count Features:** Using TF-IDF features **reduced performance** slightly (81% accuracy). This is likely because TF-IDF down-weights the importance of common sentiment-bearing words (e.g., "good", "love") which, despite being frequent, are crucial for sentiment prediction in short texts like tweets.
- **Impact of Delta Tokens:** Removing tokens that appeared equally in positive and negative contexts ("delta tokens") slightly **decreased performance to 79%**, suggesting that even seemingly neutral words can provide valuable context for the classifier.
- **Token Analysis:** Token frequency analysis revealed distinct high-frequency words for each sentiment (e.g., "good", "day" for positive vs. "like", "get" for negative after processing).

## Technologies & Libraries Used
- Python 3
- Jupyter Notebook
- **NLP & Data Handling:**
    - `pandas`
    - `numpy`
    - `nltk`
    - `twokenize`
- **Machine Learning:**
    - `scikit-learn`
- **Data Loading:**
    - `datasets` (from Hugging Face)
- **Visualization:**
    - `matplotlib`

## How to Run
To run this project, you can either execute the Python script or run the cells in the Jupyter Notebook.

### 1. Prerequisites
Ensure you have Python 3 installed. You will need to install the required libraries.

### 2. Installation
1.  Clone the repository:
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```
2.  Create a `requirements.txt` file with the following content:
    ```
    numpy
    pandas
    matplotlib
    scikit-learn
    nltk
    datasets
    twokenize
    ```
3.  Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### 3. Execution
- **To run the Python script:**
  ```bash
  python code1.py
  ```
  This will execute the entire pipeline, print statistics and classification reports to the console, display plots, and save the pre-processing stats to `preprocessing_stats.csv`.

- **To run the Jupyter Notebook:**
  Launch Jupyter Notebook and open the `jupyter.ipynb` file. Run the cells sequentially to see the output of each step, including the generated plots.
  ```bash
  jupyter notebook
  ```

## Project Files
- `code1.py`: The main Python script containing the full pre-processing, analysis, and classification pipeline.
- `jupyter.ipynb`: The Jupyter Notebook version of the script, showing the code and its corresponding outputs.
- `preprocessing_stats.csv`: The output CSV file containing the statistics (token counts, vocabulary sizes) for each pre-processing step.
- `assignment_description.pdf`: The original assignment description providing context and requirements for the project.
- `README.md`: This file.
