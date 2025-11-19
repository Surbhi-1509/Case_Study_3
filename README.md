# Case_Study_3
A Gen AI & LLM case study implementing Transformer-based models (BERT, DistilBERT) to automatically classify news articles into categories using attention mechanisms and deep learning.

## Overview
This project aims to automate news article categorization using Transformer architectures and Attention Mechanisms.
Manual sorting of thousands of daily articles is time-consuming; thus, this project leverages Natural Language Processing (NLP) for intelligent, fast, and accurate classification.

## Objective
To build a model that reads a news article’s text and predicts its topic using state-of-the-art Transformer models like BERT and DistilBERT.

## Problem Statement
In the digital media era, the vast flow of news across diverse categories requires automation for efficient curation.
Manual categorization causes delays and inconsistencies. This project develops an automated Transformer-based news classification system to handle large-scale data with high accuracy and real-time performance.

## Key Challenges
- Information Overload: Continuous inflow of massive text data.
- Timeliness: Delayed classification reduces content relevance.
- 
This solution applies self-attention and contextual embeddings to overcome these issues.

## Dataset
- news_articles.csv → Contains raw article texts.
- news_article_labels.xlsx → Contains corresponding category labels.

## Steps Involved
- Data Preprocessing – Clean text (remove punctuation, stopwords), tokenize, and pad sequences.
- Model Development – Implement and fine-tune BERT/DistilBERT with an attention-based classification head.
- Evaluation – Assess with Accuracy, Precision, Recall, F1-score, and visualize confusion matrix.

## Technologies Used
- Python
- Jupyter Notebook
- TensorFlow / PyTorch
- HuggingFace Transformers
- Scikit-learn
- Pandas, NumPy, Matplotlib, Seaborn
