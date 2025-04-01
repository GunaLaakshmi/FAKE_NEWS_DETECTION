#Get Dataset from Kaggle

import kagglehub

# Download latest version
path = kagglehub.dataset_download("emineyetm/fake-news-detection-datasets")

print("Path to dataset files:", path)

#Fake News Detection using Data Science
This project uses machine learning to classify news articles as either Real or Fake. The dataset contains two types of articles: fake and real news.
The model is trained using the Passive-Aggressive Classifier and text features are extracted using TF-IDF Vectorization.

#Project Overview
The goal of this project is to classify news articles into two categories:
  .Real News (1)
  .Fake News (0)
The dataset consists of news articles from reliable sources (Reuters.com for real news) and unreliable sources (fake news outlets) from the years 2016-2017.

The project follows a typical data science workflow:

1.Data Collection
2.Data Preprocessing
3.Exploratory Data Analysis (EDA)
4.Feature Engineering
5.Model Training
6.Model Evaluation
7.Model Deployment
