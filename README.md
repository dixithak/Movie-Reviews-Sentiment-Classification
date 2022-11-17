# Movie-Reviews-Sentiment-Classification
This dataset was created for the Kaggle competition, which can be found here: 
https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews , and it incorporates data 
from Socher et al's sentiment analysis, which can be found at
http://nlp.stanford.edu/sentiment/ .
The data came from Pang and Lee's original movie review corpus, which was based on 
Rotten Tomatoes reviews. Socher's team used crowd-sourcing to manually mark all of the 
sub-sentences with a sentiment label that ranged from "negative," "little negative," "neutral," 
"little positive," and "positive."
The data is split into training and testing The phrases and their related sentiment labels can 
be found in train.tsv. The file test.tsv just contains phrases without labels. Each sentence 
must be given a sentiment label.
The following are the sentiment labels:
0 - negative
1 – little negative
2 - neutral
3 – little positive
4 – positive
APPROACH 
Step 1: Reading Data from CSV file
Step 2: Preprocessing and filtering Data
Step 3: Generating Feature sets(included new features and combined features)
Step 4 : Saving Featuresets to CSV files
Step 5: Running cross validation on all Feature sets
Step 6: Running Naïve Bayes Classifier on Feature sets
Step 7: Running Random Forest Classifier on Feature sets(advanced experiment
