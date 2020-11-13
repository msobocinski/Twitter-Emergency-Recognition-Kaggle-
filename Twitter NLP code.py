# Libraries
import re
import pandas as pd
import numpy as np
import spacy
from spacy.matcher import Matcher
from sklearn import model_selection
from sklearn.ensemble import GradientBoostingClassifier

# Data
test = pd.read_csv(filepath_or_buffer = "test.csv")
train = pd.read_csv(filepath_or_buffer = "train.csv")

# Clean keyword column in train
for index, word in enumerate(train.loc[:,'keyword']):
    if pd.isna(word):
        pass
    else:
        train.loc[index, 'keyword'] = re.sub(pattern = '%20', repl= ' ', string = train.loc[index, 'keyword'])

# Clean keyword in train
for index, word in enumerate(test.loc[:,'keyword']):
    if pd.isna(word):
        pass
    else:
        test.loc[index, 'keyword'] = re.sub(pattern = '%20', repl= ' ', string = test.loc[index, 'keyword'])

# Data pre-processing
# Load spacy model and matcher
nlp = spacy.load("en_core_web_sm")
matcher = Matcher(nlp.vocab)

# Add text, has_link and has_exclamation columns to the train set
train['text_as_spacy'] = train['text'].apply(nlp)
train['has_link'] = 0
train['has_exclamation'] = 0

#### Term counting #####
# Create list for term counting
terms = list()

# Loop to find links and exclamation marks in training set
for index, text in enumerate(train['text_as_spacy'][0:(len(train) - 1)]):
    for token in text:
        if token.like_url:
            train['has_link'][index] = 1
        if token.text == '!':
            train['has_exclamation'][index] = 1

# New approach to counting tokens - create list with all terms
for text in train['text_as_spacy']:
    for token in text:
        terms.append(str(token.text))

# Get terms to lowercase
for string in range(0, len(terms)):
    terms[string] = terms[string].lower()

# Get unique terms
unique_terms = set(terms)
unique_terms = list(unique_terms)

# Dictionary for term count
term_count = {}

# Loop to create dict with term : term count pairs
for term in unique_terms:
    term_count[term] = terms.count(str(term))

# Convert the dictionary to Series for sorting
term_count_series = pd.Series(data=term_count)

# Sort Series
term_count_series = term_count_series.sort_values(ascending=False)

# Convert to Data Frame
term_count_frame = term_count_series.to_frame(name='Counts')

# Add terms as columns
term_count_frame['Term'] = term_count_frame.index

# Go Spacy on the frame
term_count_frame['text_as_spacy'] = term_count_frame['Term'].apply(nlp)

# Select 300 most popular terms
term_count_clean = term_count_frame

# Reindex term_count_clean and remove old index column
term_count_clean.reset_index(inplace=True)
term_count_clean.drop(columns = ['index'], inplace=True)

# Remove stopwords, interpunction and other noise
for i in range(0, len(term_count_clean)):
    t = term_count_clean['text_as_spacy'][i][0]
    if t.is_stop or t.is_digit or t.is_punct or t.like_url:
            term_count_clean.drop(index = [i], inplace = True)

# Final touch in cleaning
term_count_clean.drop(index = [23, 25, 49, 87, 92, 234, 279], inplace=True)
term_count_clean.reset_index(inplace=True)

# Select words with 50+ occurrences
term_count_clean = term_count_clean.loc[term_count_clean['Counts'] >= 50]

# Create dataframe for terms with as many rows as in training dataset and number of columns equal to no of terms
term_matrix = np.zeros([len(train), term_count_clean.shape[0]])
term_matrix = pd.DataFrame(term_matrix, columns=term_count_clean['Term'])

# Loop to find which tweet contains term
for index, text in enumerate(train['text']):
    for index2, term in enumerate(list(term_count_clean['Term'])):
        if term in text:
            term_matrix.iloc[index, index2] = 1

# Append term matrix to train set
train2 = pd.concat(objs = [train, term_matrix], axis = 1)

# Add sentiment

#### Classification with Scikit (Gradient Boosting) #####
X = train2.drop(columns = ['id', 'target', 'text', 'text_as_spacy', 'keyword', 'location'], inplace = False)
Y = train2['target']

train_set_x, val_set_x, train_set_y, val_set_y = model_selection.train_test_split(X, Y)

gb_classifier = GradientBoostingClassifier()
gb_classifier.fit(train_set_x, train_set_y)

gb_classifier.score(val_set_x, val_set_y)

# IDEAS #
# 1. Sentiment analysis - NLTK
# 2. Other classification methods
# 3. Adding location and keyword after transformation to integers


