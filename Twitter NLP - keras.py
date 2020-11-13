# Libraries
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
from sklearn import model_selection
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

# Data
test = pd.read_csv(filepath_or_buffer = "test.csv")
train = pd.read_csv(filepath_or_buffer = "train.csv")

X = train.iloc[:,0:4]
Y = train['target']

# Train and validation split
train_set_x, val_set_x, train_set_y, val_set_y = model_selection.train_test_split(X, Y)
train_set_x = np.array(train_set_x)

# Model - data preprocessing


vectorizer = TextVectorization(output_mode= 'int')

vectorizer.adapt(train_set_x[:, 3])
print(vectorizer(train_set_x[:,3]))