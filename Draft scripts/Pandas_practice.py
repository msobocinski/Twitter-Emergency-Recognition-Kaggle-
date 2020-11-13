import pandas as pd
import numpy as np

# 1. Create an array
pd.array(data = range(0,10))

# 2. Convert Series to list
s1 = pd.Series(data = range(0,10))
s1
s1.to_list()

# 3. Write a Pandas program to add, subtract, multiple and divide two Pandas Series.
s2 = pd.Series(data = np.repeat(10,10))

def calculator(s1, s2, action):
    if action == 'add':
        return (s1 + s2)
    if action == 'subtract':
        return (s1 - s2)
    if action == 'multiply':
        return (s1 * s2)
    if action == 'divide':
        return (s1 / s2)

calculator(s1, s2, 'subtract')

# 8. convert the first column of a DataFrame to a Series
df1 = pd.DataFrame(data = [(2,3,5), (8,8,2), (2,8,2)], columns = ['weight', 'height', 'speed'])
df1.columns
df1.index

s3 = df1.iloc[:,0]
s3

# 9. Sort Series
df2 = pd.DataFrame(s3)
df2.sort_values(by = 'weight')

# 10. Add stuff
s3 = s3.append(pd.Series((8,3,1)), ignore_index = True)

df1.iloc[:,2]

##### Diamonds excercise #####
diamonds = pd.read_csv('diamonds.csv')
print(diamonds.iloc[0:5, :])

diamonds
