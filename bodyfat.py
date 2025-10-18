import torch
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
df = pd.read_csv('final_data.csv')
print(df.isnull().sum())

encoder = LabelEncoder()

print(df.head())
print(df.info())
print(df.describe())
