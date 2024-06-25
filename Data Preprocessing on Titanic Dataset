import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv('train.csv')
cols_to_drop = ['Name', 'Ticket', 'Cabin']
df = df.drop(cols_to_drop, axis=1)
df['Age'] = df['Age'].interpolate()
df = df.dropna()

dummies = []
cols = ['Pclass', 'Sex', 'Embarked']
for col in cols:
    dummies.append(pd.get_dummies(df[col], prefix=col))
titanic_dummies = pd.concat(dummies, axis=1)
df = pd.concat([df, titanic_dummies], axis=1)
df = df.drop(cols, axis=1)

y = df['Survived'].values
X = df.drop('Survived', axis=1).values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")
