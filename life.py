import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB

#Read in Data
life_train = pd.read_csv("train.csv")
life_train['label'] = 'train'

life_test = pd.read_csv("test.csv")
life_test['label'] = 'test' 

#Combine the DFs for Cleaning
concat_df = pd.concat([life_train , life_test])

#Fix NaNs
concat_df.isnull().sum()

concat_df['Employment_Info_1'].value_counts()
concat_df['Employment_Info_1'].fillna(concat_df['Employment_Info_1'].mode()[0], inplace=True)

concat_df['Employment_Info_4'].value_counts()
concat_df['Employment_Info_4'].fillna(concat_df['Employment_Info_4'].mode()[0], inplace=True)

concat_df['Employment_Info_6'].value_counts()
concat_df['Employment_Info_6'].fillna(concat_df['Employment_Info_6'].mode()[0], inplace=True)


concat_df['Family_Hist_2'].value_counts()
concat_df['Family_Hist_2'].fillna(concat_df['Family_Hist_2'].mean(), inplace=True)

concat_df['Family_Hist_3'].value_counts()
concat_df['Family_Hist_3'].fillna(concat_df['Family_Hist_3'].mean(), inplace=True)

concat_df['Family_Hist_4'].value_counts()
concat_df['Family_Hist_4'].fillna(concat_df['Family_Hist_4'].mean(), inplace=True)

concat_df['Family_Hist_5'].value_counts()
concat_df['Family_Hist_5'].fillna(concat_df['Family_Hist_5'].mean(), inplace=True)


concat_df['Insurance_History_5'].value_counts()
concat_df['Insurance_History_5'].fillna(concat_df['Insurance_History_5'].mean(), inplace=True)


concat_df['Medical_History_1'].value_counts()
concat_df['Medical_History_1'].fillna(concat_df['Medical_History_1'].mode()[0], inplace=True)

concat_df['Medical_History_10'].value_counts()
concat_df['Medical_History_10'].fillna(concat_df['Medical_History_10'].mode()[0], inplace=True)

concat_df['Medical_History_15'].value_counts()
concat_df['Medical_History_15'].fillna(concat_df['Medical_History_15'].mode()[0], inplace=True)

concat_df['Medical_History_24'].value_counts()
concat_df['Medical_History_24'].fillna(concat_df['Medical_History_24'].mode()[0], inplace=True)

concat_df['Medical_History_32'].value_counts()
concat_df['Medical_History_32'].fillna(concat_df['Medical_History_32'].mode()[0], inplace=True)

#Get Dummies
features_df = pd.get_dummies(concat_df, columns = ['Product_Info_2'])

#Split'em up
life_train = features_df[features_df['label'] == 'train']
life_test = features_df[features_df['label'] == 'test']

#Drop Labels
life_train = life_train.drop('label', axis=1)
life_test = life_test.drop('label', axis=1)

#Make Response Last Column
life_train = life_train[[col for col in life_train if col not in ['Response']]
                            + ['Response']]

#Build Model

X = life_train.iloc[:, 0:-1]
y = life_train.iloc[:, -1]


nb = MultinomialNB()
nb.fit(X, y)

rf = RandomForestClassifier(n_estimators = 20)
rf.fit(X, y)

life_test = life_test.drop('Response', axis=1)

nb_submit = nb.predict(life_test)
rf_submit = rf.predict(life_test)

nb_submit = nb_submit.astype(int)
rf_submit = rf_submit.astype(int)

nb_submit = pd.Series(nb_submit, name = 'Response')
rf_submit = pd.Series(rf_submit, name = 'Response')

nb_submission = pd.concat([life_test.Id, nb_submit], axis=1)
rf_submission = pd.concat([life_test.Id, rf_submit], axis=1)

nb_submission = nb_submission.set_index('Id')
rf_submission = rf_submission.set_index('Id')

nb_submission.to_csv('nb_test.csv')
rf_submission.to_csv('rf_test.csv')
