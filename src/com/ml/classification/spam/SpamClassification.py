#
# This file reads csv file of spam and non-spam messages and segregates messages into
# spam and non spam. Split the message into words by removing stopwords.
#  Fit the words into "TfidfVectorizer" to convert into index based words.
#  we use "LogisticRegression" to train and test and predict the new value.
#
import numpy as np
import pandas as pd
import string
import nltk
#nltk.download('stopwords')

from nltk.corpus import stopwords

# Reading CSV file
message_data = pd.read_csv("D:/Study_Document/GIT/OneStopMLPython/resources/spam.csv",encoding = "latin")
message_data.head()

# Drop the columns for the dataset
message_data = message_data.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis = 1)
message_data = message_data.rename(columns = {'v1':'Spam/Not_Spam','v2':'message'})
message_data.groupby('Spam/Not_Spam').describe()

message_data_copy = message_data['message'].copy()
message_spam_nonspam = message_data['Spam/Not_Spam']

# print(message_data_copy)
# print(message_spam_nonspam)

# This Method reads each message removes special characters and splits each words
#     from line and removes stopwords.
def text_preprocess(text):
    #Below line removes special character froms string
    text = text.translate(str.maketrans('', '', string.punctuation))

    #Below line split each row into array of words like ['Go', 'jurong', 'point'..]
    text = [word for word in text.split() if word.lower() not in stopwords.words('english')]
    return " ".join(text)

message_data_copy = message_data_copy.apply(text_preprocess)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer("english")
message_mat = vectorizer.fit_transform(message_data_copy)
message_mat


from sklearn.model_selection import train_test_split
message_train, message_test, spam_nospam_train, spam_nospam_test = train_test_split(message_mat,
                                                        message_spam_nonspam,
                                                        test_size=0.3, random_state=0)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

Spam_model = LogisticRegression(solver='liblinear', penalty='l1')
Spam_model.fit(message_train, spam_nospam_train)
X_test = vectorizer.transform( ["Yeah hopefully, if tyler can't do it I could maybe ask around a bit",
                                "SIX chances to win CASH! From 100 to 20,000 pounds txt> CSH11 and send to 87575. Cost 150p/day, 6days, 16+ TsandCs apply Reply HL 4 info"] )

pred = Spam_model.predict(X_test)
print(pred)

print("Execution completed")