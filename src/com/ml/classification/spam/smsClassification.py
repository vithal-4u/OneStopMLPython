import numpy as np
import pandas as pd
import string
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# Reading CSV file
message_data = pd.read_csv("D:/Study_Document/GIT/OneStopMLPython/resources/smsDataFormated.txt",names=['Label','Body'],skiprows=1)
message_data.head()

message_data = message_data.rename(columns = {'Index':'Label','Body':'message'})
message_data.groupby('Label').describe()

message_data

message_label = message_data['Label'].copy()
message_body = message_data['message']
print("Label ---", message_label)
print("Body ---", message_body)


def text_preprocess(text):
    text = str(text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = [word for word in text.split() if word.lower() not in stopwords.words('english')]
    return " ".join(text)

message_body = message_body.apply(text_preprocess)
message_body

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer("english")
message_mat = vectorizer.fit_transform(message_body)
message_mat

#Splitting the data into test and train
from sklearn.model_selection import train_test_split
message_train, message_test, spam_nospam_train, spam_nospam_test = train_test_split(message_mat,
                                                        message_label,
                                                        test_size=0.2, random_state=0)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

Spam_model = LogisticRegression(solver='liblinear', penalty='l1')
Spam_model.fit(message_train, spam_nospam_train)

pred = Spam_model.predict(message_test)
print("accuracy_score -- ",accuracy_score(spam_nospam_test,pred))
few_test = vectorizer.transform( \
    ["+918870782800 is now available to take calls. To CALL BACK dial +918870782800.For Top HelloTunes, Airtel Call Manager or Live aarti,call 567892 (tollfree)",
    "Akshara and 6 others have added their photos & also match your partner preference. View photos on EzhavaMatrimony by visiting www.cmatri.in/?kNw8dB~n~grNC",
    "Rs.1000.00 was withdrawn using your HDFC Bank Card ending 0449 on 2017-04-01:21:31:34 at +CTS THORAIPAKKAM OATM. Avl bal: Rs.18834.44",
    "Trying to loose weight? Meet our Bariartic Surgeon ¡ Apollo Spectra Hospital MRC Nagar.Spectra Obesity Check ¡ Rs 840.T & C apply. Call 66862000.",
    "Dear Associate, Cognizant, offices in Chennai and Coimbatore will remain closed today, Dec 6, 2016. Note: Associates working on CIS, BPS and 24/7 support projects, please contact your manager for instructions."] )

pred_few = Spam_model.predict(few_test)
print(pred_few)

print("Execution Completed")