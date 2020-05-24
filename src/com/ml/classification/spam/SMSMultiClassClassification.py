import sklearn
import pandas as pd
import nltk
from IPython.display import display
from nltk.corpus import stopwords
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

#nltk.download('stopwords')
smsData = pd.read_csv("D:/Study_Document/GIT/OneStopMLPython/resources/smsDataFormated.txt",names=['Label','Body'],skiprows=1)
smsData.head()

stopList = list(stopwords.words('english'))

cleaned = []
for i in range(len(smsData['Body'])):
    clean = smsData['Body'][i]
    clean = clean.lower().split()
    clean = [word for word in clean if word not in stopList]
    clean = ' '.join(str(w) for w in clean)
    print("SMS-----",clean)
    cleaned.append(clean)

intLabel = smsData['Label'].copy()
i=0
for l in intLabel:
    if l == 'Network':
        intLabel[i]=0
    elif l == 'Cognizant':
        intLabel[i]=1
    elif l == 'Promotional':
        intLabel[i]=2
    elif l == 'spam':
        intLabel[i]=3
    elif l == 'Trans':
        intLabel[i]=4
    else:
        intLabel[i]=5
    i=i+1

smsData.insert(loc=2, column="Cleaned", value=cleaned)
smsData.insert(loc=0, column="IntLabel", value=intLabel)
print(smsData.head())

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer("english")
X = vectorizer.fit_transform(smsData['Cleaned'])
Y = smsData['IntLabel'].values.astype('int')
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,stratify=Y,test_size=.4)

from sklearn.svm import LinearSVC
classifier = OneVsRestClassifier(LinearSVC(max_iter=3000))
classifier.fit(X_train, Y_train)


from sklearn.metrics import accuracy_score,f1_score,confusion_matrix,classification_report
predicted = classifier.predict(X_test)
print("Accuracy Score : ",accuracy_score(Y_test, predicted))
print("F1 Score : ",accuracy_score(Y_test, predicted))
print("Confusion Matrix : \n",confusion_matrix(Y_test, predicted))
print('\nClasification report:\n', classification_report(Y_test, predicted))

few_test = vectorizer.transform( \
    ["+918870782800 is now available to take calls. To CALL BACK dial +918870782800.For Top HelloTunes, Airtel Call Manager or Live aarti,call 567892 (tollfree)",
    "Akshara and 6 others have added their photos & also match your partner preference. View photos on EzhavaMatrimony by visiting www.cmatri.in/?kNw8dB~n~grNC",
    "Rs.1000.00 was withdrawn using your HDFC Bank Card ending 0449 on 2017-04-01:21:31:34 at +CTS THORAIPAKKAM OATM. Avl bal: Rs.18834.44",
    "Trying to loose weight? Meet our Bariartic Surgeon ¡ Apollo Spectra Hospital MRC Nagar.Spectra Obesity Check ¡ Rs 840.T & C apply. Call 66862000.",
    "Dear Associate, Cognizant, offices in Chennai and Coimbatore will remain closed today, Dec 6, 2016. Note: Associates working on CIS, BPS and 24/7 support projects, please contact your manager for instructions."] )

pred_few = classifier.predict(few_test)

for l in pred_few:
    if l == 0:
        print("Network")
    elif l == 1:
        print("Cognizant")
    elif l == 2:
        print("Promotional")
    elif l == 3:
        print("spam")
    elif l == 4:
        print("Trans")
    else:
        print("Personal")

print("Execution Completed")