import pandas as pd
msg=pd.read_csv('naivetext.csv',names=['message','label'])

print('the dimensions of the dataset',msg.shape)
msg['labelnum']=msg.label.map({'pos':1,'neg':0})
x=msg.message
y=msg.labelnum
print(x)
print(y)

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y)

print('\n the total number of training data:',ytrain.shape)
print('\n the total number of test data:',ytest.shape)

from sklearn.feature_extraction.text import CountVectorizer
count_vect=CountVectorizer()
xtrain_dtm=count_vect.fit_transform(xtrain)
xtest_dtm=count_vect.transform(xtest)
print('\n the words or toens in the text documents\n')
print(count_vect.get_feature_names())

df=pd.DataFrame(xtrain_dtm.toarray(),columns=count_vect.get_feature_names())

from sklearn.naive_bayes import MultinomialNB
df=MultinomialNB().fit(xtrain_dtm,ytrain)
predicted=df.predict(xtest_dtm)

from sklearn import metrics
print('\n accuracy of the classifier is',metrics.accuracy_score(ytest,predicted))

print('\n confusion matrix',)
print(metrics.confusion_matrix(ytest,predicted))
print('\n the values of precison',metrics.precision_score(ytest,predicted))
print('\n the value of recall',metrics.recall_score(ytest,predicted))
                                    
