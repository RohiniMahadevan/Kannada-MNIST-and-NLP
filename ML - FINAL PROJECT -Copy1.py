#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
get_ipython().system('pip install -U spacy')
get_ipython().system('python -m spacy download en_core_web_sm')
import spacy
from sklearn.model_selection import train_test_split
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import confusion_matrix,recall_score,f1_score,precision_score
import math


# In[6]:


df = pd.read_csv(r"C:\Users\rohin\Downloads\FinalBalancedDataset.csv\FinalBalancedDataset.csv")


# In[7]:


len(df)


# In[8]:


df.shape


# In[9]:


df.head()


# In[10]:


df.drop(['Unnamed: 0'], axis=1)


# In[11]:


import string
    
def remove_punc_dig(text : str):
    '''
    text : str 
    This function will remove all the punctuations and digits from the "text"
    '''
    to_remove = string.punctuation + string.digits
    cur_text = ""
    for i in range(len(text)):
        if text[i] in to_remove:
            cur_text += " "
        else:
            cur_text += text[i].lower()
    cur_text = " ".join(cur_text.split())
    return cur_text


# In[12]:


df['cur_tweet'] = df['tweet'].apply(lambda x:remove_punc_dig(x))


# In[13]:


df.head()


# In[14]:


df.drop(['tweet'], axis=1)


# In[15]:


from spacy.lang.en.stop_words import STOP_WORDS


# In[16]:


nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])


# In[17]:


def remove_stop_words(text: str):
    '''
    text : str
    This function will remove stop words like I,my,myself etc
    '''
    filtered_sentence = []
    for word in text.split(' '):
        lexeme = nlp.vocab[word]
        if lexeme.is_stop == False:
            filtered_sentence.append(word) 
    return " ".join(filtered_sentence)


# In[18]:


df['filtered_cur_tweet'] = df['cur_tweet'].apply(lambda x : remove_stop_words(x))


# In[19]:


df.drop(['cur_tweet'], axis=1)


# In[20]:


def lemmatizer(text : str):
    '''
    text : str
    Applying lemmatization for all words of "text"
    '''
    return " ".join([token.lemma_ for token in nlp(text)])


# In[21]:


df['lemma_cur_tweet'] = df['filtered_cur_tweet'].apply(lambda x : lemmatizer(x))


# In[22]:


df.drop(['filtered_cur_tweet'], axis=1)


# In[23]:


from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


# In[24]:


bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, stop_words='english')
bow = bow_vectorizer.fit_transform(df['lemma_cur_tweet'])
bow.shape


# In[25]:


tfidf_vectorizer = TfidfVectorizer(max_df=0.90, min_df=2, stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(df['lemma_cur_tweet'])
tfidf.shape


# In[26]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB


# In[27]:


X = df.loc[:,'lemma_cur_tweet']
Y = df['Toxicity']


# In[28]:


X_train, X_test,y_train, y_test = train_test_split(X,Y, random_state=100, test_size=0.20)


# In[29]:


X_train, X_test,y_train, y_test = train_test_split(bow,df['Toxicity'],random_state=100,test_size = 0.20)


# In[30]:


model_1 = DecisionTreeClassifier()
model_1.fit(X_train, y_train)


# In[31]:


train_pred = model_1.predict(X_train)
test_pred = model_1.predict(X_test)


# In[32]:


predictions = model_1.predict(X_test)
predictions


# In[33]:


from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))


# In[34]:


print(confusion_matrix(y_test,predictions))


# In[92]:


import seaborn as sns
cm = confusion_matrix(y_test,test_pred)
cm_matrix = pd.DataFrame(data=cm,columns=['Actual Positive', 'Actual Negative'],
                        index=['Predict Positive', 'Predict Negative'])
sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')
plt.show()


# In[35]:


model_1.score(X_train,y_train),model_1.score(X_test,y_test)


# In[36]:


model_2 = RandomForestClassifier()
model_2.fit(X_train, y_train)


# In[37]:


predictions_1 = model_2.predict(X_test)
predictions_1


# In[38]:


print(classification_report(y_test,predictions_1))


# In[39]:


print(confusion_matrix(y_test,predictions_1))


# In[94]:


import seaborn as sns
cm = confusion_matrix(y_test,predictions_1)
cm_matrix = pd.DataFrame(data=cm,columns=['Actual Positive', 'Actual Negative'],
                        index=['Predict Positive', 'Predict Negative'])
sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')
plt.show()


# In[40]:


model_2.score(X_train,y_train),model_2.score(X_test,y_test)


# In[41]:


model_3 = KNeighborsClassifier(n_neighbors=2)
model_3.fit(X_train,y_train)


# In[42]:


predictions_2 = model_3.predict(X_test)
predictions_2


# In[43]:


print(classification_report(y_test,predictions_2))


# In[44]:


print(confusion_matrix(y_test,predictions_2))


# In[95]:


import seaborn as sns
cm = confusion_matrix(y_test,predictions_2)
cm_matrix = pd.DataFrame(data=cm,columns=['Actual Positive', 'Actual Negative'],
                        index=['Predict Positive', 'Predict Negative'])
sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')
plt.show()


# In[45]:


model_3.score(X_train,y_train),model_3.score(X_test,y_test)


# In[46]:


from sklearn.svm import LinearSVC
from sklearn import svm
model_4 = svm.LinearSVC()
model_4.fit(X_train,y_train)


# In[47]:


predictions_3 = model_4.predict(X_test)
predictions_3


# In[48]:


print(classification_report(y_test,predictions_3))


# In[49]:


print(confusion_matrix(y_test,predictions_3))


# In[96]:


import seaborn as sns
cm = confusion_matrix(y_test,predictions_3)
cm_matrix = pd.DataFrame(data=cm,columns=['Actual Positive', 'Actual Negative'],
                        index=['Predict Positive', 'Predict Negative'])
sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')
plt.show()


# In[50]:


model_4.score(X_train,y_train),model_4.score(X_test,y_test)


# In[51]:


from sklearn.naive_bayes import MultinomialNB
model_5 = MultinomialNB()
model_5.fit(X_train,y_train)


# In[52]:


predictions_4 = model_5.predict(X_test)
predictions_4


# In[53]:


print(classification_report(y_test,predictions_4))


# In[54]:


print(confusion_matrix(y_test,predictions_4))


# In[97]:


import seaborn as sns
cm = confusion_matrix(y_test,predictions_4)
cm_matrix = pd.DataFrame(data=cm,columns=['Actual Positive', 'Actual Negative'],
                        index=['Predict Positive', 'Predict Negative'])
sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')
plt.show()


# In[55]:


model_5.score(X_train,y_train),model_5.score(X_test,y_test)


# In[59]:


print(f"Accuracy score: {accuracy_score(y_test, predictions)}") #Decisiontree


# In[72]:


print(f"Accuracy score: {accuracy_score(y_test, predictions_1)}") #Randomforest


# In[73]:


print(f"Accuracy score: {accuracy_score(y_test, predictions_2)}") #Kneigboursclassifier


# In[74]:


print(f"Accuracy score: {accuracy_score(y_test, predictions_3)}") #Linearsvc


# In[75]:


print(f"Accuracy score: {accuracy_score(y_test, predictions_4)}")  #MultinomialNB


# In[76]:


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix


# In[81]:


y_scores = model_1.predict_proba(X_test)
fpr, tpr, thresholds = roc_curve(y_test, y_scores[:,1])   #Decisiontrees


# In[82]:


fig = plt.figure(figsize=(6, 6))
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()


# In[79]:


y_scores = model_2.predict_proba(X_test)
fpr, tpr, thresholds = roc_curve(y_test, y_scores[:,1])   #Randomforest


# In[80]:


fig = plt.figure(figsize=(6, 6))
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()


# In[83]:


y_scores = model_3.predict_proba(X_test)
fpr, tpr, thresholds = roc_curve(y_test, y_scores[:,1])   #KNeighborsclassifier


# In[84]:


fig = plt.figure(figsize=(6, 6))
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()


# In[86]:


y_scores = model_5.predict_proba(X_test)
fpr, tpr, thresholds = roc_curve(y_test, y_scores[:,1])  #MultinomialNB


# In[87]:


fig = plt.figure(figsize=(6, 6))
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

