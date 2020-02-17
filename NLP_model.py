
# coding: utf-8

# In[24]:


import pandas as pd
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
import pickle
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


# In[4]:


df = pd.read_csv('spam.csv' , encoding='latin-1')
df.drop(['Unnamed: 2' , 'Unnamed: 3', 'Unnamed: 4'] ,axis = 1 , inplace = True)


# In[9]:


df['label'] = df['class'].map({'ham': 0 , 'spam': 1})
X = df['message']
y = df['label']


# In[15]:


cv = CountVectorizer()
X = cv.fit_transform(X)


# In[17]:


pickle.dump(cv, open('transform.pkl' ,'wb'))


# In[28]:


X_train, X_test , y_train , y_test = train_test_split(X , y , test_size  = 0.3 , random_state = 40)
clf = MultinomialNB()
clf.fit(X_train , y_train)
pickle.dump(clf , open('nlp_model.pkl' , 'wb'))

