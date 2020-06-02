#!/usr/bin/env python
# coding: utf-8

# # Course Project - IMDB moving rating

# ## 1 Import Packages

# In[1]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt



import os
for dirname, _, filenames in os.walk('C:\\USERS\\MI-PC\\Desktop\\HKUST\\MFIT5010\\IMDB'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[2]:


from os import path
from pandas import DataFrame
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
import re


# In[3]:


import nltk
from nltk.stem import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import LancasterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('wordnet')


# In[4]:


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import style
import matplotlib.colors


# In[5]:


import wordcloud   # Sentiment-based Word Clouds
from wordcloud import WordCloud, STOPWORDS 
from PIL import Image


# In[6]:


os.chdir("C:\\USERS\\MI-PC\\Desktop\\HKUST\\MFIT5010\\IMDB")
os.getcwd()


# In[7]:


df=pd.read_csv('C:/Users/MI-PC/Desktop/HKUST/MFIT5010/IMDB/IMDB Dataset.csv',header=0,error_bad_lines=True,encoding='utf8')
df.dtypes


# In[8]:


df.head()


# ### 2.1 WordCloud

# In[9]:


specific_wc = ['br', 'movie', 'film']
sw = list(set(stopwords.words('english')))
sw = sw + specific_wc

print(sw[:5])
print(len(sw))


# In[10]:


sentences = []
labels = []

for ind, row in df.iterrows():
    labels.append(row['sentiment'])
    sentence = row['review']
    for word in sw:
        token = " "+word+" "
        sentence = sentence.replace(token, " ")
        sentence = sentence.replace(" ", " ")
    sentences.append(sentence)


# #### Word Cloud on All Reviews

# In[11]:


wc = WordCloud(width = 600, height = 400, 
                    background_color ='white', 
                    stopwords = sw, 
                    min_font_size = 10, colormap='Paired_r').generate(' '.join(sentences[:100]))
plt.imshow(wc)


# #### Word Cloud on Positive Reviews

# In[12]:


pos_rev = ' '.join(df[df['sentiment']=='positive']['review'].to_list()[:10000])
wc = WordCloud(width = 600, height = 400, 
                    background_color ='white', 
                    stopwords = sw, 
                    min_font_size = 10, colormap='GnBu').generate(pos_rev)
plt.imshow(wc)


# #### Word Cloud on Negative Reviews

# In[13]:


pos_rev = ' '.join(df[df['sentiment']=='negative']['review'].to_list()[:10000])
wc = WordCloud(width = 600, height = 400, 
                    background_color ='white', 
                    stopwords = sw, 
                    min_font_size = 10, colormap='RdGy').generate(pos_rev)
plt.imshow(wc)


# ## 3 Sentimental Analysis
# ### 3.1 SentScore 
# #### Run sentimental analysis on "Review" and return compound value

# In[14]:


def sc(x):
    score=SentimentIntensityAnalyzer().polarity_scores(x)
    return score['compound']


# #### Create a new column "SentScore" to store the compound score of the sentimental analysis above 

# In[15]:


df["SentScore"]=df["review"].map(sc)


# In[16]:


df.head()


# ### 3.2 SentClass
# #### Run sentimental analysis on "Review" and return compound value

# In[17]:


def sca(lb):
    if lb >= .6:
        return "Very Good"
    elif (lb > .2) and (lb < .6):
        return "Good"
    elif (lb > -.2) and (lb < .2):
        return "Average"
    elif (lb > -.6) and (lb < -.2):
        return "Disappointing"
     
    else:
        return "Regrettable"


# #### Create a new column "SentClass" to to indicate the class of the review

# In[18]:


df["SentClass"]=df["SentScore"].map(sca)


# In[19]:


df.head(15)


# ### 3.3 Sentiment_bin
# #### Define a function for "sentiment" column, postive = 1 and nagative = 0

# In[20]:


def num(lb):
    if lb == 'positive':
        return 1   
    else:
        return 0


# #### Create a new column "sentiment_bin" to apply the function above

# In[21]:


df["sentiment_bin"]=df["sentiment"].map(num)


# In[22]:


df.head(15)


# ### 3.4 SentScore_bin
# #### Apply a similar function for "SentScore" column, postive = 1 and nagative = 0

# In[23]:


def numscore(lb):
    if lb >= 0:
        return 1     
    else:
        return 0


# #### Create a new column "SentScore_bin" to apply the function above

# In[24]:


df["SentScore_bin"]=df["SentScore"].map(numscore)


# In[25]:


df.head(15)


# ### 4 Text Adjustment and Cleaning

# #### a. Make Text Lower Case

# In[26]:


df["review"]  = df["review"].str.lower()


# #### b. Remove digits from text

# In[27]:


def Remove_digit(text):
    result = re.sub(r"\d", "", text)
    return result


# #### c. Remove HTML from text

# In[28]:


def remove_html(text):
    result = re.sub(r'<.*?>','',text) 
    return result


# #### d. Remove special text characters

# In[29]:


def remove_spl(text):
    result = re.sub(r'\W',' ',text) 
    return result


# #### e.  Link words with similar meaning as one word

# In[30]:


def lem_word(text):
    result= WordNetLemmatizer().lemmatize(text)
    return result


# #### Apply all functions above to column "review"

# In[31]:


df["review"]  = df["review"].apply(Remove_digit)
df["review"]  = df["review"].apply(remove_html)
df["review"]  = df["review"].apply(remove_spl)
df["review"]  = df["review"].apply(lem_word)


# In[32]:


df.head()


# #### Store the adjusted text to object 'corpus1' and transform into a list

# In[33]:


corpus1=df['review'].tolist()


# #### Create an object 'corpus' which includes the first 1000 values of 'corpus1'

# In[34]:


corpus=corpus1[ :1000]


# ### 4.2 Vectorisation
# #### Denfine N-gram range to be Unigrams (n-gram size = 1) and Bigrams (terms compounded by 2 words)

# In[35]:


from sklearn.feature_extraction import text

cv = text.CountVectorizer(input=corpus,ngram_range=(1,2),stop_words='english')
matrix = cv.fit_transform(corpus)

corpus2 = pd.DataFrame(matrix.toarray(), columns=cv.get_feature_names())


# In[36]:


corpus2.head()


# #### Dimension of Data

# There are 1000 rows which are consistent with the selected amount of rows of the list "corpus", and there are 110012 colums are humangous which indicated a giant matrix have been created.

# In[37]:


corpus2.shape


# ### 5 Term Frequency and Inverse Document Frequency
# #### Remove the English stop_words

# In[38]:


tf = text.TfidfVectorizer(input=corpus, ngram_range=(1,2),stop_words='english')

matrix1 = tf.fit_transform(corpus)

X = pd.DataFrame(matrix1.toarray(), columns=tf.get_feature_names())


# In[39]:


X.head()


# #### Set y to be the first 1000 values of column "SentScore_bin"

# In[40]:


y = df['SentScore_bin'][:1000].values


# In[41]:


print(y)


# ### 6 Run Multiple models on 'SentScore_bin' as "y"
# #### Split X and y in train and test data. Run Random Forest Classifier on X = Vectorized Matrix and y = SentScore_bin.

# #### 6.1 RandomForest Classifier

# In[42]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=23)


# #### Set the Random Forest Classifier and parameters. Fit the model on X and y train data.

# In[43]:


from sklearn.ensemble import RandomForestClassifier
text_classifier=RandomForestClassifier(bootstrap=False, criterion="gini", max_features=0.3, min_samples_leaf=4, min_samples_split=9, n_estimators=100)
text_classifier.fit(X_train, y_train)


# #### Run the prediction on X test data and store in the object 'predictions'

# In[44]:


predictions = text_classifier.predict(X_test)


# #### Accuracy Score

# In[45]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
 
print(confusion_matrix(y_test,predictions))  
print(classification_report(y_test,predictions))  
print(accuracy_score(y_test, predictions))


# #### 6.2 Logistic Regression

# In[46]:


from sklearn.linear_model import LogisticRegression


# #### Train the model and Fitting the model for TFIDF Features

# In[47]:


lr=LogisticRegression(C=1.0,class_weight=None,dual=False,fit_intercept=True,intercept_scaling=1,l1_ratio=None,max_iter=100,
multi_class='auto',n_jobs=None,penalty='l2',random_state=23,solver='lbfgs',tol=0.0001,verbose=0,warm_start=False)

lr_tfidf=lr.fit(X_train,y_train)
print(lr_tfidf)


# #### Predicting the model for TFIDF Features

# In[48]:


lr_tfidf_predict=lr.predict(X_test)
print(lr_tfidf_predict)


# #### Accuracy score

# In[49]:


lr_tfidf_score=accuracy_score(y_test,lr_tfidf_predict)
print("lr_tfidf_score :",lr_tfidf_score)


# #### Classification Report for TFIDF Features

# In[50]:


lr_tfidf_report=classification_report(y_test,lr_tfidf_predict,target_names=['0','1'])
print(lr_tfidf_report)


# #### 6.3 GRADIENT BOOSTING CLASSIFIER

# In[45]:


from sklearn.ensemble import GradientBoostingClassifier


# In[46]:


clf=GradientBoostingClassifier(n_estimators=80,random_state=23)


# In[47]:


clf.fit(X_train,y_train)


# In[48]:


clf.score(X_test,y_test)


# In[49]:


from sklearn.model_selection import GridSearchCV
mod=GridSearchCV(clf,param_grid={'n_estimators': [80,100,120,140,160]})


# In[50]:


mod.fit(X_train,y_train)


# In[51]:


mod.best_estimator_


# In[52]:


clf=GradientBoostingClassifier(n_estimators=100,random_state=23)
clf.fit(X_train,y_train)


# In[53]:


clf.score(X_test,y_test)


# In[54]:


clf.feature_importances_


# In[55]:


feature_imp=pd.Series(clf.feature_importances_)
feature_imp.sort_values(ascending=False)


# ### 7 Run Multiple models on 'sentiment_bin' as "y"
# #### Repeat as above, but set the first 1000 values of column 'sentiment_bin' as "y"

# In[67]:


y = df['sentiment_bin'][:1000].values


# #### Split the data in train and test data

# In[68]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=23)


# #### 7.1 Random Forest Classifier

# In[59]:


from sklearn.ensemble import RandomForestClassifier
text_classifier=RandomForestClassifier(bootstrap=False, criterion="gini", max_features=0.3, min_samples_leaf=4, min_samples_split=9, n_estimators=100)
text_classifier.fit(X_train, y_train)


# In[60]:


predictions = text_classifier.predict(X_test)


# #### Accuracy Score

# In[61]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
 
print(confusion_matrix(y_test,predictions))  
print(classification_report(y_test,predictions))  
print(accuracy_score(y_test, predictions))


# #### 7.2 Logistic Regression

# In[62]:


from sklearn.linear_model import LogisticRegression
#training the model
lr=LogisticRegression(C=1.0,class_weight=None,dual=False,fit_intercept=True,intercept_scaling=1,l1_ratio=None,max_iter=100,
multi_class='auto',n_jobs=None,penalty='l2',random_state=23,solver='lbfgs',tol=0.0001,verbose=0,warm_start=False)
#Fitting the model for tfidf features
lr_tfidf=lr.fit(X_train,y_train)
print(lr_tfidf)


# #### Predicting the model for TFIDF Features

# In[63]:


lr_tfidf_predict=lr.predict(X_test)
print(lr_tfidf_predict)


# #### Accuracy Score

# In[64]:


lr_tfidf_score=accuracy_score(y_test,lr_tfidf_predict)
print("lr_tfidf_score :",lr_tfidf_score)


# #### Classification report for TFIDF Features

# In[65]:


lr_tfidf_report=classification_report(y_test,lr_tfidf_predict,target_names=['0','1'])
print(lr_tfidf_report)


# #### 7.3 Gradient Boosting Classifier

# In[69]:


from sklearn.ensemble import GradientBoostingClassifier
clf=GradientBoostingClassifier(n_estimators=80,random_state=23)
clf.fit(X_train,y_train)


# In[70]:


clf.score(X_test,y_test)


# In[72]:


from sklearn.model_selection import GridSearchCV
mod=GridSearchCV(clf,param_grid={'n_estimators': [80,100]})


# In[73]:


mod.fit(X_train,y_train)


# In[74]:


mod.best_estimator_


# In[76]:


clf=GradientBoostingClassifier(ccp_alpha=0.0, criterion='friedman_mse', init=None,
                           learning_rate=0.1, loss='deviance', max_depth=3,
                           max_features=None, max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=100,
                           n_iter_no_change=None, presort='deprecated',
                           random_state=23, subsample=1.0, tol=0.0001,
                           validation_fraction=0.1, verbose=0,
                           warm_start=False)
clf.fit(X_train,y_train)


# In[77]:


clf.score(X_test,y_test)

