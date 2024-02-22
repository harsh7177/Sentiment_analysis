


import tensorflow 
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import one_hot, Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Embedding,Activation,Dropout,LSTM
from tensorflow.keras.layers import Conv1D, MaxPool1D, GlobalAveragePooling1D
import numpy as np 
from numpy import array 
import pandas as pd
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report
from wordcloud import WordCloud
import nltk,re
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


fake=pd.read_csv('https://raw.githubusercontent.com/laxmimerit/fake-real-news-dataset/main/data/Fake.csv')


# In[7]:


fake.head(5)


# In[29]:


real=pd.read_csv('https://raw.githubusercontent.com/laxmimerit/fake-real-news-dataset/main/data/True.csv')


# In[5]:


fake.columns


# In[6]:


fake['subject'].value_counts()


# In[9]:


plt.figure(figsize=(10,6))
sns.countplot(x='subject',data=fake)


# In[10]:


# Wordcloud


# In[16]:


text=' '.join(fake['text'].tolist())


# In[21]:


wordcloud=WordCloud(width=1920,height=1000).generate(text)


# In[22]:


fig=plt.figure(figsize=(10,10))
plt.imshow(wordcloud)
plt.axis('off')  
plt.tight_layout(pad=0)
plt.show()


# In[23]:


real['subject'].value_counts()


# In[24]:


text=' '.join(real['text'].tolist())


# In[25]:


wordcloud=WordCloud(width=1920,height=1000).generate(text)


# In[26]:


fig=plt.figure(figsize=(10,10))
plt.imshow(wordcloud)
plt.axis('off')  
plt.tight_layout(pad=0)
plt.show()


# # difference in texxt:
# 1. some texts are ttweets from twitter
# 2. few text do not contain any publication info
# 3. most of text contains reuters info such as WASHINGTON (Reuters)

# In[27]:


# cleaning of data


# In[28]:


real.sample(5)


# In[45]:


unknown_publishers=[]
for index,row in enumerate(real.text.values):
  try:
    record=row.split(' - ', maxsplit=1)
    record[1]

    assert(len(record[0])<120)
  except:
    unknown_publishers.append(index)


# In[46]:


len(unknown_publishers)


# In[47]:


real.iloc[unknown_publishers].text


# In[43]:


real=real.drop(8970,axis=0)


# In[ ]:





# In[48]:


publisher = []
temp_text = []
for index,row in enumerate(real.text.values):
  if index in unknown_publishers:
    temp_text.append(row)
    publisher.append('unknown')
  else:
    record=row.split(" - ", maxsplit=1)
    publisher.append(record[0].strip())
    temp_text.append(record[1].strip())


# In[49]:


real['publisher']=publisher 
real['text']=temp_text


# In[50]:


real.head()


# In[51]:


real.shape


# In[53]:


empty_fake_index=[index for index,text in enumerate(fake.text.tolist()) if str(text).strip()=='']


# In[54]:


fake.iloc[empty_fake_index]


# In[56]:


real['text']=real['title']+' '+real['text']
fake['text']=fake['title']+' '+fake['text']


# In[57]:


real['text']=real['text'].apply(lambda x: str(x).lower())
fake['text']=fake['text'].apply(lambda x: str(x).lower())


# In[60]:


real['class']=1
fake['class']=0


# In[61]:


real=real[['text','class']]


# In[62]:


fake=fake[['text','class']]


# In[64]:


data=real.append(fake,ignore_index=True)


# In[65]:


get_ipython().system('pip install spacy==2.2.3')
get_ipython().system('python -m spacy download en_core_web_sm')
get_ipython().system('pip install beautifulsoup4==4.9.1')
get_ipython().system('pip install textblob==0.15.3')
get_ipython().system('pip install git+https://github.com/laxmimerit/preprocess_kgptalkie.git --upgrade --force-reinstall')


# In[69]:


data['text']=data['text'].apply(lambda x: re.sub('[^a-z A-Z 0-9]+','',x))


# In[70]:


#word2vec:


# In[72]:


data['text']


# In[73]:


import gensim


# In[74]:


y=data['class'].values


# In[77]:


X=[d.split() for d in data['text'].tolist()]


# In[78]:


print(X[0])


# In[80]:


DIM=100
w2v_model=gensim.models.Word2Vec(sentences=X,size=DIM,window=10,min_count=1)


# In[81]:


w2v_model['love']


# In[82]:


w2v_model.most_similar('india')


# In[83]:


tokenizer=Tokenizer()
tokenizer.fit_on_texts(X)


# In[84]:


X=tokenizer.texts_to_sequences(X)


# In[86]:


tokenizer.word_index


# In[91]:


plt.hist([len(x) for x in X],bins=700)
plt.show()


# In[92]:


nos=np.array([len(x) for x in X]) 
len(nos[nos>1000])


# In[93]:


maxlen=1000
X=pad_sequences(X,maxlen=maxlen)


# In[94]:


X


# In[95]:


len(X[100])


# In[96]:


vocab_size=len(tokenizer.word_index)+1


# In[97]:


vocab=tokenizer.word_index


# In[99]:


def get_weight_matrix(model):
  weight_matrix=np.zeros((vocab_size, DIM))
  for word,i in vocab.items():
    weight_matrix[i]=model.wv[word] 
    return weight_matrix


# In[100]:


embedding_vectors=get_weight_matrix(w2v_model)


# In[102]:


embedding_vectors.shape


# In[107]:


model=Sequential()
model.add(Embedding(vocab_size,output_dim=DIM,weights=[embedding_vectors],input_length=maxlen,trainable=False)) # false as we already trained
model.add(LSTM(units=128))
model.add(Dense(1,activation='sigmoid'))
model.compile(optimizer='adam',loss='binary_crossentropy', metrics=['acc'])


# In[104]:


model.summary()


# In[113]:


X_train,X_test,y_train,y_test = train_test_split(X,y)


# In[109]:


model.fit(X_train,y_train,validation_split=0.3,epochs=6)


# In[110]:


y_pred=(model.predict(X_text) >=0.5).astype(int)


# In[122]:


print(classification_report(y_test, y_pred))


# In[128]:


x=['this is a news']
x=tokenizer.texts_to_sequences(x)
x=pad_sequences(x,maxlen=maxlen)


# In[131]:


(model.predict(x)>=0.5).astype(int)  ### result = fake


