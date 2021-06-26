## This library is for detecting fake content present in the internet. This library is for 7 languages out of them 1 is English and 4 is European (Italian, German, French & Spanish) and 2 is Indian (Hindi & Bengali).

[GitHub](https://github.com/soumayan/fake-news-detection "You can find models in this link")

**Installing library (First step)**
`pip install soumayan4==1.0.2`

***
### Now we will see how to implement this library upon English and other 4 European languages.

**Downloading part for English and 4 European languages. This code should be run after you pip install the above library, else you will get error.**

```python
!polyglot download ner2.en    # downloading model ner

!polyglot download pos2.en    # downloading model pos

!polyglot download sentiment2.en  # downloading model sentiment


!polyglot download embeddings2.en
!polyglot download pos2.en

!polyglot download embeddings2.fr
!polyglot download pos2.fr

!polyglot download embeddings2.es
!polyglot download pos2.es

!polyglot download embeddings2.de
!polyglot download pos2.de

!polyglot download embeddings2.it
!polyglot download pos2.it

!python -m spacy download en_core_web_sm
!polyglot download sentiment2.en
!python -m spacy download fr_core_news_sm
!polyglot download sentiment2.fr
!python -m spacy download de_core_news_sm
!polyglot download sentiment2.de
!python -m spacy download it_core_news_sm
!polyglot download sentiment2.it
!python -m spacy download es_core_news_sm
!polyglot download sentiment2.es
```

**Now we will see how to use this library**

```python
from soumayan4 import italian_fake  # you can import other functions also like german_fake

data={'text':['warmes Wasser entfernt Korona','how are you?','we are all fine']}
import pandas as pd
df = pd.DataFrame(data) #This is small data for testing our library

!wget https://github.com/soumayan/fake-news-spreader/blob/main/italian/italian_model_svm.sav?raw=true
#This above code is for downloading model present in github, you can change language and model name to use different types of model and languages

italian_fake(df,'text','svm')
#This is how you have to give input to the model, first argument is your dataframe name, second argument is attribute name upon which you want to apply this library, here it is text. Third one is the model name, here model name should be same what you have downloaded before using wget

df.head()
#now you will see there are many features present like NER and other POS with news_output column. If news_output is 0 then it is real else content is fake
```
***
## Now we will see how to implement this library upon bengali language.

**First we have to download some libraries in models directory and have to change current directory to models**

```python
!pip install -U bnlp_toolkit
!mkdir models
%cd models
!wget https://github.com/sagorbrur/bnlp/raw/master/model/bn_spm.model
!wget https://github.com/sagorbrur/bnlp/raw/master/model/bn_spm.vocab
!wget https://github.com/sagorbrur/bnlp/raw/master/model/bn_ner.pkl
!wget https://github.com/sagorbrur/bnlp/raw/master/model/bn_pos.pkl
!wget https://github.com/soumayan/fake-news-spreader/blob/main/bengali/bengali_model_knn.sav?raw=true
```
**Now we will create a small dataset and will apply our library upon this**

```python
data={'text':['বিজেপি কখনও জাল খবর ছড়ায় না']}
#data={'text':['কিছু লোক ভুয়া খবর ছড়িয়ে দেয়']}

import pandas as pd
t = pd.DataFrame(data) # creating dataframe

from soumayan4 import bengali_fake

train=bengali_fake(t,'text','knn')
train.head() # you will get your result in news_output column and you will also get some additional features like NER and POS.
```
***
## Now we will see how to implement this upon Hindi language.

```python
!wget https://github.com/soumayan/fake-news-spreader/blob/main/hindi/hindi_model_svm.sav?raw=true 
#install the models according to your need

data={'text':['हर कोई फर्जी खबर फैलाता है']}
import string
import pandas as pd
df = pd.DataFrame(data) 
# creating a small dataset for testing

from soumayan4 import hindi_fake
t=hindi_fake(df,'text','svm')

t.head()
#you will get output in news_output column of dataframe t

```

**Note**  - **For bigger datasets like 12000 to 15000 it can take upto 20 minutes time and currently this library will run error free in google colab but we are not sure about other environments** 
