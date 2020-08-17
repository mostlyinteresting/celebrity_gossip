import os
import re
import numpy as np
import pandas as pd
import time
import datetime as dt
import random
import pytz
import sqlite3

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow_hub as hub

import tokenization

import nltk
from nltk.corpus import stopwords
from  nltk.stem import SnowballStemmer
# nltk.download('stopwords')

import re
import numpy as np
#from sklearn.manifold import TSNE

random_state = 12345

stop_words = stopwords.words("english")
stemmer = SnowballStemmer("english")

hashtags = re.compile(r"^#\S+|\s#\S+")
mentions = re.compile(r"^@\S+|\s@\S+")
https = re.compile(r"https?://\S+")
http = re.compile(r"http?://\S+")

connTwitter = sqlite3.connect('twitter.sqlite')
conn = sqlite3.connect('celebrity_news.sqlite')

def bert_encode(texts, tokenizer, max_len=512):
    all_tokens = []
    all_masks = []
    all_segments = []
    
    for text in texts:
        text = tokenizer.tokenize(text)
            
        text = text[:max_len-2]
        input_sequence = ["[CLS]"] + text + ["[SEP]"]
        pad_len = max_len - len(input_sequence)
        
        tokens = tokenizer.convert_tokens_to_ids(input_sequence)
        tokens += [0] * pad_len
        pad_masks = [1] * len(input_sequence) + [0] * pad_len
        segment_ids = [0] * max_len
        
        all_tokens.append(tokens)
        all_masks.append(pad_masks)
        all_segments.append(segment_ids)
    
    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)

def build_model(bert_layer, max_len=512):
    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    input_mask = Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
    segment_ids = Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")

    _, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
    clf_output = sequence_output[:, 0, :]
    out = Dense(1, activation='sigmoid')(clf_output)
    
    model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)
    model.compile(Adam(lr=1e-5), loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

def process_text(text):
    text = hashtags.sub(' hashtag', text)
    text = mentions.sub(' entity', text)
    text = https.sub(' website', text)
    text = http.sub(' website', text)
  
    return text.strip().lower()

if __name__ == '__main__':
    
    handle = 'tmz'
    yVar0 = 'retweets'
    yVar = 'label'
    
    tweets = pd.read_sql_query(f"SELECT * FROM {handle}", connTwitter)
    tweets = tweets[ tweets['urls'] != '' ]
    tweets.drop_duplicates('urls', keep = 'first', inplace = True)
    
    urls = pd.read_sql_query(f"SELECT * FROM {handle}_tweets", conn)
    tweets = tweets.merge(urls, how = 'inner', on = ['urls','id'])

    data = pd.read_sql_query(f"SELECT * FROM {handle}_articles0", conn)
    data.rename(columns = {'url':'urlFull'}, inplace = True)
    data = data.merge(tweets, how = 'inner', on = ['urlFull'])    
    
    data.loc[ (data['t'].isnull()) & (data['comments'].str.contains('/')),'t'] = data.loc[data['t'].isnull() & (data['comments'].str.contains('/')),'comments']
    data['t'] = pd.to_datetime(data['t'])
    data['text'] = data['title'] + ' ' + data['subtitle'] + ' ' +  data['article']
    data = data[ data['text'].notnull() ]
    data['text'] = data['text'].apply(process_text)
    data.sort_values(by = 't', ascending = True, inplace = True)
    data.reset_index(drop = True, inplace = True)
    dailyMedianCount = data.set_index('t').groupby(pd.Grouper(freq = 'D'))['urlFull'].count().median()
    data[f'{yVar0}q'] = data[yVar0].rolling(int(dailyMedianCount*30)).quantile(.7).fillna(method='bfill')
    data[yVar] = 0
    data.loc[ data[yVar0] > data[f'{yVar0}q'],yVar] = 1
    data[yVar] = data[yVar].astype(int)
    
    module_loc = "https://tfhub.dev/tensorflow/bert_en_wwm_uncased_L-24_H-1024_A-16/2"
    bert_layer = hub.KerasLayer(module_loc, trainable=True)
    
    modelName = f'{handle}_significance_bert_2.h5'
    
    testSize = 0.2
    train, test = train_test_split(data, test_size=testSize,shuffle = False)
    maxLen = 160
    
    # n_classes = train[yVar].nunique()
    
    vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
    do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
    tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)
    
    #visualization
    # all_input = bert_encode(data['text'].values, tokenizer, max_len=maxLen)
    # X = all_input[0]
    # X_embedded = TSNE(n_components=2, random_state=random_state).fit_transform(X)
    # X_embedded = pd.DataFrame(X_embedded, columns = ['x','y'])
    # X_embedded.index = data.index
    # X_embedded.plot(x = 'x', y = 'y', kind = 'scatter')
    
    #prediction
    train_input = bert_encode(train['text'].values, tokenizer, max_len=maxLen)
    test_input = bert_encode(test['text'].values, tokenizer, max_len=maxLen)
    train_labels = train[yVar].values
    test_labels = test[yVar].values
    
    counts = np.unique(train_labels, return_counts = True)
    counts = pd.DataFrame(data = {'count':counts[1]}, index = counts[0]).sort_values('count', ascending = False)
    counts['m'] = counts['count'].values[0]/counts['count']
    
    class_weight = dict(zip(counts.index, counts['m']))
    
    model = build_model(bert_layer, max_len=maxLen)
    # model.summary()
    
    checkpoint = ModelCheckpoint(modelName, monitor='val_loss', save_best_only=True)

    train_history = model.fit(
        train_input, train_labels,
        validation_data = (test_input, test_labels),
        shuffle = False,
        epochs=8,
        class_weight=class_weight,
        callbacks=[checkpoint],
        batch_size=16
    )
    
    model.load_weights(modelName)
    test_pred = model.predict(test_input)
    test_pred = np.round(test_pred).reshape(-1)
    
    print(classification_report(test_labels, test_pred))
    
