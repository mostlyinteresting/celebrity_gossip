import os
from concurrent.futures import ProcessPoolExecutor as PoolExecutor
import requests
from bs4 import BeautifulSoup
import pandas as pd
import sqlite3
import time
import random
import numpy as np
import datetime as dt
import re
import json
from requests_html import HTMLSession

conn = sqlite3.connect('celebrity_news.sqlite')

platform = 'tmz'

def saveSubmission(data, dCONN, tableName, try_number=1, if_exists = 'append'):
    try:
        data.to_sql(tableName, dCONN, if_exists=if_exists, index=False)
    except:
        time.sleep(2**try_number + random.random()*0.01) #exponential backoff
        return saveSubmission(data, dCONN, tableName, try_number=try_number+1)
    else:
        return
    
def getProxies():
    #calls api proxy service and returns proxies
    return

def createSessions(proxiesAll, n):
    
    sessions = {}
    for k in range(n):
            
        u = proxiesAll.loc[k,'username']
        p = proxiesAll.loc[k,'password']
        address = proxiesAll.loc[k,'proxy_address']
        port = proxiesAll.loc[k,'ports']['http']
        
        proxies = {
        'http': f'http://{u}:{p}@{address}:{port}'
        }
        
        s = HTMLSession()
        s.proxies = proxies
        
        sessions[k] = s
        
    return sessions

def getArticleTitle(x):
    x = x.findAll('span')
    return ' '.join([ y.get_text() for y in x ])

def getArticleLink(x):
    return x.find('a')['href']

def getSubmission(url, s, try_number=1):
    try:
        response = s.get(url)
    except:
        time.sleep(2**try_number + random.random()*0.01) #exponential backoff
        return getSubmission(url, s, try_number=try_number+1)
    else:
        return response
    
def getArticle(url, session):

    def cleanElement(x):
        if x.startswith('\n') and x.endswith('\n'):
            x = ''
        return x
    
    r = session.get(url, timeout = 5)
    soup = BeautifulSoup(r.text, "lxml")
        
    
    try:
        t = soup.find('h5',{'class':'article__published-at'}).get_text().replace('\n','')
        t = re.split('\s{2,}',t)
        if '' in t:
            t.remove('')
        if len(t) == 3:
            comments = t[1]
        else:
            comments = t[0]
        t = t[-1]
        t = pd.to_datetime(t)
    except:
        try:
            t = soup.find('h5',{'class':'article__published-at'}).get_text().replace('\n','')
            t = re.split('\s+',t, maxsplit = 1)[-1].strip()
            comments = None
        except:
            t = None
            comments = None
            
    p = soup.find('div',{'class':'article__blocks clearfix'}).findAll('p')
    article = ' '.join([ x.get_text() for x in p ])
    
    try:
        title = soup.find('h1',{'class':'article__headline-title'}).get_text()
    except:
        try:
            title = soup.find('h1',{'class':'article__header-title'}).get_text()
        except:
            title = None
    try:
        subtitle = soup.find('h2',{'class':'article__header-title'}).findAll('span')
        subtitle = ' '.join([ x.get_text() for x in subtitle ])
    except:
        subtitle = None
    
    out = [url, t, title, subtitle, article, comments]
    outCols = ['url', 't', 'title', 'subtitle', 'article','comments']
    out = pd.DataFrame([out], columns = outCols)
    
    return out

def getShares(x):
    
    sharesPattern = re.compile(r'\d+\s+\bshares\b')
    try:
        shares = int(''.join(re.findall(sharesPattern, x)).replace(' shares','').strip())
    except:
        shares = 0

    return shares

def process(k, data, limit, workers, sessions):
    
    data = data.loc[(k*limit):((k+1)*limit),:].reset_index(drop = True)
    sessions = sessions[(k*workers):((k+1)*workers):]
    
    limit = int(np.ceil(data.shape[0]/workers))
    
    for l in range(limit):
        
        articles = []
        for i in range(workers):
            
            session = sessions[i]
            ind = i+workers*l
            if ind < data.shape[0]:
                
                url = data.loc[ind,'url']
                
                try:
                    article = getArticle(url, session)
                    articles.append(article)
                except:
                    pass
    
        articles = pd.concat(articles)
        if articles.shape[0] > 0:
            saveSubmission(articles, conn, f'{platform}_articles0', try_number=1, if_exists = 'append')

def main():
    
    proxiesAll = getProxies()
    workers = proxiesAll.shape[0]
    sessions = createSessions(proxiesAll, n = workers)
    sessions = list(sessions.values())
    
    urls = pd.read_sql_query(f"SELECT * FROM '{platform}_urls'", conn)
    ex = pd.read_sql_query(f"SELECT * FROM {platform}_articles0", conn)
    
    urls = urls[ ~urls['url'].isin(ex['url'].tolist()) ].reset_index(drop = True)
    
    n = urls.shape[0]
    executors = 5
    workers = int(workers/executors)
    limit = int(np.ceil(n/executors))
    
    with PoolExecutor(max_workers=executors) as executor:
        for k in range(executors):
            executor.map(process, [k], [urls], [limit], [workers], [sessions])
    
if __name__ == '__main__':
    
    main()
