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

def saveSubmission(data, dCONN, tableName, try_number=1, if_exists = 'append'):
    try:
        data.to_sql(tableName, dCONN, if_exists=if_exists, index=False)
    except:
        time.sleep(2**try_number + random.random()*0.01) #exponential backoff
        return saveSubmission(data, dCONN, tableName, try_number=try_number+1)
    else:
        return
    
def getProxies():
    #calls api service and returns proxies
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
    
def getPage(url, session):

    r = getSubmission(url, session, try_number=1)
    soup = BeautifulSoup(r.text, "lxml")
    articles = soup.findAll('header',{'class':'article__header'})
    urls = [ getArticleLink(x) for x in articles ]
    articles = [ getArticleTitle(x) for x in articles ]
    nextURL = 'https://www.tmz.com' + soup.select('a[data-context*="next-page"]')[0]['href']
    out = pd.DataFrame({'article':articles, 'url':urls})
    
    return out, nextURL

def main():
    
    platform = 'tmz'
    
    proxiesAll = getProxies()
    workers = proxiesAll.shape[0]
    sessions = createSessions(proxiesAll, n = workers)
    sessions = list(sessions.values())
    indList = list(range(workers))
    
    page = 1
    url = 'https://www.tmz.com/?page=1'
    data, nextPage = getPage(url, sessions[indList[0]])
    allNew = 1
    indList.pop(0)
    nLeft = len(indList)
    if nLeft == 0:
        indList = list(range(workers))
    page += 1
    while allNew == 1:
        print(page)
        d, nextPage = getPage(nextPage, sessions[indList[0]])
        if (d['url'].isin(data['url'])).sum() == 1:
            allNew == 0
        d = d[~d['url'].isin(data['url'])]
        if d.shape[0] == 0:
            allNew == 0
        data = pd.concat((data,d))
        indList.pop(0)
        nLeft = len(indList)
        if nLeft == 0:
            indList = list(range(workers))
        page += 1
    
    data.drop_duplicates('url', keep = 'first', inplace = True)
    saveSubmission(data, conn, f'{platform}_urls', try_number=1, if_exists = 'append')
    
if __name__ == '__main__':
    
    main()
