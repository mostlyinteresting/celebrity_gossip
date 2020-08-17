import os
from concurrent.futures import ProcessPoolExecutor as PoolExecutor        
import numpy as np
import pandas as pd
import sqlite3
import requests
import time
import random

conn = sqlite3.connect('celebrity_news.sqlite')
connTwitter = sqlite3.connect('twitter.sqlite')

handle = 'tmz'

def saveSubmission(data, dCONN, tableName, try_number=1, if_exists = 'append'):
    try:
        data.to_sql(tableName, dCONN, if_exists=if_exists, index=False)
    except:
        time.sleep(2**try_number + random.random()*0.01) #exponential backoff
        return saveSubmission(data, dCONN, tableName, try_number=try_number+1)
    else:
        return
    
def getProxies():
    #function calls api of proxy service and retrieves all proxies
    return proxiesAll

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
        
        s = requests.Session()
        s.proxies = proxies
        
        sessions[k] = s
        
    return sessions

def nav2url(x,s):
    #Navigate to full url
    try:
        r = s.get(x, allow_redirects=True, timeout = 5)
        u = r.url
    except:
        u = None
    return u

def process(k, data, limit, workers, sessions):
    
    data = data.loc[(k*limit):((k+1)*limit),:].reset_index(drop = True)
    sessions = sessions[(k*workers):((k+1)*workers):]
    
    limit = int(np.ceil(data.shape[0]/workers))
    
    for l in range(limit):
        
        for i in range(workers):
            
            session = sessions[i]
            ind = i+workers*l
            if ind < data.shape[0]:
                
                urlShort = data.loc[ind,'urls']
                
                try:
                    data.loc[ind,'urlFull'] = nav2url(urlShort,session)
                except:
                    pass
        if k == 0:
            print(str(l) + ' of ' + str(limit) )
    saveSubmission(data, conn, f'{handle}_tweets', try_number=1, if_exists = 'append')

def main():
    
    proxiesAll = getProxies()
    workers = proxiesAll.shape[0]
    sessions = createSessions(proxiesAll, n = workers)
    sessions = list(sessions.values())
    
    #news
    data = pd.read_sql_query(f"SELECT * FROM {handle}", connTwitter)
    data = data[ data['urls'] != '' ]
    data['date'] = pd.to_datetime(data['date'])
    data.sort_values('date', ascending = True, inplace = True)
    data.drop_duplicates('urls', keep = 'first', inplace = True)
    data.reset_index(drop = True, inplace = True)
    data['urlFull'] = None
    data = data[['id','urls','urlFull']]
    
    n = data.shape[0]
    executors = 5
    workers = int(workers/executors)
    limit = int(np.ceil(n/executors))
    
    with PoolExecutor(max_workers=executors) as executor:
        for k in range(executors):
            executor.map(process, [k], [data], [limit], [workers], [sessions])

if __name__ == '__main__':
    
    main()
