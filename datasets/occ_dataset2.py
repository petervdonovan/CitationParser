import sparql
import requests
import time
import random
import re
import pandas as pd
import pickle

path = input('raw_dois dictionary file path: ')

with open(path, 'rb') as dbfile:
    raw_dois = pickle.load(dbfile)

OC_API = 'https://w3id.org/oc/index/api/v1'
data = {'author': [],
    'year': [],
    'title': [],
    'page': [],
    'volume': [],
    'source_title': [],
    'issue': []}
def failed_get_meta():
    print('WARNING: Could not interpret response.')
    for ls in data.values():
        ls.append(None)
def save_meta(meta):
    for key, ls in data.items():
        ls.append(meta[key])
tstart = time.time()
for i, doi in enumerate(raw_dois['dois']):
    t0 = time.time()
    meta = None
    wait = 300
    backoff = 5
    while meta is None:
        try:
            meta = requests.get(
                    OC_API + '/metadata/{}'.format(doi)
                )
        except ConnectionError as e:
            print('Bad connection. Retrying in {wait} seconds...')
            time.sleep(wait)
            wait *= backoff
    try:
        meta = meta.json()[0]
    except (ValueError, IndexError) as e:
        failed_get_meta()
        continue
    if i % 50 == 0:
        total_elapsed = time.time() - tstart
        proportion_complete = (i+1) / len(raw_dois['dois'])
        remaining_time = total_elapsed \
                * (1-proportion_complete) / proportion_complete
        print('{:.2f}% complete after {:.2f} hours. {:.2f} hours'
              ' remaining.'.format(
                  proportion_complete * 100,
                  total_elapsed / 3600,
                  remaining_time / 3600))
    save_meta(meta)

data['raw'] = raw_dois['raw_text']
occ = pd.DataFrame(data=data)

with open('{}_with_metadata.pickle'.format(path), 'wb') as dbfile:
    pickle.dump(occ, dbfile, pickle.HIGHEST_PROTOCOL)