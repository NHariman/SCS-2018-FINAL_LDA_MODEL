# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 15:43:48 2018

@author: NHari
"""
"""!!!!!!!!!DON'T FORGET TO CHANGE THE DATE!!!!!!!!!!!!"""
#%% Import NYTimesAPI
date = "20180626"
prev_date = "20180606"
import time
import pickle
import os
from NYTimesArticleAPI import * #import package
api = articleAPI("API_KEY") #Article API from the NY times
last_pub_date = 19770130 #start date of last run
search_term = "cancer"
#%%
"""load data from yesterday"""
#%% 
with open((prev_date + '_'+ search_term + '_raw_data.py'), 'rb') as fp:
    complete_articles_collection = pickle.load(fp)
with open((prev_date +'_'+ search_term + '_processed_data.py'), 'rb') as fp:
    complete_collection = pickle.load(fp)
#articles_collection = np.load((prev_date + '- raw_data.npy')).item()
#collection = np.load((prev_date + '- processed_data.npy')).item()
#%%
"""only run the VERY FIRST TIME""""
#%%
complete_articles_collection = []
complete_collection = []
#%%
"""final code (19/04/2018), DESIRED AMOUNT OF DOCUMENTS: 50 000

TROUBLESHOOTING
    * IF YOU FORGET WHAT THE LAST END DATE IS, USE complete_articles_collection[i]["pub_date"] WHERE i IS THE LAST DIR IN THE ARRAY
    
    """
#%% Create directory
if not os.path.exists('PATH'+ search_term + '_raw_data/'):
    os.makedirs('PATH'+ search_term + '_raw_data/')
if not os.path.exists('PATH'+ search_term + '_processed_data/'):
    os.makedirs('PATH'+ search_term + '_processed_data/')
#%%
start_date = int(last_pub_date) + 1 #if first run of the day, check where you left off and enter that value manually! 
print("search term: " + search_term)
print("start date: " + str(start_date))
for j in range(100): #put at 100 for 100 pages per starting date
    articles = api.search(q= search_term,
                      fq= {"document_type": ["article", "News"],
                           # "body": ["cancer", "tumor"],
                           # "headline": ["cancer", "tumor"] ,
                           "type_of_material": ["article","News"]
                           },
                      begin_date = start_date, #format: YYYYMMDD replace with last recorded date
                      sort="oldest",
                      fl=["headline","abstract","lead_paragraph", "snippet", "type_of_material", "document_type", "pub_date"],
                      #page=j
                      )
    #print(articles["response"]["meta"]["hits"]) #prints amount of hits obtained through this search term     
    if j is 0: #create a dictionary if unavailable
        articles_collection = []
        articles_collection = {"raw_articles": articles["response"]["docs"]} #saves raw docs
        collection = {}
        collection = {'articles': [articles["response"]["docs"][0]["headline"]["main"] + " - " + articles["response"]["docs"][0]["snippet"]]}
        for i in range(9): #adds new articles to existing list
            collection["articles"].append(articles["response"]["docs"][i+1]["headline"]["main"] + ' - ' + articles["response"]["docs"][i+1]["snippet"])
        print(j)
        time.sleep(1)
    else:
        articles_collection["raw_articles"].extend(articles["response"]["docs"]) #saves raw docs
        for i in range(10): #adds new articles to existing list
            collection["articles"].append(articles["response"]["docs"][i]["headline"]["main"] + ' - ' + articles["response"]["docs"][i]["snippet"])       
        print(j)
        time.sleep(1) 

#"""if there's an error be sure to run this part manually!!"""
print(articles["response"]["docs"][i]["pub_date"]) #prints the last recorded date saved to replace the other date for the next run
last_pub_date = (articles["response"]["docs"][i]["pub_date"][0] + articles["response"]["docs"][i]["pub_date"][1] + articles["response"]["docs"][i]["pub_date"][2] + articles["response"]["docs"][i]["pub_date"][3] + articles["response"]["docs"][i]["pub_date"][5] + articles["response"]["docs"][i]["pub_date"][6] + articles["response"]["docs"][i]["pub_date"][8] + articles["response"]["docs"][i]["pub_date"][9])

#saves to mega list
complete_articles_collection.extend(articles_collection["raw_articles"])
complete_collection.extend(collection["articles"])
last_pub_date = (articles["response"]["docs"][i]["pub_date"][0] + articles["response"]["docs"][i]["pub_date"][1] + articles["response"]["docs"][i]["pub_date"][2] + articles["response"]["docs"][i]["pub_date"][3] + articles["response"]["docs"][i]["pub_date"][5] + articles["response"]["docs"][i]["pub_date"][6] + articles["response"]["docs"][i]["pub_date"][8] + articles["response"]["docs"][i]["pub_date"][9])

print(last_pub_date)

with open(('C:/Users/NHari/Documents/MSc/Intership/Pytho/data_new/'+ search_term + '_raw_data/' + '19700101' +'_to_' + str(last_pub_date) + '_'+ search_term + '_raw.py'), 'wb') as fp:
    pickle.dump(complete_articles_collection, fp)
with open(('C:/Users/NHari/Documents/MSc/Intership/Pytho/data_new/'+ search_term +'_processed_data/' + '19700101' + '_to_' + str(last_pub_date) + '_'+ search_term + '_processed.py'), 'wb') as fp:
    pickle.dump(complete_collection, fp)

#%%
""" save data using numpy
DON'T FORGET TO SAVE EACH VERSION AS A NEW ONE!!
#save:
np.save('my_file.npy', dictionary)

#load:
read_dictionary = np.load('my_file.npy').item()

"""
with open(('C:/Users/NHari/Documents/MSc/Intership/Pytho/data_new/' + date + '_'+ search_term +'_raw_data.py'), 'wb') as fp:
    pickle.dump(complete_articles_collection, fp)
    
with open(('C:/Users/NHari/Documents/MSc/Intership/Pytho/data_new/'+ date + "_"+ search_term + "_processed_data.py"), 'wb') as fp:
    pickle.dump(complete_collection, fp)

#np.save((date + '- raw_data.npy'), complete_articles_collection)
#np.save((date + '- processed_data.npy'), complete_collection)
