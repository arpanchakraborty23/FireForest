import pandas as pd 
from pymongo import MongoClient
import json
from dotenv import load_dotenv
import os 

load_dotenv()
uri=os.getenv('url')
db=os.getenv('database')
collection=os.getenv('clean_data')


url=uri
client=MongoClient(url)
db=client[db]
collection=db[collection]

df=pd.read_csv('experiment/data/Fire_Forest_Clean.csv')
data=df.to_dict(orient='records')
collection.insert_many(data)
print(df.head())