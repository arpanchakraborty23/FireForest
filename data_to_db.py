import pandas as pd 
from pymongo import MongoClient
import json

url='mongodb+srv://www588650:arpan@cluster1.kdt4bq1.mongodb.net/'
client=MongoClient(url)
db=client['CSV']
collection=db['Algarian_Fire_Forest']

df=pd.read_csv('experiment/data/Algerian_forest_fires_dataset.csv')
data=df.to_dict(orient='records')
collection.insert_many(data)
print(df.head())