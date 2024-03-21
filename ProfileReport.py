import pandas as pd 
from pandas_profiling import ProfileReport

df=pd.read_csv('/workspace/FireForest/experiment/data/Fire_Forest_Clean.csv')

profile=ProfileReport(df,explorative=True)

profile.to_file("your_report.html")