from A_data import simple_labeling
from A_data.i_download import download
    
df, name = download("BTC/USDT", '1h', (2022, 1, 10, 1), (2022, 7, 12, 2))  
   
new_df, new_name = simple_labeling(df, name, 8, 2, 0.01)

print(new_df.head())
new_df.info()
print(len(new_df))
