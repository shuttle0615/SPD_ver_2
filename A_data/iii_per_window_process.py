from __init__ import *

# before we start, we need to drop N/A values
# data need to be seperated for train, test, validation. 
# check how they did evaluation part

def per_window_process(df, name):
    
    if os.path.exists(processed_data / (name + '_processed.csv')):
        return pd.read_csv(processed_data / (name + '_processed.csv'))
    
    pass