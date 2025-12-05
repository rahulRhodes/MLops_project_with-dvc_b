import pandas as pd
import numpy as np
import os



def load_data(url:str)->pd.DataFrame:
    try:
        df=pd.read_csv(url)
        return df
    
    except pd.errors.ParserError as e:
        print(f"Error: Failed to parse the csv file from {url}")
        print(e)
        raise 
    
    except Exception as e:
        print(f"An unexcepted error occue while loading the error")
        print(e)
        raise
    

def basic_preprocess_load(df: pd.DataFrame)->pd.DataFrame:
    try:
        df=df.dropna()
        df=df.drop_duplicates()
        df['review_date']=pd.to_datetime(df['review_date'])

        return df
    except KeyError as e:
        print(f"Error: Missing column {e} in the dataframe.")
        raise

    except Exception as e:

        print(f"Error occurred :")
        print(e)
        raise


def save_data(cleaned_data:pd.DataFrame,data_path:str)->None:
    try:
        data_path=os.path.join(data_path,"cleaned")
        os.makedirs(data_path,exist_ok=True)
        
        cleaned_data.to_csv(os.path.join(data_path,"clean.csv"),index=False)


    except Exception as e:
        print(f"An error occurred while saving the data")
        print(e)
        raise

def main():
    try:
        df=load_data(url='data/raw/zomato_reviews.csv')
        df_main=basic_preprocess_load(df)
        save_data(df_main,'data')


    except Exception as e:
        print(f"Error: {e}")
        print("Failed to ingest the data")



if __name__=='__main__':
    main()

        

    


