import pandas as pd
import numpy as np
import os

df_train=pd.read_csv("./Data/processed/train.csv")
df_test=pd.read_csv("./Data/processed/train.csv")




def counts_word(row):
    row_count=len(row.split())
    return row_count



    






#train
df_train['is_helpful']=df_train.helpful.apply(lambda x:1 if x>0 else 0)
df_train['word_count']=df_train.review_text.apply(counts_word)
df_train['long_text_rating_prop']=df_train['rating']/df_train['word_count']





#test
df_test['is_helpful']=df_test.helpful.apply(lambda x:1 if x>0 else 0)
df_test['word_count']=df_test.review_text.apply(counts_word)
df_test['long_text_rating_prop']=df_test['rating']/df_test['word_count']



df_train=df_train.drop(['review_id','review_date','helpful'],axis=1)
df_test=df_test.drop(['review_id','review_date','helpful'],axis=1)





data_path=os.path.join("data","features")
os.makedirs(data_path,exist_ok=True)
df_train.to_csv(os.path.join(data_path,"train_final.csv"),index=False)
df_test.to_csv(os.path.join(data_path,"test_final.csv"),index=False)