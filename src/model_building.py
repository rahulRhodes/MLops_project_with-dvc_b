#by the randomforestclassifier
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.pipeline import Pipeline # Use imblearn's Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import pandas as pd
import pickle


train_df=pd.read_csv("./data/features/train_final.csv")

X_train=train_df.drop(['is_helpful'],axis=1)
y_train=train_df['is_helpful']





# Removed: sm = SMOTE()
# Removed: X_train_res, y_train_res = sm.fit_resample(X_train, y_train)


ColumnTransition=ColumnTransformer([
    ("cat",TfidfVectorizer(),'review_text'), # Simplified TfidfVectorizer
    ("num","passthrough",['rating'])
])


pipe=Pipeline([
    ("preprocess",ColumnTransition),
    ("sampling", SMOTE(random_state=42)), # Add SMOTE step after preprocessing
    ("model",RandomForestClassifier(random_state=42))
])

pipe.fit(X_train,y_train)


pickle.dump(pipe, open('model.pkl','wb'))
