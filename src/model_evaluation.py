import numpy as np
import pandas as pd

import pickle
import json

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.metrics import precision_score, recall_score, roc_auc_score


test_df=pd.read_csv("./data/features/test_final.csv")

X_test=test_df.drop(['is_helpful'],axis=1)
y_test=test_df['is_helpful']



pipe=pickle.load(open('model.pkl','rb'))



y_pred_proba = pipe.predict_proba(X_test)[:, 1]
y_pred=pipe.predict(X_test)

print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))





# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

metrics_dict={
    'accuracy':accuracy,
    'precision':precision,
    'recall':recall,
    'auc':auc
}

with open('metrics.json', 'w') as file:
    json.dump(metrics_dict, file, indent=4)