import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier

def feature_importance(df, candidate_features = [], target = [],top = 25):

  target = df.columns[-1]
  candidate_features = list(df.columns[:-1])
  X = df[candidate_features] 
  Y = df[target]

  xgb = XGBClassifier(n_estimators=100)
  xgb.fit(X, Y)
  feature_imp = pd.DataFrame(sorted(zip(xgb.feature_importances_,candidate_features)), columns=['Value','Feature'])
  plt.figure(figsize=(20, 10))
  sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))
  plt.title('Feature Importance')
  plt.tight_layout()
  plt.show()
  plt.show()
  aux_df = pd.DataFrame(feature_imp.sort_values(by="Value", ascending=False))
  selected_features = aux_df["Feature"][:top].values

  return selected_features