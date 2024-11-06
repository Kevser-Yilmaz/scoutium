
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)

pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

df_sa = pd.read_csv("scoutium_attributes.csv", sep=";")
df_sa.head()
df_spl = pd.read_csv("scoutium_potential_labels.csv", sep=";")
df_spl.head()
df = pd.merge(df_sa, df_spl, on = ["task_response_id", 'match_id', 'evaluator_id', "player_id"])

df = df[df["position_id"] != 1]
df = df[df["potential_label"] !="below_average"]

pivot_table = df.pivot_table(index=["player_id", "position_id", "potential_label"],
                             columns="attribute_id", values="attribute_value")
pivot_table.head()

pivot_table = pivot_table.reset_index(drop=False)

pivot_table.columns = pivot_table.columns.map(str)
pivot_table.head()

pivot_table.dtypes

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


labelEncoderCols = ["potential_label"]

for col in labelEncoderCols:
    pivot_table = label_encoder(pivot_table, col)

pivot_table.head()

num_cols = [col for col in pivot_table.columns if col not in ["player_id", "position_id", "potential_label"]]

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
pivot_table[num_cols] = scaler.fit_transform(pivot_table[num_cols])
pivot_table.head()

y = pivot_table["potential_label"]
X = pivot_table.drop(["potential_label", "player_id"], axis=1)

X.shape
y.value_counts()



def base_models(X, y, scoring="roc_auc"):
    print("Base Models....")
    classifiers = [('LR', LogisticRegression()),
                   ('KNN', KNeighborsClassifier()),
                   ("SVC", SVC()),
                   ("CART", DecisionTreeClassifier()),
                   ("RF", RandomForestClassifier()),
                   ('Adaboost', AdaBoostClassifier()),
                   ('GBM', GradientBoostingClassifier()),
                   ('XGBoost', XGBClassifier(eval_metric='logloss')),
                   ('LightGBM', LGBMClassifier(verbose=-1) ),
                   #('CatBoost', CatBoostClassifier(verbose=False))
                   ]

    for name, classifier in classifiers:
        cv_results = cross_validate(classifier, X, y, cv=3, scoring=scoring)
        print(f"{scoring}: {round(cv_results['test_score'].mean(), 4)} ({name}) ")

base_models(X, y, scoring="accuracy")
base_models(X, y, scoring="roc_auc")
base_models(X, y, scoring="f1")

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')


model =  RandomForestClassifier()
model.fit(X, y)

plot_importance(model, X)

import shap

explainer = shap.Explainer(model)
shap_values = explainer(X)


print("shap_values shape:", shap_values.shape)
print("X shape:", X.shape)

shap_values = shap_values[:, :, 0]

shap.summary_plot(shap_values, X)
shap.plots.waterfall(shap_values[0])
shap.plots.scatter(shap_values[:, "4332"], color=shap_values)
shap.plots.beeswarm(shap_values)
shap.plots.bar(shap_values)

