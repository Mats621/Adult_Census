#Imports
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier




#Funktionen zum Laden des Datensatzes, verbunden mit allgemeinen Bearbeitungen (Entfernen mancher Spalten, drop duplicates...)
#____________________________________________________________________________________________________
#Entfernen beliebiger Spalten (wenn notwendig) und doppelte Eintraege, LabelEncoder der TargetSpalte
def drop_cols(dataframe, drop_col):
    dataframe = dataframe.drop(drop_col, axis=1)
    dataframe = dataframe.drop_duplicates()
    label_encoder = LabelEncoder()
    dataframe[dataframe.columns[-1]] = label_encoder.fit_transform(dataframe[dataframe.columns[-1]])
    return dataframe

#Erstellen einer Ladefunktion für die Datensätze
def load_file(filename):
    #Feature Names zum Einlesen
    feature_names = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'Target']
    #Einlesen
    df = pd.read_csv(filename, names=feature_names, na_values='?', sep=', ', engine='python')
    #Preprocessing
    df = drop_cols(df, ['fnlwgt', 'education'])
    X = df.drop('Target', axis=1)
    y = df['Target']
    return X, y




#Modelle anlegen die es zu trainieren gilt
#____________________________________________________________________________________________________
# define models to test
def get_models():
    models, names = list(), list()
	# DT
    models.append(DecisionTreeClassifier(max_depth=8))
    names.append('DT')
	# RF
    models.append(RandomForestClassifier(n_estimators=100))
    names.append('RF')
    #SVM
    models.append(SVC())
    names.append('SVM')
	# KNN
    models.append(KNeighborsClassifier(n_neighbors=50))
    names.append('KNN')
	# LR
    models.append(LogisticRegression(max_iter=500))
    names.append('LR')
    return models, names




#Evaluationsfunktion der Modelle
#____________________________________________________________________________________________________
def evaluate_model(X, y, model):
	# define evaluation procedure
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
	# evaluate model
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    return scores