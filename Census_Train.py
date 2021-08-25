#Imports
#________________________________________________________________________________________________________________
import numpy as np
import matplotlib.pyplot as plt

from Census_Functions import load_file, get_models, evaluate_model

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

import joblib




#Daten einlesen
#________________________________________________________________________________________________________________
train_file = 'Datasets/adult.data'
test_file = 'Datasets/adult.test'

X_train, y_train = load_file(train_file)
X_test, y_test = load_file(test_file)




#Preprocessing
#________________________________________________________________________________________________________________
#Preprocessing unterteilen je nach Spaltentyp

#Numerische Spalten: imputen, scalen
numeric_features = list(X_train.select_dtypes(include=['int64', 'float64']).columns)
numeric_transformer = Pipeline(steps=[
                              ('imputer', SimpleImputer(missing_values=np.nan, strategy='median')),
                              ('scaler', MinMaxScaler())])

#Kategorische Spalten: imputen, one hot encoden
categorical_features = list(X_train.select_dtypes(include=['bool', 'object']).columns)
categorical_transformer = Pipeline(steps=[
                                  ('imputer', SimpleImputer(missing_values=np.nan, strategy='most_frequent')),
                                  ('encoder', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(transformers=[
                                    ('num', numeric_transformer, numeric_features),
                                    ('cat', categorical_transformer, categorical_features)])




#Modelle trainieren
#________________________________________________________________________________________________________________
#Start training
models, names = get_models()
results = []
trained_model = []
for i in range(len(models)):
    pipeline = Pipeline(steps=[
                    ('preprocessor', preprocessor),
                    ('classifier', models[i])])
    pipeline.fit(X_train, y_train)
    trained_model.append((names[i], pipeline))
    #Evaluation und Speicherung der Ergebnisse
    scores = evaluate_model(X_test, y_test, pipeline)
    results.append(scores)
    print('>%s %.3f (%.3f)' % (names[i], np.mean(scores), np.std(scores)))




#Modelle speichern
#________________________________________________________________________________________________________________
for i in range(len(trained_model)):
    filename = 'Trained_Models/'+trained_model[i][0]+'_Adult_Census.pkl'
    joblib.dump(trained_model[i][1], filename)



#Modelle auswerten
#________________________________________________________________________________________________________________
plt.figure(figsize=(10,8))
plt.boxplot(results, labels=names, showmeans=True)
plt.grid()
plt.show()