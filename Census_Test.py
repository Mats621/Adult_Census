#Imports
#________________________________________________________________________________________________________________
from Census_Functions import load_file
import joblib




#Daten einlesen
#________________________________________________________________________________________________________________

filename = 'Datasets/test_person_1.test'
X_test, y_test = load_file(filename)




#Modelle laden
#________________________________________________________________________________________________________________
#DecisionTree
dt = joblib.load('Trained_Models/DT_Adult_Census.pkl')
#RandomForest
rf = joblib.load('Trained_Models/RF_Adult_Census.pkl')
#SupportVectorMachines
svm = joblib.load('Trained_Models/SVM_Adult_Census.pkl')
#KNearestNeighbours
knn = joblib.load('Trained_Models/KNN_Adult_Census.pkl')
#LogisticRegression
lr = joblib.load('Trained_Models/LR_Adult_Census.pkl')

#Modelle in Liste schreiben
models = [dt, rf, svm, knn, lr]
names = ['DT', 'RF', 'SVM', 'KNN', 'LR']




#Predictions ausgeben lassen
#________________________________________________________________________________________________________________
for i in range(len(models)):
    prediction = models[i].predict(X_test)
    if prediction == 0:
        prediction = '<=50K'
    else:
        prediction = '>50K'
    print('The income will be '+prediction+ '$, based on the '+ names[i]+'-model.')
