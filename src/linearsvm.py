import os
import json
import numpy as np
from sklearn import svm, metrics
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_recall_fscore_support as score
from smart_open import smart_open

# ignore Future Warning
import warnings
import sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)


class LinearSVM:
    def __init__(self, name):
        self.name = name
        self.C = None
        self.model = None


    def train(self, X_train, y_train):
        if not self.C:
            print("-------------------WARNING-------------------")
            print("please tune the hyperparameters for SVM first")
            print("---------------------------------------------")
            self.tune(X_train, y_train) 
        
        self.model = svm.SVC(kernel='linear', C=self.C)
        self.model.fit(X_train, y_train)

    def test(self, X_test, y_test):
        if not self.model:
            print("-------------------WARNING-------------------")
            print("please train the SVM model first")
            return 

        y_pred = self.model.predict(X_test)
        print(metrics.classification_report(y_test, y_pred))
        print("\noverall accuracy: %.3f" % metrics.accuracy_score(y_test, y_pred))
        precision, recall, fscore, support = score(y_test, y_pred, average='macro')
        
        # save to file
        with smart_open(os.path.join('results', '%s.md' % self.name), 'w', encoding='utf-8') as output:
            output.write("C in linear SVM %.3f\n" % self.C)
            output.write("accuracy %.3f\n" % metrics.accuracy_score(y_test, y_pred))
            output.write("precision %.3f\n" % precision)
            output.write("recall %.3f\n" % recall)
            output.write("F1 score %.3f\n" % fscore)

    def tune(self, X_train, y_train):
        if self.C:
            print("-------------------WARNING-------------------")
            print("please proceed as the penalty has been tuned")
            return 
        
        C_range = 10. ** np.arange(-3, 3, step=1)
        grid = GridSearchCV(svm.SVC(), [{'C': C_range, 'kernel': ['linear']}], cv=5, n_jobs=-1, verbose=3)
        grid.fit(X_train, y_train)

        print("\nbest parameters set found on development set\n")
        print(grid.best_params_)

        self.C = grid.best_params_['C']
