import pickle
import numpy as np
from readsnp import readdata
from wilcoxonsnp import gen_snp
from chi_square_split import split_snp_chi2
from decisiontree import Classifer
from sklearn.utils import resample
from sklearn.model_selection import KFold
from sklearn import svm
from sklearn.metrics import accuracy_score


with open("snp_data.p", "rb") as f:
    X, y, n_snp = pickle.load(f)

# with open("snp_info.p", "rb") as f:
#     SNP, snp_info = pickle.load(f)

# print (len(y))
# print (n_snp)
# print (len (SNP))

# weak, strong = split_snp_chi2(X, y, np.array(SNP)-1, 0.05)
# print (len(weak), len(strong))
# with open("temp_chi2.p", 'wb') as f:
#     pickle.dump((weak, strong), f)
y = np.array(y)
y = y.astype(int) 

with open("temp_chi2.p",'rb') as f:
    weak, strong = pickle.load(f)
    ntree = 200

    # kf = KFold(n_splits=5)
    # ntree = 200
    # mtry = 50
    # mtryw = mtry*len(weak)//(len(weak)+len(strong))
    # mtrys = mtry*len(strong)//(len(weak)+len(strong))

    # for train_index, test_index in kf.split(X):
    #     X_train, X_test = X[train_index], X[test_index]
    #     y_train, y_test = y[train_index], y[test_index]
    #     y_pred_essemble = []
    #     for i in range(ntree):
    #         X_i, y_i = resample(X_train, y_train, random_state=i)
    #         clf = DecisionTreeClassifier()
    #         # select SNPs
    #         weak_idx = np.random.choice(weak, mtryw)
    #         strong_idx = np.random.choice(strong, mtrys)
    #         idx = np.concatenate([weak_idx, strong_idx])

    #         #prediction
    #         clf.fit(X_i[:,idx], y_i)
    #         y_pred_essemble.append(clf.predict(X_test[:,idx]))

    #     y_pred_essemble = np.array(y_pred_essemble)
    #     y_pred = np.around(np.mean(y_pred_essemble, axis=0))
    #     print (accuracy_score(y_test, y_pred))
    
    kf = KFold(n_splits=5, random_state=0)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # mtry = 50
        # idx = np.concatenate((strong, weak))
        # idx = np.random.choice(idx, mtry)
     
        # clf = svm.SVC(decision_function_shape='ovo')
        # clf.fit(X_train[:,SNP], y_train)
        # y_pred = clf.predict(X_test[:,SNP])
        # print (accuracy_score(y_test, y_pred))
        print ("----")

        y_pred_essemble = []
        for i in range(ntree):
            X_i, y_i = resample(X_train, y_train, random_state=i)
            clf = Classifer(strong, weak, 200, 5)
            #prediction
            clf.fit(X_i, y_i)
            y_pred_essemble.append(clf.predict(X_test))

        y_pred_essemble = np.array(y_pred_essemble)
        y_pred = np.around(np.mean(y_pred_essemble, axis=0))
        print (accuracy_score(y_test, y_pred))
        print ("---------------------------")