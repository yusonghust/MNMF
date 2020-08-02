from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
import random
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn import svm
import logging

def Classifier(inputs,cfg,logger):
    X = [cfg.look_up[i] for i in cfg.node_list]
    Y = [cfg.node_label[i] for i in cfg.node_list]
    if cfg.multilabel:
        Y = MultiLabelBinarizer().fit_transform(Y)
    state = random.getstate()
    random.shuffle(X)
    random.setstate(state)
    random.shuffle(Y)
    for clf_ratio in cfg.clf_ratio:
        index = int(len(cfg.node_list)*clf_ratio)
        X_train = inputs[X[0:index]]
        Y_train = Y[0:index]
        X_test  = inputs[X[index:]]
        Y_test  = Y[index:]
        if cfg.multilabel==False:
            logger.info('multiclass classification, clf_ratio = {:.1f}'.format(clf_ratio))
            clf = LogisticRegression(random_state=0, solver='lbfgs',
                          multi_class='multinomial').fit(X_train,Y_train)
            Y_pred = clf.predict(X_test)
            logger.info('LogisticRegression Macro-f1 = {:.3f} Micro-f1 = {:.3f}'.format(f1_score(Y_test,Y_pred,average='macro'),f1_score(Y_test,Y_pred,average='micro')))
            clf = svm.LinearSVC(loss='hinge', tol=0.00001, C=1.0).fit(X_train,Y_train)
            Y_pred = clf.predict(X_test)
            logger.info('SVMClassifier C=1 Macro-f1 = {:.3f} Micro-f1 = {:.3f}'.format(f1_score(Y_test,Y_pred,average='macro'),f1_score(Y_test,Y_pred,average='micro')))
            clf = svm.LinearSVC(loss='hinge', tol=0.00001, C=0.5).fit(X_train,Y_train)
            Y_pred = clf.predict(X_test)
            logger.info('SVMClassifier C=0.5 Macro-f1 = {:.3f} Micro-f1 = {:.3f}'.format(f1_score(Y_test,Y_pred,average='macro'),f1_score(Y_test,Y_pred,average='micro')))
        else:
            logger.info('multilabel classification, clf_ratio = {:.1f}'.format(clf_ratio))
            clf = OneVsRestClassifier(LogisticRegression(random_state=0, solver='lbfgs',
                          multi_class='multinomial',max_iter=500),n_jobs=12).fit(X_train,Y_train)
            Y_pred = clf.predict(X_test)
            logger.info('LogisticRegression Macro-f1 = {:.3f} Micro-f1 = {:.3f}'.format(f1_score(Y_test,Y_pred,average='macro'),f1_score(Y_test,Y_pred,average='micro')))
            clf = OneVsRestClassifier(svm.LinearSVC(loss='hinge', tol=0.00001, C=1.0, max_iter=5000),n_jobs=12).fit(X_train,Y_train)
            Y_pred = clf.predict(X_test)
            logger.info('SVMClassifier C=1 Macro-f1 = {:.3f} Micro-f1 = {:.3f}'.format(f1_score(Y_test,Y_pred,average='macro'),f1_score(Y_test,Y_pred,average='micro')))
            clf = OneVsRestClassifier(svm.LinearSVC(loss='hinge', tol=0.00001, C=0.5, max_iter=5000),n_jobs=12).fit(X_train,Y_train)
            Y_pred = clf.predict(X_test)
            logger.info('SVMClassifier C=0.5 Macro-f1 = {:.3f} Micro-f1 = {:.3f}'.format(f1_score(Y_test,Y_pred,average='macro'),f1_score(Y_test,Y_pred,average='micro')))