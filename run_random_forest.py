from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix, precision_score, recall_score, r2_score, f1_score
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
import sklearn.feature_selection

import pandas as pd
import numpy as np
import time
from sys import argv



def main(csv, gs=0, name='rf'):

    df = pd.read_csv(csv)
    
    X_train, X_test, y_train, y_test, dfc = make_training_set(df)
    if gs == 0:
        gb = False
        rf = make_rf(X_train, y_train)
    else:
        gb=True
        rf = grid_search(X_train, y_train)

    #y_pred, acc, precision, recall = rf_metrics(dfc, X_test, y_test)

    top_feats =  print_feat_importance(rf, dfc)
    #return top_feats

    log_result(top_feats, y_train, X_test, y_test, rf, dfc, name, csv, gb)
    return rf

    
def log_result(top_feats, y_train, X_test, y_test, rf, dfc, name, csv, gb):

    y_pred, acc, precision, recall = rf_metrics(rf, X_test, y_test)
    
    fname = '../model_logs/'+str(time.ctime())+'_'+name+'.txt'
        #print fname
    
    with open(fname, 'a+') as f:

        f.write('model used: {0} \ngradient boosted= {6} \nacc= {1} \nprecision= {2} \nrecall= {3} \ncsv used: \
 {4} \nimportant features:\n {5}'.format(name, acc, precision, recall, csv, top_feats, gb))
    

def make_training_set(df):
    #print 'A clear'
    dfc = df.copy()
    #print dfc
    dfc = dfc.dropna()
    #print dfc
    #df = df.fillna(value=something, inplace=True)
    print 'HELLLLLLLLOOOOO'

    y = dfc[['ball','call_strike', 'contact', 'swg_strike']].values

    dfc.drop(['ball','call_strike', 'contact', 'swg_strike'], axis=1, inplace=True)
    
    X = dfc.values
    print y.shape

    X_train, X_test, y_train, y_test = train_test_split(X,y)
    return X_train, X_test, y_train, y_test, dfc

    


def make_rf(X_train, y_train, n_jobs=-1):
    
    rf = RandomForestClassifier(n_jobs=-1, min_samples_split=1, min_samples_leaf=2, n_estimators=60, random_state=1, bootstrap=True)
                       
    rf.fit(X_train, y_train) #train your model with training set...
    return rf


def rf_metrics(model, X_test, y_test):
    
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    #mse = mean_squared_error(y_test, y_pred)
    #confusion = confusion_matrix(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    #r2 = r2_score(y_test, y_pred)

    #return y_pred, acc, mse, confusion, precision, recall
    return y_pred, acc, precision, recall


def print_feat_importance(model, dfcopy):
    print 'HEEEELLLLLLOOOOOOO'
    '''
    feat_importance = model.feature_importances_
    most_important = np.argsort(feat_importance)[::-1] #how to reverse order
    imp_feat_list =  zip(np.sort(feat_importance)[::-1], dfcopy.columns[most_important])
    '''
    lab_feats = sorted(zip(dfcopy.columns, model.feature_importances_), key=lambda x : x[1])[::-1]
        
    total,cnt = 0,0
    for n,v in lab_feats:
        total+=v
        if total<=.95:
            cnt+=1
            #print cnt,n,v

    return lab_feats[:cnt]

    '''
    print
    for a, b in imp_feat_list:
        print b, a
    return feat_importance
    
    plt.clf()
    plt.bar(np.arange(10), np.sort((feat_importance)[::1]))
    plt.xticks(np.arange(10), dfcopy.columns[most_important].tolist(), rotation=90)
    plt.savefig('hist.jpg',) 
    '''

def grid_search(X_train, y_train):
    
    random_forest_grid = {'max_depth': [None],
                        'max_features': [None],
                        'min_samples_split': [1],
                        'min_samples_leaf': [10],
                        'bootstrap': [True],
                        'n_estimators': [70],
                        'random_state': [1]}

    rf_gridsearch = GridSearchCV(RandomForestClassifier(),
                                   random_forest_grid,
                                   n_jobs=-1,
                                   verbose=True,
                                   scoring='accuracy')
    
    rf_gridsearch.fit(X_train, y_train)

    print "best parameters:", rf_gridsearch.best_params_

    best_rf_model = rf_gridsearch.best_estimator_

    return best_rf_model


if __name__ == '__main__':
    
    script, csv = argv
    rf = main(csv,1)
