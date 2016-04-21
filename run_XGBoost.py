import xgboost as xgb
import pandas as pd
import numpy as np
import time
from sklearn.metrics import accuracy_score, mean_squared_error, \
    confusion_matrix, precision_score, recall_score, r2_score, f1_score
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import LabelEncoder
import sklearn.feature_selection
from sys import argv


def main(csv, gs=0, name='xgboost'):
    print 'HEEELLLLOOOO'
    df = pd.read_csv(csv)
    print df.columns
    print df.shape

    X_train, X_test, y_train, y_test, dfc = make_training_set(df)
    
    if gs == 0:
        boost = make_xgboost(X_train, y_train)
    else:
        boost = grid_search(X_train, y_train)

    #y_pred, acc, precision, recall = rf_metrics(dfc, X_test, y_test)

    #top_feats =  print_feat_importance(boost, dfc)
    #return top_feats

    log_result(y_train, X_test, y_test, boost, dfc, name, csv)

    return boost


    
def log_result( y_train, X_test, y_test, model, dfc, name, csv):

    y_pred, acc, precision, recall = metrics(model, X_test, y_test)
    
    fname = '../model_logs/'+str(time.ctime())+'_'+name+'.txt'
        #print fname
    
    with open(fname, 'a+') as f:

        f.write('model used: {0} \nacc= {1} \nprecision= {2} \nrecall= {3} \ncsv used: \
 {4} \nimportant features:\n {5}'.format(name, acc, precision, recall, csv, 'not included'))

        
def make_training_set(df):

    dfc = df.copy()
    
    dfc.dropna(inplace=True)
    print 'shape: ',dfc.shape

    y = dfc[['ball','call_strike', 'contact', 'swg_strike']].values

    #print 'y shape: ',y.shape

    dfc.drop(['ball','call_strike', 'contact', 'swg_strike'], axis=1, inplace=True)
    
    X = dfc.values
    print 'X.shape',X.shape

    '''
    label_map = {0: 'ball', 1: 'call_strike', 2: 'contact', 3: 'swg_strike'}

    flattened_Y = map(lambda x: label_map[x], np.where(y == 1)[1])

    encoder = LabelEncoder()
    encoder.fit(flattened_Y)
    y = encoder.transform(flattened_Y)
    '''
    y = np.argmax(y, axis=1)    
    X_train, X_test, y_train, y_test = train_test_split(X,y)
    #X_train = xgb.DMatrix(X_train)
    #y_train = xgb.DMatrix(y_train)
    return X_train, X_test, y_train, y_test, dfc

#need to do inverse transform on whatever the predicition is


def make_xgboost(X_train, y_train, n_jobs=-1):

    #xg_train = xgb.DMatrix(X_train, label=y_train)
    #xg_test = xgb.DMatrix(X_test, label=y_test)
    # setup parameters for xgboost
    params = {}
    # use softmax multi-class classification
    
    params['objective'] = 'multi:softmax'
    # scale weight of positive examples
    #params['eta'] = 0.1
    #params['max_depth'] = 6
    #params['silent'] = 1
    #params['nthread'] = 4
    params['num_class'] = 4
    #watchlist = [ (xg_train,'train'), (xg_test, 'test') ]
  
    print "X_train shape:", X_train.shape
    print "y_train shape:", y_train.shape
    #print "x_test shape:", X_test.shape
    #print "y_test shape:", y_test.shape

    
    boost = xgb.XGBClassifier(objective='multi:softmax')
    #boost.set_params(params)
    #boost.train(param, X_train)
    #boost.objective[multi:softmax]
    boost.fit(X_train, y_train) #train your model with training set...
    return boost


def grid_search(X_train, y_train):
    
    xgb_params = {'max_depth': [4], 'n_estimators': [100], 'objective':['multi:softmax']}

    xgb_gridsearch = GridSearchCV(xgb.XGBClassifier(),
                                   xgb_params,
                                   n_jobs=-1,
                                   verbose=True,
                                   scoring='accuracy')
    
    xgb_gridsearch.fit(X_train, y_train)

    print "best parameters:", xgb_gridsearch.best_params_

    best_xgb_model = xgb_gridsearch.best_estimator_

    return best_xgb_model


def metrics(model, X_test, y_test):
    
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    #mse = mean_squared_error(y_test, y_pred)
    #confusion = confusion_matrix(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    #r2 = r2_score(y_test, y_pred)

    #return y_pred, acc, mse, confusion, precision, recall
    return y_pred, acc, precision, recall


if __name__ == '__main__':
    
    script, csv = argv
    xgb = main(csv,1)

