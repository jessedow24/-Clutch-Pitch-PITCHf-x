from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from sklearn.cross_validation import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score
from keras.utils import np_utils 
import pandas as pd
import numpy as np
import time
from sys import argv


def main(df, name='neural_net'):

    #df = pd.read_csv(csv)
    
    X_train, X_test, y_train, y_test, dfc = make_training_set(df)

    precision_score_train, precision_score_test, recall_score_train, recall_score_test, acc_train, acc_test, model = run_model(X_test, X_train, y_test, y_train, prob_threshold = 20, layers = 1, nodes = 64, dropout = 50)

    log_result(precision_score_train, precision_score_test, recall_score_train, recall_score_test, name, csv, acc_train, acc_test)

    return model 

def log_result(pstr, psts, rstr, rsts, name, csv, acc_train, acc_test):
    print "log_result RUNNING"
    #y_pred, acc, precision, recall = rf_metrics(rf, X_test, y_test)
    
    fname = '../model_logs/'+str(time.ctime())+'_'+name+'.txt'
        #print fname
    
    with open(fname, 'a+') as f:

        f.write('model used: {0} \nacc_train{6}, \nacc_test{7} \nprecision_score_train= {1} \nprecisio_score_testn= {2} \nrecall_score_train= {3} \nrecall_score_test= {4}\ncsv used: {5}'.format(name, pstr, psts, rstr, rsts, csv, acc_train, acc_test))

   

def get_model(X, dropout = 50, activation = 'relu', node_count = 64, init_func='uniform', layers = 5):
        print "get_model RUNNING"
	# Fit a sequential model with some given number of nodes. 
	model = Sequential()

	# Fit the first hidden layer manually, becuase we have to fit it 
	# with the x-shape by the node_count. 
	model.add(Dense(output_dim=node_count, input_dim=X.shape[1])) #input_dim=input_dim
	model.add(Activation(activation))
	model.add(Dropout(dropout / 100.0))

	# We can fit any additional layers like this, provided they 
	# have the same node_count (except the last one). 
	for layer in xrange(layers): 
		model.add(Dense(X.shape[1]))
		model.add(Activation(activation))
		#model.add(Dropout(dropout / 100.0))

	model.add(Dense(output_dim=4))
	model.add(Activation('softmax'))

	sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(loss='mse', optimizer=sgd)

	return model

def run_model(X_test, X_train, y_test, y_train, prob_threshold = 20, layers = 5, nodes = 64, dropout = 50):
    
    print "run_model RUNNING"
    # Grab the model 
    model = get_model(X_test, layers =layers, dropout = dropout)
    model.fit(X_train, y_train, nb_epoch=20, batch_size=16, verbose = 0)

    # Get the training and test predictions from our model fit. 
    train_predictions  = model.predict_proba(X_train)
    test_predictions = model.predict_proba(X_test)
    # Set these to either 0 or 1 based off the probability threshold we 
    # passed in (divide by 100 becuase we passed in intergers). 
    train_preds = (train_predictions) >= prob_threshold / 100.0
    test_preds = (test_predictions) >= prob_threshold / 100.0

    # Calculate the precision and recall. Only output until 
    precision_score_train = precision_score(y_train, train_preds)
    precision_score_test = precision_score(y_test, test_preds)
    acc_train = accuracy_score(y_train, train_preds)
    acc_test = accuracy_score(y_test, test_preds)

    recall_score_train = recall_score(y_train, train_preds)
    recall_score_test = recall_score(y_test, test_preds)

    return precision_score_train, precision_score_test, recall_score_train, recall_score_test, acc_train, acc_test, model


def make_training_set(df):

    dfc = df.copy()

    dfc = dfc.dropna()

    print 'make_training_set RUNNING'

    y = dfc[['ball','call_strike', 'contact', 'swg_strike']].values

    dfc.drop(['ball','call_strike', 'contact', 'swg_strike'], axis=1, inplace=True)
    
    X = dfc.values
    print 'X.shape = ',X.shape
    print 'Y.shape = ',y.shape

    X = normalize(X)

    
    X_train, X_test, y_train, y_test = train_test_split(X,y)
    return X_train, X_test, y_train, y_test, dfc



def normalize(X):
    return (X - X.mean(axis = 0)) / X.std(axis = 0, ddof = 1)



    
if __name__ == '__main__':

    script, csv = argv
    df = pd.read_csv(csv)
    nn = main(df, 'neural_net')
 
