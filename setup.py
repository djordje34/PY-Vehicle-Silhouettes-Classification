from os import listdir
from os.path import isfile, join

import kerastuner as kt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import regularizers
from keras.layers import Dense, Dropout, Input
from keras.models import Sequential
from tensorflow import keras

from utils import Plotter, Preprocessor

PATH = "data/"
COMP = "csv/"

MODELS = []

def getData(*args:any)->pd.DataFrame:
    """
    Args:
        any
            Represents any of the possible parameters that might be added in the future development of this project
        
    Does:
        Data preprocessing function
        
        Fill zeroes check for NaN test if data evenly divided between classes
    
    Returns:
        csv data
"""
    datas = [f for f in listdir(PATH) if isfile(join(PATH, f))]
    [Preprocessor.cleaner(f"{PATH+fold}") for fold in datas]
    
    
    X_train,y_train,X_val,y_val,X_test,y_test,preonehot,df = Preprocessor.aggregate("csv")
    
    Plotter.correlation(df)
    Plotter.pair(preonehot)
    Plotter.count(preonehot)
    
    return X_train,y_train,X_val,y_val,X_test,y_test,preonehot,df
    
def getModel(hp:kt.Hyperband)->Sequential:
    
    """
    Args: 
        hp
    Does:
        Sequential ANN classification w/ relu's for activation in hidden layer
        
        7 train; 1.5 validation; 1.5 test
        
        Classic 4 node output softmax with categorical cross-entropy
    
    Returns:
        Keras sequential model
    """
    
    model = Sequential()
    model.add(Input(shape=(18,)))
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-1,1e-2, 1e-3, 1e-4])
    
    for i in range(1, hp.Int("num_layers", 1, 4)):
            model.add(
            Dense(
                units=hp.Int("units_" + str(i), min_value=2, max_value=192, step=1),
                activation="relu", kernel_regularizer=regularizers.l2(0.01))
            )
            #model.add(Dropout(hp.Choice(name='dropout'+str(i),values=[0.1,0.2,0.25,0.3])))
            
    model.add(Dense(4, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                metrics=['accuracy'])
    
    return model


def Setup():
    """Function with complete integration of logic from utility and other classes
    """
    getData()
    X_train,y_train,X_val,y_val,X_test,y_test,_,_ = getData()
    #early_stopping = EarlyStopping()
    
    tuner = kt.Hyperband(getModel,
                     objective='val_accuracy',max_epochs=100,factor=3, directory='dir', project_name='hiperparam_modeli')
    
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    tuner.search(X_train, y_train, epochs=100,validation_data=(X_val, y_val), callbacks=[stop_early])
    all_hps = tuner.get_best_hyperparameters(num_trials=5)
    best_hp=all_hps[0]
    
    h_model = tuner.hypermodel.build(best_hp)
    h_model.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val))
    
    h_model.summary()
    h_eval_dict = h_model.evaluate(X_test, y_test, return_dict=True)
    print(h_eval_dict["accuracy"])
    scores = pd.DataFrame(best_hp.values,index=[0])
    scores['score'] = Preprocessor.readJSON(best_hp.values[r"tuner/trial_id"])
    for hp in all_hps[1:]:
        if "tuner/trial_id" in hp.values:
            hp.values['score'] = Preprocessor.readJSON(hp.values[r"tuner/trial_id"])
            scores=scores.append(hp.values,ignore_index=True)
            
    scores=scores[["learning_rate", "num_layers", "tuner/trial_id", "score"]]
    scores.to_csv("csv/model_scores.csv",index=None)
    
    pred = h_model.predict(X_test)
    pd.DataFrame(pred).to_csv("csv/predicted.csv",index=None)
    y_pred=np.argmax(pred, axis=1)
    y_test=np.argmax(y_test.to_numpy(), axis=1)
    Plotter.confusion_matrix(y_pred,y_test)
    h_model.save("model")
    
    Plotter.architecture(h_model)
    Plotter.graphicalmodel(h_model)


def main():
    Setup()

if __name__=='__main__':
    main()