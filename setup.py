from os import listdir
from os.path import isfile, join

import kerastuner as kt
import pandas as pd
import tensorflow as tf
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
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
    
    
    train,validate,test,df,preonehot = Preprocessor.aggregate("csv")
    
    Plotter.correlation(df)
    Plotter.pair(df)
    Plotter.count(preonehot)
    
    
def getModel(hp:kt.Hyperband)->Sequential:
    
    """
    Args: 
        hp
    Does:
        Sequential ANN classification w/ relu's for activation in hidden layer
        
        10 folds cross validation to use (probably)-> 7 train; 1.5 validation; 1.5 test yields optimal results
        
        Classic 4 node output softmax with categorical cross-entropy
    
    Returns:
        Keras sequential model
    """
    
    model = Sequential()
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    for i in range(1, hp.Int("num_layers", 2, 6)):
            model.add(
            keras.layers.Dense(
                units=hp.Int("units_" + str(i), min_value=8, max_value=72, step=4),
                activation="relu")
            )
            
    model.add(Dense(4, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                metrics=['accuracy'])
    
    return model


def Setup():
    """Function with complete integration of logic from utility and other classes
    """
    getData()
    data,valid,test = pd.read_csv(COMP+"train.csv"),pd.read_csv(COMP+"validation.csv"),pd.read_csv(COMP+"test.csv")
    #early_stopping = EarlyStopping()
    X_train = data.iloc[:,0:18].values
    y_train = data.iloc[:,18:].values
    X_val = valid.iloc[:,0:18].values
    y_val = valid.iloc[:,18:].values
    teX = test.iloc[:,0:18].values
    teY = test.iloc[:,18:].values
    
    tuner = kt.Hyperband(getModel,
                     objective='val_accuracy',max_epochs=100,factor=3, directory='dir', project_name='khyperband')
    
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    tuner.search(X_train, y_train, epochs=100,validation_data=(X_val, y_val), callbacks=[stop_early])
    all_hps = tuner.get_best_hyperparameters(num_trials=5)
    best_hp=all_hps[0]
    
    h_model = tuner.hypermodel.build(best_hp)
    h_model.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val))
    
    h_model.summary()
    h_eval_dict = h_model.evaluate(teX, teY, return_dict=True)
    print(h_eval_dict["accuracy"])
    scores = pd.DataFrame(best_hp.values,index=[0])
    scores['score'] = Preprocessor.readJSON(best_hp.values[r"tuner/trial_id"])
    for hp in all_hps[1:]:
        if "tuner/trial_id" in hp.values:
            hp.values['score'] = Preprocessor.readJSON(hp.values[r"tuner/trial_id"])
            scores=scores.append(hp.values,ignore_index=True)
            
    scores=scores[["learning_rate", "num_layers", "tuner/trial_id", "score"]]
    scores.to_csv("csv/model_scores.csv",index=None)
    
    
    h_model.save("model")
    
    Plotter.architecture(h_model)
    Plotter.graphicalmodel(h_model)


def main():
    Setup()

if __name__=='__main__':
    main()