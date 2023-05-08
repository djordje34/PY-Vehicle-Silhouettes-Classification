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
    
    
def getModel(hp)->tf.keras.Model:
    
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
def main():
    
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
    
    #model = getModel()
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    tuner.search(X_train, y_train, epochs=100,validation_data=(X_val, y_val), callbacks=[stop_early])
    best_hp=tuner.get_best_hyperparameters()[0]
    h_model = tuner.hypermodel.build(best_hp)
    h_model.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val))
    
    predicted = h_model.predict(teX)
    print(predicted)
    
    h_model.summary()
    h_eval_dict = h_model.evaluate(teX, teY, return_dict=True)
    print(h_eval_dict)
    h_model.save("model")
    Plotter.architecture(h_model)
    
    """
    train_model = model.fit(X_train, y_train,
                  batch_size=16,
                  epochs=100,
                  verbose=1,
                  validation_data=(X_val, y_val))

    predicted = model.predict(teX)
    print(predicted)
    score = model.evaluate(teX, teY, verbose=0)
    print('Acc:', score[1])
    """

if __name__=='__main__':
    main()