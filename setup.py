from os import listdir
from os.path import isfile, join

import pandas as pd
import seaborn as sns
import tensorflow as tf

from utils import Plotter, Preprocessor

PATH = "data/"
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
    
    
def getModel(*args:any)->tf.keras.Model:
    
    """
    Args: 
        any 
    Does:
        Sequential ANN classification w/ relu's for activation in hidden layer
        
        10 folds cross validation to use (probably)-> 7 train; 1.5 validation; 1.5 test yields optimal results
        
        Classic 4 node output softmax
    
    Returns:
        Keras sequential model
    """



def main():
    getData()

if __name__=='__main__':
    main()