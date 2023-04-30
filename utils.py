import glob
import re
from os import listdir
from os.path import exists, isfile, join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class Preprocessor:
    @staticmethod
    def cleaner(path:str,dataPos:int=22,colPos:int=19)->None:
        
        """
        Utility function for conversion of .dat to .csv for better pandas handling.
        Includes optional one-hot encoding  if given .dat contains more than 2 classes.
        
        Args:
            path: str
                Represents path to the .dat file
                    Default value: 22
            dataPos: int
                Index of a row from which data starts in .dat file
                    Default value: 19
            colPos: int
                Index of a row containing names of columns

        """
        
        print(f"Reading \"{path}\"")

        wild = pd.read_csv(path,sep="\t")
        wild_rows =wild.iloc[colPos][0].split(',')

        rows = [re.sub(r'@\S+\s','',row).replace(" ","") for row in wild_rows]
        rows.append('Class')
        df = [(item[0].split(',')) for item in wild.iloc[dataPos:].values]
        data=pd.DataFrame(data=df,columns=rows)
        
        data.to_csv(f"csv/{path.split('.')[0].split('/')[1]}.csv",index=None)
        print("Done!")
    
    @staticmethod
    def onehot(df:pd.DataFrame):
        """Performs onehot encoding

        Args:
            df (pd.DataFrame): Dataframe that needs onehot

        Returns:
            _type_: onehot encoded Dataframe
        """
        df = df.join(pd.get_dummies(df['Class'])).drop('Class',axis=1)
        return df
    
    
    
    @staticmethod
    def NaDetector(df:pd.DataFrame,replacer:str=0)->pd.DataFrame:
        """Replace NaN(s) with chosen value

        Args:
            df: pd.DataFrame
                Dataframe to be checked for NaN
            replacer: str
                How to replace Nan values
                Default value: 0\n
                If this parameter is zero (0), replace all NA values with 0 else replace with mean of the current column values
        Returns:
            pd.Dataframe
        """
        for col in df.columns:
            df.fillna(0 if replacer==0 else df[col].mean(),inplace=True) if df.isna().values.any() else df
        return df


    def aggregate(folder:str)->list[pd.DataFrame]:
        """Aggregate multiple Dataframes into three dataframes; training, valdiate, test

        Args:
            folder: str
                Path leading to a folder where csv folds are stored  
        Returns:
            list[pd.DataFrame]
                List of training, validation and test dataframes
        """
        csv_files = glob.glob(folder + "/ve*.csv")
        print(csv_files)
        df_l = [pd.read_csv(file) for file in csv_files]
        #print(df_l)
        df = pd.concat(df_l,ignore_index=True)
        preonehot = pd.DataFrame(df)
        df = Preprocessor.onehot(df)
        train, validate, test = np.split(df.sample(frac=1, random_state=42), 
                       [int(.8*len(df)), int(.9*len(df))])
        
        df.to_csv(f"{folder}/aggregated.csv",index=None)
        train.to_csv(f"{folder}/train.csv",index=None)
        validate.to_csv(f"{folder}/validation.csv",index=None)
        test.to_csv(f"{folder}/test.csv",index=None)
        return train,validate,test,df,preonehot
    
    
    
class Plotter():
    sns.set(font_scale=0.6)
    @staticmethod
    def correlation(df:pd.DataFrame):
        """Analyze given Dataframe and store it's correlation heatmap as an image

        Args:
            df (pd.DataFrame): Dataframe which gets analyzed for correlation between columns
        """
        if not exists("plots/heatmap.png"):
            df=df.iloc[:,:18]
            df=df.corr()
            heatmap=sns.heatmap(df, annot=True, fmt=".1g", cmap='coolwarm',linewidth=2).get_figure()
            heatmap.savefig("plots/heatmap.png") 

    def pair(df:pd.DataFrame):
        """Analyze given Dataframe columns and plot them against each other, resulting in an image 
        showing how each column pair affects each class

        Args:
            df (pd.DataFrame): Dataframe which get analyzed for column:class relationship
        """
        if not exists("plots/pairgrid.png"):
            g = sns.PairGrid(df, hue="Class")
            g.map_diag(sns.histplot)
            g.map_offdiag(sns.scatterplot)
            g.add_legend()
            g.figure.savefig("plots/pairgrid.png") 
            
            
    def count(df:pd.DataFrame):
        """Generates plot describing relationship between each dataset size

        Args:
            df (pd.DataFrame): aggregated pre-onehot dataset
        """
        if not exists("plots/countplot.png"):
            fig = sns.countplot(x = 'Class', hue = 'Class', data = df, palette = 'magma')
            plt.title('Classes')
            fig.figure.savefig("plots/countplot.png")

def main():
    pass

if __name__=='__main__':
    main()