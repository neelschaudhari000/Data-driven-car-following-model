import pandas as pd
from pathlib import Path
import pickle
from tensorflow import keras


class FileProcessing():
    p = Path().cwd()
    stringpath = str(p)[0:str(p).rfind('\\')] + '\\data'
    print(f"original File path: {p}")
    print(f"Data File path: { stringpath}")
#    def __init__(self):
#        self.initialize()

    def read_input(self, file_name):
        '''
        xyz
        Read the input file into a dataframe. 
        Input: File name for the file present in Data folder. 
        Output: Dataframe name. 
        '''

        ngsimfile = self.stringpath + '/' + file_name + '.csv'
        df = pd.read_csv(ngsimfile, low_memory=False)
        print(f"File Read Complete: {ngsimfile}")
        return df

    def read_model(self, model_name, delta_time, neural=False):
        '''
        xyz
        Read the input file into a dataframe. 
        Input: File name for the file present in Data folder. 
        Output: Dataframe name. 
        '''
        if neural:
            model_package_name = self.stringpath + '/' + \
                model_name + str(delta_time) + '.keras'
            model = keras.models.load_model(model_package_name)
        else:
            model_package_name = self.stringpath + '/' + \
                model_name + str(delta_time) + '.pkg'

            model = pickle.load(open(model_package_name, 'rb'))
        print(f"Model Load Completed: {model_package_name}")
        return model

    def export_file(self, df, file_name):
        '''
        Export the working Data frame into csv file of the mentioned name.  
        Input: 
            df
        Ouptut: 
            df
        '''

        filepathname = self.stringpath + '\\' + file_name + '.csv'
        df.to_csv(filepathname, index=False)
        print(f"Files Save Completed: {filepathname}")
        return True

    def merge_files(self, df1, df2):
        '''
        Merge the I-80 and US-101 Highway dataframe.  
        Input: 
            df
        Ouptut: 
            df
        '''
        df = pd.concat([df1, df2])

        print(
            f" Merged Record Count:{df.shape[0]}, df1:{df1.shape[0]}, df2:{df2.shape[0]}")

        return df
