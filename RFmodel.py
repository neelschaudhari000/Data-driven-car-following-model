from asyncio.windows_events import NULL
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import random
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import FileProcessing
import warnings
warnings.filterwarnings("ignore")

class ModelClass():
    
    def preprocessing(self,df,time_frame):
        df["nextframeAcc"] = df.groupby(
            ["L-F_Pair"], as_index=False)["v_Acc"].shift(-10*time_frame)
        df["nextframesvel"] = df.groupby(
            ["L-F_Pair"], as_index=False)["v_Vel"].shift(-10*time_frame)
        df["nextframeposition"] = df.groupby(
            ["L-F_Pair"], as_index=False)["Local_Y"].shift(-10*time_frame)
        df['Pair_Time_Duration']=(df.groupby(['L-F_Pair'],as_index=False).cumcount()*0.1) +0.1
        df['PrecVehType'] = df['Preceding_Vehicle_Class'].map({'Motorcycle': 1, 'Car': 2, 'Heavy Vehicle': 3, 'Free Flow':4})
        df['Vehicle.type'] = df['v_Class']
        df = df[df["Preceding_Vehicle_Class"].notna()]
        df = df[df["v_Vel"].notna()]
        df = df[df["Rear_to_Front_Space_Headway"].notna()]
        df = df[df["Local_Y"].notna()]
        df = df[df["nextframeAcc"].notna()]
        df = df[df["Pair_Time_Duration"].notna()]
        df = df[df["nextframeposition"].notna()]
        df = df[df["nextframesvel"].notna()]
        df = df[df["PrecVehType"].notna()]
        df = df[df["Vehicle.type"].notna()]
        return df
    
    def select_training_pairs(self,df):
        random.seed(2109)
        pairs = df["L-F_Pair"].unique()
        pairs = pairs.tolist()
        v = round(len(pairs)*0.7)   
        pairs = random.sample(pairs, v)
        return pairs

    def split_df_into_train_test(self,df,train_pairs):
        #converting the total dataset to 70/30% pair for train and test. 
        train = df[df['L-F_Pair'].isin(train_pairs)]
        test = df[~df['L-F_Pair'].isin(train_pairs)]
        return train, test

    def fit_rfmodel(self,train,test,number_of_estimators):
        X_train = train[["Rear_to_Front_Space_Headway",'PrecVehType','Vehicle.type','Velocity Difference_Following-Preceding','v_Vel']]
        y_train= train['nextframeAcc']
        X_test = test[["Rear_to_Front_Space_Headway",'PrecVehType','Vehicle.type','Velocity Difference_Following-Preceding','v_Vel']]
        y_test= test['nextframeAcc']
        rf = RandomForestRegressor(n_estimators = number_of_estimators,n_jobs=-1)
        rf.fit(X_train,y_train)
        return X_train, y_train, X_test, y_test,rf

    def prediction_test_pairs(self, df, pair_from, pair_to):
        unique_pairs_values = df['L-F_Pair'].unique()
        unique_pairs_list = unique_pairs_values.tolist()
        unique_pairs_df = unique_pairs_list[pair_from:pair_to]
        return unique_pairs_df

    def prediction(self,test,unique_pairs_df,rf,delta_time):
        predicted_df = []
        input_df = pd.DataFrame()
        # unique_pairs_df is the test range
        for i in unique_pairs_df:
            # Q this is the input data frame
            input_df = test[test['L-F_Pair']== i]
            spacing = np.zeros(input_df.shape[0])
            local_y_subject = np.zeros(input_df.shape[0])
            local_y_preceding = np.zeros(input_df.shape[0])
            dv = np.zeros(input_df.shape[0])
            vel = np.zeros(input_df.shape[0])
            pred_acc = np.zeros(input_df.shape[0])

            vel[0] = input_df.iloc[0]['v_Vel']
            spacing[0] = input_df.iloc[0]['Rear_to_Front_Space_Headway']
            dv[0] = input_df.iloc[0]['Velocity Difference_Following-Preceding']

            local_y_subject[0] = input_df.iloc[0]['Local_Y']
            local_y_preceding[0] = input_df.iloc[0]['previous_Local_Y']
            preceding_vehicle_class = input_df.iloc[0]['PrecVehType']
            vehicle_class = input_df.iloc[0]['Vehicle.type']
            length_preceding_vehicle = input_df.iloc[0]['preceding_vehicle_length']

            predict_for_input = np.array(
                [spacing[0], preceding_vehicle_class, vehicle_class, dv[0], vel[0]]).reshape(1, -1)
            pred_acc[0] = rf.predict(predict_for_input)
            print(
                f"j: {0} input:{predict_for_input},subject localy:{local_y_subject[0]},preceding_local_y:{local_y_preceding[0]},spacing:{spacing[0]} pred_acc: {pred_acc[0]}")
            vel[1] = vel[0]+(pred_acc[0]*delta_time)


            dv[1] = vel[1] - input_df.iloc[1]['previous_Vehicle_Velocity']

            s_subject = ((vel[0]*delta_time ) +
                            (0.5*pred_acc[0]*pow(delta_time, 2)))
                            #should be 1  second here

            local_y_subject[1] = local_y_subject[0] + s_subject
            local_y_preceding[1] = input_df.iloc[1]['previous_Local_Y'] 

            spacing[1] = local_y_preceding[1] - \
                local_y_subject[1] - length_preceding_vehicle

            for j in range(1, len(input_df)):
                predict_for_input = np.array(
                    [spacing[j], preceding_vehicle_class, vehicle_class, dv[j], vel[j]]).reshape(1, -1)
                
                pred_acc[j] = rf.predict(predict_for_input)
                if j == len(input_df)-1:
                    break
                
                vel[j+1] = vel[j]+(pred_acc[j]*0.1)


                dv[j+1] = vel[j+1] - input_df.iloc[j+1]['previous_Vehicle_Velocity']


                s_subject = ((vel[j]*0.1) +
                                (0.5*pred_acc[j]*pow(0.1, 2)))
                                

                
                local_y_subject[j+1] = local_y_subject[j] + s_subject
                local_y_preceding[j+1] = input_df.iloc[j+1]['previous_Local_Y']

                spacing[j+1] = local_y_preceding[j+1] - \
                    local_y_subject[j+1] - length_preceding_vehicle

                print(f"j: {j} input:{predict_for_input},subject localy:{local_y_subject[j]},preceding_local_y:{local_y_preceding[j]},spacing:{spacing[j]} pred_acc: {pred_acc[j]}")

            print(f"input_df shape: {input_df.shape}")
            print(f"pred_acc shape: {pred_acc.shape}")
            input_df['predicted_acceleration'] = pred_acc
            input_df['predicted_velocity'] = vel
            input_df['predicted_spacing'] = spacing

            predicted_df.append(input_df)
            result = pd.concat(predicted_df)
            #r.append(r2_score(Q[target_variable], Q['pacc']))      
            return result

    def plot_1(self, df,nextframe,prediction,title):
        plt.figure(figsize=(10, 8))
        ax = sns.lineplot(x=df["Pair_Time_Duration"], y = df[nextframe], color="r", label="Actual Value")
        sns.lineplot(x=df["Pair_Time_Duration"], y =df[prediction],  color="b", label="Fitted Values" )
        plt.title(title)
        plt.show()
        plt.close()
        return plt
