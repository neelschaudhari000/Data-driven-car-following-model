import numpy as np
# import pandas as pd ip
import random


class Transformation():
    '''
    Class for the definition of variables that will perform one or other transformation on the input data.
    '''

    def convert_feet_to_metre(self, df):
        '''
        Convert the input variables which exist in Feet to Metres or Feet/Second to Metre/Second and so on. 
        Input: 
            df
        Ouptut: 
            df
        '''
        df['v_length'] = df['v_length']*0.3048
        df['Space_Headway'] = df['Space_Headway']*0.3048
        df['v_Vel'] = df['v_Vel']*0.3048
        df['v_Acc'] = df['v_Acc']*0.3048
        df['Local_X'] = df['Local_X']*0.3048
        df['Local_Y'] = df['Local_Y']*0.3048

        return df

    def drop_not_required_columns(self, df):
        '''
        Drop the columns that are not required for our Model.
        Input: 
            df
        Ouptut: 
            df
        '''
        df = df.drop(columns=['Movement', 'Direction', 'Section_ID', 'Int_ID',
                     'D_Zone', 'O_Zone', 'v_Width', 'Total_Frames', 'Global_X', 'Global_Y'])
        return df

    def create_column_placeholders(self, df):
        '''
        Create PLaceholder for columns that will be generated and populated to use in the Model 
        Input: 
            df
        Ouptut: 
            df
        '''
        df['Preceding_Vehicle_Class'] = np.NaN
        df['Rear_to_Front_Space_Headway'] = np.NaN
        df['Front_To_Rear_Time_Headway'] = np.NaN
        df['Velocity Difference_Following-Preceding'] = np.NaN
        df['Acceleration Difference_Following-Preceding'] = np.NaN
        df['L-F_Pair'] = df['Preceding'].astype(str) + \
            '-' + df['Vehicle_ID'].astype(str)
        df["v_Class_Name"] = df["v_Class"].map(
            {1: "Motorcycle", 2: "Car", 3: "Heavy Vehicle"})
        #df['Relative_Time'] = df['Global_Time'] - df['Global_Time'].min() + 1
        return df

    def bifurcate_highways(self, df):
        '''
        filter out US-101 and I-80 into separate HIghway dataframes as same Vehicle IDs exist in both Hughways.
        Input: 
            df
        Ouptut: 
            df
        '''

        filtered_U = df[((df['Location'] == 'us-101'))]

        filtered_I = df[(df['Location'] == 'i-80')]

        return filtered_U, filtered_I

    def class_vehicle_sets(self, df):
        '''
        Create the set for the vehicle classes, Motorcycle, Car and Heavy Vehicle
        Input: 
            df
        Output:
            Vehicle ID set in sequence: 1. Motorcycle , 2. Car, 3. Heavy Vehicle
        '''

        filtered_vClass = df[['Vehicle_ID', 'v_Class',
                              'v_length']].drop_duplicates().sort_values('v_Class')
        v_Class_M = set(
            filtered_vClass[(filtered_vClass['v_Class'] == 1)]['Vehicle_ID'])
        v_Class_C = set(
            filtered_vClass[(filtered_vClass['v_Class'] == 2)]['Vehicle_ID'])
        v_Class_HV = set(
            filtered_vClass[(filtered_vClass['v_Class'] == 3)]['Vehicle_ID'])

        return v_Class_M, v_Class_C, v_Class_HV

    def preceding_vehicle_class(self, df, v_Class_M, v_Class_C, v_Class_HV):
        '''
        Find and populate the preceding Vehicle Class Name into Preceding_Vehicle_Class column, and populate the Vehicle_combination as well. 
        Input: 
            df,the three deciding Class sets of Motorcycle and Heavy Vehicle. 
        Ouptut:
            df
        '''
        result = []

        for i in df.index:
            if df['Preceding'][i] in v_Class_C:
                result.append('Car')
            elif df['Preceding'][i] in v_Class_HV:
                result.append('Heavy Vehicle')
            elif df['Preceding'][i] in v_Class_M:
                result.append('Motorcycle')
            else:
                result.append('Free Flow')
        df['Preceding_Vehicle_Class'] = result
        df['Vehicle_combination'] = df['Preceding_Vehicle_Class'] + \
            '-' + df['v_Class_Name']
        return df

    def preceding_vehicle_length(self, df):
        '''
        Find and populate the preceding Vehicle Length into preceding_vehicle_length column. If there is no Preceding vehicle, it will be populated with 0 in case of Vehicle ID 0
        Input: 
            df
        Ouptut: 
            df
        '''
        vehicle_lengths = df[['Vehicle_ID', 'v_length']]
        # print(f"{vehicle_lengths.duplicated().sum()} duplicate values have been removed")
        vehicle_lengths.drop_duplicates(inplace=True)
        # print(vehicle_lengths.shape)
        x = vehicle_lengths.groupby(['Vehicle_ID']).mean()
        dict_var = x.to_dict()['v_length']
        df["preceding_vehicle_length"] = df["Preceding"].map(dict_var)
        df["preceding_vehicle_length"] = df["preceding_vehicle_length"].fillna(
            0)
        return df

    def front_to_front_bumper_details_changed_to_rear_to_front_bumper_details(self, df):
        '''
        Change the details from the Front to Front Bumper to Rear of Lead to Front Bumper of Subject Vehicle.
            1. Space Headway
            2. Time Headway
        Input: 
            df
        Ouptut: 
            df
        '''
        df['Rear_to_Front_Space_Headway'] = df['Space_Headway'] - \
            df['preceding_vehicle_length']
        df['Front_To_Rear_Time_Headway'] = df['Rear_to_Front_Space_Headway'] / df['v_Vel']
        df["Front_To_Rear_Time_Headway"] = df["Front_To_Rear_Time_Headway"].replace(
            np.NaN, 9999)
        df["Front_To_Rear_Time_Headway"] = df["Front_To_Rear_Time_Headway"].replace(
            np.inf, 9999)
        return df

    def map_preceding_vehicle_details(self, df):
        '''
        Update Preceding Vehicle Details for the below columns
            1. Previous Vehicle Acceleration.
            2. Previous Vehicle Velocity.
            3. Previous Vehicle Lane Change details
            4. Populate any missing details for Vehicle ID 0 with either 0 or False. 
        Input: 
            df
        Ouptut: 
            df
        '''
        lane_verify = df[['Vehicle_ID', 'Lane_ID']]
        # print(f"{lane_verify.duplicated().sum()} duplicate values have been removed")
        lane_verify.drop_duplicates(inplace=True)
        lane_verify = lane_verify.groupby(
            ['Vehicle_ID'], as_index=False).count()
        # print(lane_verify.shape)
        lane_change_vehicles = set(
            lane_verify[lane_verify["Lane_ID"] > 1]['Vehicle_ID'])
        df['lane_changes'] = df['Vehicle_ID'].isin(
            lane_change_vehicles)
        right_df = df[['Vehicle_ID', 'Global_Time',
                       'v_Vel', 'v_Acc', 'lane_changes', 'Local_Y', 'v_Class']]
        right_df.rename(columns={'Vehicle_ID': 'Prec_Vehicle_ID', 'v_Vel': 'preceding_Vehicle_Velocity',
                        'v_Acc': 'preceding_Vehicle_Acceleration', "lane_changes": "preceding_car_lane_changes", "Local_Y": "preceding_Local_Y", "v_Class": "preceding_v_Class"}, inplace=True)
        df['Prec_Vehicle_ID'] = df['Preceding']
        df = df.merge(right=right_df, how='left', on=(
            'Prec_Vehicle_ID', 'Global_Time'))
        df['preceding_Vehicle_Velocity'] = df['preceding_Vehicle_Velocity'].fillna(
            0)
        df['preceding_Vehicle_Acceleration'] = df['preceding_Vehicle_Acceleration'].fillna(
            0)
        df['preceding_Local_Y'] = df['preceding_Local_Y'].fillna(9999)

        df['preceding_car_lane_changes'] = df['preceding_car_lane_changes'].fillna(
            False)
        df['preceding_v_Class'] = df['preceding_v_Class'].fillna(0)
        df['Velocity Difference_Following-Preceding'] = df['v_Vel'] - \
            df['preceding_Vehicle_Velocity']
        df['Acceleration Difference_Following-Preceding'] = df['v_Acc'] - \
            df['preceding_Vehicle_Acceleration']
        df["prevsecAcc"] = df.groupby(
            ["L-F_Pair"], as_index=False)["v_Acc"].shift(+1)
        df["jerk"] = (df["v_Acc"] - df["prevsecAcc"]) / 0.1

        return df

    def time_pairs(self, df):
        '''
        Map the Pairs for the Preceding and lead vehicle. 
        Input: 
            df
        Ouptut: 
            df
        '''
        df = df.sort_values(by=['Global_Time'],
                            ascending=True, ignore_index=True)
        df['pair_Time_Duration'] = (df.groupby(
            ['L-F_Pair'], as_index=False).cumcount()*0.1)
        x = (df[['L-F_Pair', 'pair_Time_Duration']].groupby(['L-F_Pair'],
             as_index=False).max(['pair_Time_Duration']))
        dict_lenght = dict(zip(x['L-F_Pair'], x['pair_Time_Duration']))
        df["total_pair_duration"] = df["L-F_Pair"].map(dict_lenght)

        return df

    def remap_pair_time(self, df):
        '''
        Map the Pairs for the Preceding and lead vehicle. 
        Input: 
            df
        Ouptut: 
            df
        '''
        df = df.sort_values(by=['Global_Time'],
                            ascending=True, ignore_index=True)
        df['pair_Time_Duration'] = (df.groupby(
            ['L-F_Pair'], as_index=False).cumcount()*0.1)
        x = (df[['L-F_Pair', 'pair_Time_Duration']].groupby(['L-F_Pair'],
             as_index=False).max(['pair_Time_Duration']))
        dict_lenght = dict(zip(x['L-F_Pair'], x['pair_Time_Duration']))
        df["total_pair_dur"] = df["L-F_Pair"].map(dict_lenght)

        return df

    def train_test_pair(self, df, split, neural=False, seed=0):
        '''
        Read the input file into a dataframe.
        Input: File name for the file present in Data folder.
        Output: Dataframe name.
        '''

        if seed > 0:
            random.seed(seed)

        total_pairs = df["L-F_Pair"].unique()
        total_pairs = total_pairs.tolist()
        test_split_cnt = round(len(total_pairs)*split)
        test_split_pairs = random.sample(total_pairs, test_split_cnt)
        train_df = df[df['L-F_Pair'].isin(test_split_pairs)]
        test_df = df[~df['L-F_Pair'].isin(test_split_pairs)]
        if neural:
            validation_split_cnt = round(test_split_cnt*0.1)
        else:
            validation_split_cnt = round(test_split_cnt*0.0)

        validation_split_pairs = random.sample(
            test_split_pairs, validation_split_cnt)
        val_df = df[df['L-F_Pair'].isin(validation_split_pairs)]
        train_df = df[~df['L-F_Pair'].isin(validation_split_pairs)]
        return train_df, val_df, test_df
