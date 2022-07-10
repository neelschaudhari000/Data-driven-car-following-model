import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

st.write("""
# Penguin Prediction App
This app predicts the **Palmer Penguin** species!
Data obtained from the [palmerpenguins library](https://github.com/allisonhorst/palmerpenguins) in R by Allison Horst.
""")

st.sidebar.header('User Input Features')

st.sidebar.markdown("""
[Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/penguins_example.csv)
""")

# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        Leadvehicletype = st.sidebar.selectbox('PrecVehType',('0','1','2'))
        yourvehicletype = st.sidebar.selectbox('Vehicle.type',('0','1','2'))
        Rear_to_Front_Space_Headway = st.sidebar.slider('Rear_to_Front_Space_Headway', -10,10,5)
        Velocity_difference = st.sidebar.slider('Velocity Difference_Following-Preceding', -10,10,0)
        yourvehiclevelocity = st.sidebar.slider('v_Vel', 0,10,2)
        df = {'PrecVehType': Leadvehicletype,
                'Vehicle.type': yourvehicletype,
                'Rear_to_Front_Space_Headway': Rear_to_Front_Space_Headway,
                'Velocity Difference_Following-Preceding': Velocity_difference,
                'v_Vel': yourvehiclevelocity}
        features = pd.DataFrame(df, index=[0])
        return features
    input_df = user_input_features()

# Combines user input features with entire penguins dataset
# This will be useful for the encoding phase
data = pd.read_csv('test.csv')
data1 = data.drop(columns=['nextframeAcc'])
df = pd.concat([input_df,data1],axis=0)


# Displays the user input features
st.subheader('User Input features')

#if uploaded_file is not None:
 #   st.write(df)
#else:
 #   st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
  #  st.write(df)

# Reads in saved classification model
rf04 = pickle.load(open('reaction_time04.pkl', 'rb'))

def prediction(test,unique_pairs_df,rf,delta_time):
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
# Apply model to make predictions
prediction = prediction(df,['1978-1984'],rf04,0.4)


st.subheader('Prediction')
acc = np.array(['predicted_acceleration'])
st.write(prediction[acc])