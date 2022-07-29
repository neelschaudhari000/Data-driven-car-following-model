from cProfile import label
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns



st.write("""
# Single Lane Car-Following Trajectory Prediction
This app predicts the **Acceleration and calculate trajectories (speed, Spacing, and jerk)**
""")



reaction_time = st.sidebar.selectbox('reaction time',(0.1,0.2,0.3,0.5,1,2,4))
Vehicle_combination = st.sidebar.selectbox('Vehicle_combination',('Car-Heavy Vehicle', 'Heavy Vehicle-Car', 'Car-Car'))
#Vehicle_Pair_No = st.sidebar.selectbox('Vehicle_combination',('2322-2330', '551-560', '1304-1309', '2785-2804', '3084-3094','439-444', '2695-2725', '2725-2717', '1635-1642'))

if reaction_time == 0.1:
    df = pd.read_csv("Prediction_set_Predited_data_0.1.csv")
elif reaction_time == 0.2:
    df = pd.read_csv("Prediction_set_Predited_data_0.2.csv")
elif reaction_time == 0.3:
    df = pd.read_csv("Prediction_set_Predited_data_0.3.csv")
elif reaction_time == 0.5:
    df = pd.read_csv("Prediction_set_Predited_data_0.5.csv")
elif reaction_time == 1:
    df = pd.read_csv("Prediction_set_Predited_data_1.csv")
elif reaction_time == 2:
    df = pd.read_csv("Prediction_set_Predited_data_2.csv")
else:
    df = pd.read_csv("Prediction_set_Predited_data_4.csv")
df = df[df["Vehicle_combination"] == Vehicle_combination]

if Vehicle_combination == 'Car-Heavy Vehicle':
    Vehicle_Pair_No = st.sidebar.selectbox('Vehicle_combination',('2322-2330','2785-2804','2725-2717'))
elif Vehicle_combination == 'Heavy Vehicle-Car':
    Vehicle_Pair_No = st.sidebar.selectbox('Vehicle_combination',('551-560','3084-3094','2695-2725'))
else:
    Vehicle_Pair_No = st.sidebar.selectbox('Vehicle_combination',('1304-1309','439-444','1635-1642'))

df = df[df["L-F_Pair"] == Vehicle_Pair_No ]


df["Pair Time Duration"] = df["pair_Time_Duration"]
df["Actual Velocity"] = df["nextframesvel"]
df["Actual Acceleration"] = df["nextframeAcc"]
df["Actual Jerk"] = df["nextframejerk"]
df["Actual Spacing"] = df["nextFrameSpacing"]

#plt.figure(figsize=(15, 9))
fig1, ax = plt.subplots()
#fig1 = plt.figure(figsize = (16, 10))
#plt.figure(figsize=(15, 9))
plt.title("Actual Velocity vs Predicted Velocity",color = "green", size = 18)
plt.xlabel("Pair Time Duration (s)", color = "black", size = 18)
plt.ylabel("Velocity (m/s)", color = "black", size = 18)
plt.xticks( range(0,60,5) )
plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)
sns.lineplot(x = df["Pair Time Duration"],y= df['Actual Velocity'],color="black",label = "Actual Velocity")
ax=sns.lineplot(x = df["Pair Time Duration"],y= df['rf_predicted_velocity'],color="b",label = "RF Predicted Velocity")
ax=sns.lineplot(x = df["Pair Time Duration"],y= df['knn_predicted_velocity'],color="orange",label = "KNN Predicted Velocity")
ax=sns.lineplot(x = df["Pair Time Duration"],y= df['cnn_predicted_velocity'],color="purple",label = "CNN Predicted Velocity")
plt.legend(bbox_to_anchor =(1.075, 1.0), ncol = 1)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.tick_params(left = False, bottom = False)




fig2, ax = plt.subplots()
#fig2 = plt.figure(figsize = (16, 10))
plt.title("Actual Spacing vs predicted Spacing",color = "gold", size = 18)
plt.xlabel("Pair Time Duration (s)", color = "black", size = 18)
plt.ylabel("Spacing (m)", color = "black", size = 18)
plt.xticks( range(0,60,5) )
plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)
sns.lineplot(x = df["Pair Time Duration"],y= df['Actual Spacing'],color="black",label = "Actual Spacing")
ax=sns.lineplot(x = df["Pair Time Duration"],y= df['rf_predicted_spacing'],color="b",label = "RF Predicted Spacing")
ax=sns.lineplot(x = df["Pair Time Duration"],y= df['knn_predicted_spacing'],color="orange",label = "KNN Predicted Spacing")
ax=sns.lineplot(x = df["Pair Time Duration"],y= df['cnn_predicted_spacing'],color="purple",label = "CNN Predicted Spacing")
plt.legend(bbox_to_anchor =(1.075, 1.0), ncol = 1)
#for line, name in zip(ax.lines, df[['Actual Velocity','rf_predicted_velocity','knn_predicted_acceleration','cnn_predicted_acceleration']].columns):
 ##  ax.annotate(name, xy=(0.949, y), xytext=(6, 0),
   #                    color=line.get_color(), xycoords=ax.get_yaxis_transform(),
    #                  textcoords="offset points", size=15, va = "center")
#plt.legend(bbox_to_anchor =(1.075, 1.0), ncol = 1)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.tick_params(left = False, bottom = False)



def acceleration(df,actual,prediction,actual_label, prediction_label,title):
      fig, ax = plt.subplots()
      #plt.figure(figsize=(15, 9))
      plt.title(title, color = "grey", size = 18)
      plt.xlabel("Time Duration (s)", color = "black", size = 18)
      plt.ylabel("Acceleration (m/s^2)", color = "black", size = 18)
      plt.xticks( range(0,60,5) )
      sns.lineplot(x = df["pair_Time_Duration"],y= df[actual],color="black", label = actual_label)
      ax=sns.lineplot(x = df["pair_Time_Duration"],y= df[prediction],color="b", label = prediction_label)
      plt.legend(bbox_to_anchor =(1.075, 1.0), ncol = 1)
      ax.spines['right'].set_visible(False)
      ax.spines['top'].set_visible(False)
      ax.spines['left'].set_visible(False)
      ax.spines['bottom'].set_visible(False)
      ax.tick_params(left = False, bottom = False)
    

fig3, ax = plt.subplots()
#plt.figure(figsize=(15, 9))
plt.title("Actual Subject Acceleration vs RF Predicted Acceleration", color = "grey", size = 18)
plt.xlabel("Time Duration (s)", color = "black", size = 18)
plt.ylabel("Acceleration (m/s^2)", color = "black", size = 18)
plt.xticks( range(0,60,5) )
sns.lineplot(x = df["pair_Time_Duration"],y= df['Actual Acceleration'],color="black", label = "Actual Subject Acceleration")
ax=sns.lineplot(x = df["pair_Time_Duration"],y= df["rf_predicted_acceleration"],color="b", label = "RF Predicted Acceleration")
plt.legend(bbox_to_anchor =(1.075, 1.0), ncol = 1)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.tick_params(left = False, bottom = False)

#fig3 = acceleration(df,"Actual Acceleration","rf_predicted_acceleration","Actual Subject Acceleration","RF Predicted Acceleration","Actual Subject Acceleration vs RF Predicted Acceleration")

fig4, ax = plt.subplots()
#plt.figure(figsize=(15, 9))
plt.title("Actual Subject Acceleration vs KNN Predicted Acceleration", color = "grey", size = 18)
plt.xlabel("Time Duration (s)", color = "black", size = 18)
plt.ylabel("Acceleration (m/s^2)", color = "black", size = 18)
plt.xticks( range(0,60,5) )
sns.lineplot(x = df["pair_Time_Duration"],y= df['Actual Acceleration'],color="black", label = "Actual Subject Acceleration")
ax=sns.lineplot(x = df["pair_Time_Duration"],y= df["knn_predicted_acceleration"],color="b", label = "KNN Predicted Acceleration")
plt.legend(bbox_to_anchor =(1.075, 1.0), ncol = 1)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.tick_params(left = False, bottom = False)
#fig4 = acceleration(df,"Actual Acceleration","knn_predicted_acceleration","Actual Subject Acceleration","KNN Predicted Acceleration","Actual Subject Acceleration vs KNN Predicted Acceleration")

fig5, ax = plt.subplots()
#plt.figure(figsize=(15, 9))
plt.title("Actual Subject Acceleration vs CNN Predicted Acceleration", color = "grey", size = 18)
plt.xlabel("Time Duration (s)", color = "black", size = 18)
plt.ylabel("Acceleration (m/s^2)", color = "black", size = 18)
plt.xticks( range(0,60,5) )
sns.lineplot(x = df["pair_Time_Duration"],y= df['Actual Acceleration'],color="black", label = "Actual Subject Acceleration")
ax=sns.lineplot(x = df["pair_Time_Duration"],y= df["cnn_predicted_acceleration"],color="b", label = "CNN Predicted Acceleration")
plt.legend(bbox_to_anchor =(1.075, 1.0), ncol = 1)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.tick_params(left = False, bottom = False)
#fig5 = acceleration(df,"Actual Acceleration","cnn_predicted_acceleration","Actual Subject Acceleration","CNN Predicted Acceleration","Actual Subject Acceleration vs CNN Predicted Acceleration")


def jerk(df,actual,prediction,actual_label, prediction_label,title):
      fig3, ax = plt.subplots()
      #plt.figure(figsize=(15, 9))
      plt.title(title, color = "red", size = 18)
      plt.xlabel("Time Duration (s)", color = "black", size = 18)
      plt.ylabel("Jerk (m/s^3)", color = "black", size = 18)
      plt.xticks( range(0,60,5) )
      sns.lineplot(x = df["pair_Time_Duration"],y= df[actual],color="black", label = actual_label)
      ax=sns.lineplot(x = df["pair_Time_Duration"],y= df[prediction],color="b", label = prediction_label)
      plt.legend(bbox_to_anchor =(1.075, 1.0), ncol = 1)
      ax.spines['right'].set_visible(False)
      ax.spines['top'].set_visible(False)
      ax.spines['left'].set_visible(False)
      ax.spines['bottom'].set_visible(False)
      ax.tick_params(left = False, bottom = False)
      return fig3
#12
fig6, ax = plt.subplots()
#plt.figure(figsize=(15, 9))
plt.title("Actual Subject Jerk vs RF Predicted Jerk", color = "red", size = 18)
plt.xlabel("Time Duration (s)", color = "black", size = 18)
plt.ylabel("Jerk (m/s^2)", color = "black", size = 18)
plt.xticks( range(0,60,5) )
sns.lineplot(x = df["pair_Time_Duration"],y= df['Actual Jerk'],color="black", label = "Actual Subject Jerk")
ax=sns.lineplot(x = df["pair_Time_Duration"],y= df["rf_predicted_jerk"],color="b", label = "RF Predicted Jerk")
plt.legend(bbox_to_anchor =(1.075, 1.0), ncol = 1)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.tick_params(left = False, bottom = False)
#fig6 = jerk(df,"Actual Jerk","rf_predicted_jerk","Actual Subject Jerk","RF Predicted Jerk","Actual Subject Jerk vs RF Predicted Jerk")


fig7, ax = plt.subplots()
#plt.figure(figsize=(15, 9))
plt.title("Actual Subject Jerk vs KNN Predicted Jerk", color = "red", size = 18)
plt.xlabel("Time Duration (s)", color = "black", size = 18)
plt.ylabel("Jerk (m/s^2)", color = "black", size = 18)
plt.xticks( range(0,60,5) )
sns.lineplot(x = df["pair_Time_Duration"],y= df['Actual Jerk'],color="black", label = "Actual Subject Jerk")
ax=sns.lineplot(x = df["pair_Time_Duration"],y= df["knn_predicted_jerk"],color="b", label = "KNN Predicted Jerk")
plt.legend(bbox_to_anchor =(1.075, 1.0), ncol = 1)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.tick_params(left = False, bottom = False)
#fig7 = jerk(df,"Actual Jerk","knn_predicted_jerk","Actual Subject Jerk","KNN Predicted Jerk","Actual Subject Jerk vs KNN Predicted Jerk")

fig8, ax = plt.subplots()
#plt.figure(figsize=(15, 9))
plt.title("Actual Subject Jerk vs CNN Predicted Jerk", color = "red", size = 18)
plt.xlabel("Time Duration (s)", color = "black", size = 18)
plt.ylabel("Jerk (m/s^2)", color = "black", size = 18)
plt.xticks( range(0,60,5) )
sns.lineplot(x = df["pair_Time_Duration"],y= df['Actual Jerk'],color="black", label = "Actual Subject Jerk")
ax=sns.lineplot(x = df["pair_Time_Duration"],y= df["cnn_predicted_jerk"],color="b", label = "CNN Predicted Jerk")
plt.legend(bbox_to_anchor =(1.075, 1.0), ncol = 1)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.tick_params(left = False, bottom = False)
#fig8 = jerk(df,"Actual Jerk","cnn_predicted_jerk","Actual Subject Jerk","CNN Predicted Jerk","Actual Subject Jerk vs CNN Predicted Jerk")



plot_selection = st.sidebar.selectbox('Plot Selection',('All','Velocity', 'Acceleration', 'Jerk', 'Spacing'))

if plot_selection == 'Acceleration':
    st.pyplot(fig3)
    st.pyplot(fig4)
    st.pyplot(fig5)
elif plot_selection == 'Velocity':
    st.pyplot(fig1)
elif plot_selection == 'Spacing':
    st.pyplot(fig2)
elif plot_selection == 'Jerk':
    st.pyplot(fig6)
    st.pyplot(fig7)
    st.pyplot(fig8)  

else:
    st.pyplot(fig3)
    st.pyplot(fig4)
    st.pyplot(fig5)
    st.pyplot(fig1)
    st.pyplot(fig2)   
    st.pyplot(fig6)
    st.pyplot(fig7)
    st.pyplot(fig8)