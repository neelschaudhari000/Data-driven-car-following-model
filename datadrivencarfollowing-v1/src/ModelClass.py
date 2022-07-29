from tensorflow.keras import layers
from tensorflow import keras

import tensorflow
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
import random
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import FileProcessing


import warnings
warnings.filterwarnings("ignore")


def fit_and_run_neural(self, df, time_frame):
    shift_instance = time_frame*10
    df, train_df, val_df, test_df, X_train, y_train, X_val, y_val, X_test, y_test = self.preprocessing(
        df, shift_instance, True)
    model = self.define_neural_network(X_train)
    model = self.fit_neural_network(
        model, X_train, y_train, X_val, y_val, time_frame)
    predict_on_pair = self.prediction_test_pairs(test_df, 10, 12)
    predict_on_pair[0]
    print(f"Prediction being done on :{predict_on_pair[0]}")
    target_variable = 'nextframeAcc'

    predicted_data = self.prediction(
        test_df, predict_on_pair, target_variable, model, time_frame)
    prediction_1 = predicted_data[predicted_data["L-F_Pair"]
                                  == predict_on_pair[0]]
    self.display_prediction_plots(prediction_1, time_frame, 'CNN ')

    return df, train_df, val_df, test_df, X_train, y_train, X_val, y_val, X_test, y_test, predicted_data, model


def fit_and_run_neural(self, df, time_frame):
    shift_instance = time_frame*10
    df, train_df, val_df, test_df, X_train, y_train, X_val, y_val, X_test, y_test = self.preprocessing(
        df, shift_instance, True)
    model = self.define_neural_network(X_train)
    model = self.fit_neural_network(
        model, X_train, y_train, X_val, y_val, time_frame)
    predict_on_pair = self.prediction_test_pairs(test_df, 10, 12)
    predict_on_pair[0]
    print(f"Prediction being done on :{predict_on_pair[0]}")
    target_variable = 'nextframeAcc'

    predicted_data = self.prediction(
        test_df, predict_on_pair, target_variable, model, time_frame)
    prediction_1 = predicted_data[predicted_data["L-F_Pair"]
                                  == predict_on_pair[0]]
    self.display_prediction_plots(prediction_1, time_frame, 'CNN ')

    return df, train_df, val_df, test_df, X_train, y_train, X_val, y_val, X_test, y_test, predicted_data, model


class ModelClass():

    '''

    '''
    file = FileProcessing.FileProcessing()

    def fit_and_run_neural(self, train_df,  val_df, test_df,  time_frame):
        shift_instance = time_frame*10
        train_df, test_df, X_train, y_train, X_test, y_test, val_df, X_val, y_val = self.preprocessing(
            train_df, test_df, shift_instance, True, val_df)
        model = self.define_neural_network(X_train)
        model = self.fit_neural_network(
            model, X_train, y_train, time_frame, X_val, y_val)
        predict_on_pair = self.prediction_test_pairs(test_df, 10, 12)
        predict_on_pair[0]
        print(f"Prediction being done on :{predict_on_pair[0]}")
        target_variable = 'nextframeAcc'

        predicted_data = self.prediction(
            test_df, predict_on_pair, target_variable, model, time_frame)
        prediction_1 = predicted_data[predicted_data["L-F_Pair"]
                                      == predict_on_pair[0]]
        self.display_prediction_plots(prediction_1, time_frame, 'CNN ')

        return train_df, val_df, test_df, X_train, y_train, X_val, y_val, X_test, y_test, predicted_data, model

    def preprocessing(self, train_df,  test_df, time_frame, neural=False, val_df=[]):

        train_df = self.create_prediction_columns(train_df, time_frame)
        if len(val_df) > 0:
            val_df = self.create_prediction_columns(val_df, time_frame)
        test_df = self.create_prediction_columns(test_df, time_frame)

        X_train, y_train, X_test, y_test, X_val, y_val = self.feature_selection(
            train_df, test_df, val_df)

        return train_df, test_df, X_train, y_train, X_test, y_test, val_df, X_val, y_val

    def feature_selection(self, train_df, test_df, val_df=[]):
        features = ['Rear_to_Front_Space_Headway', 'preceding_v_Class', "v_Class",
                    'Velocity Difference_Following-Preceding', 'v_Vel']
        X_train = train_df[features]
        y_train = train_df['nextframeAcc']

        X_test = test_df[features]
        y_test = test_df['nextframeAcc']

        if len(val_df) > 0:
            y_val = val_df['nextframeAcc']
            X_val = val_df[features]
        else:
            y_val = []
            X_val = []

        return X_train, y_train,  X_test, y_test, X_val, y_val

    def create_prediction_columns(self, df, n):
        '''
        create the prediction pair by shifting the actual data up by the mentioned number(0.1*n seconds) to create the timeseries info
        '''
        df["nextframeAcc"] = df.groupby(
            ["L-F_Pair"], as_index=False)["v_Acc"].shift(-1*n)
        df["nextframesvel"] = df.groupby(
            ["L-F_Pair"], as_index=False)["v_Vel"].shift(-1*n)
        df["nextframeposition"] = df.groupby(
            ["L-F_Pair"], as_index=False)["Local_Y"].shift(-1*n)
        df["nextframejerk"] = df.groupby(
            ["L-F_Pair"], as_index=False)["jerk"].shift(-1*n)
        df["nextFrameSpacing"] = df.groupby(
            ["L-F_Pair"], as_index=False)["Rear_to_Front_Space_Headway"].shift(-1*n)
        df["precnextframeposition"] = df.groupby(
            ["L-F_Pair"], as_index=False)["preceding_Local_Y"].shift(-1*n)
        df["precnextframesvel"] = df.groupby(
            ["L-F_Pair"], as_index=False)["preceding_Vehicle_Velocity"].shift(-1*n)
        df = df[df['nextframeposition'].notna()]
        df = df[df['nextframesvel'].notna()]
        df = df[df['nextframeAcc'].notna()]
        df = df[df['nextFrameSpacing'].notna()]

        return df

    def prediction_test_pairs(self, df, pair_from, pair_to, vehicle_combination=''):
        if vehicle_combination > '':
            df = df[(df['Vehicle_combination'] == vehicle_combination)]
        unique_pairs_values = df['L-F_Pair'].unique()
        unique_pairs_list = unique_pairs_values.tolist()
        if pair_to == 9999:
            unique_pairs_df = unique_pairs_list
        else:
            unique_pairs_df = unique_pairs_list[pair_from:pair_to]

        return unique_pairs_df

    def define_neural_network(self, input_df):

        input_df = tensorflow.expand_dims(input_df, axis=-1)

        input = keras.Input(shape=(input_df.shape[1], 1))

        x = layers.Conv1D(filters=16, kernel_size=(
            2), padding='same', activation="sigmoid", name='Block1_Conv1')(input)
        x = layers.Conv1D(filters=16, kernel_size=(
            2), padding='same', activation="sigmoid", name='Block1_Conv2')(x)

        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.05)(x)
        x = layers.Conv1D(filters=32, kernel_size=(
            2), padding='same', activation="elu", name='Block2_Conv1')(x)
        x = layers.Conv1D(filters=32, kernel_size=(
            2), padding='same', activation="elu", name='Block2_Conv2')(x)

        x = layers.Dropout(0.05)(x)
        x = layers.Conv1D(filters=32, kernel_size=(
            2), padding='same', activation="tanh", name='Block3_Conv1')(x)
        x = layers.Conv1D(filters=32, kernel_size=(
            2), padding='same', activation="tanh", name='Block3_Conv2')(x)

        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.05)(x)
        # prework for fully connected layer.
        x = layers.Flatten()(x)
        # Fully connected layers
        x = layers.Dense(128, activation='tanh')(x)
        x = layers.Dense(64, activation='sigmoid')(x)
        x = layers.Dense(16, activation='tanh')(x)
        outputs = layers.Dense(1, activation="elu")(x)

        model = keras.Model(inputs=input, outputs=outputs)

        model.compile(optimizer="adam",
                      loss="mean_squared_error",
                      metrics=[keras.metrics.RootMeanSquaredError()])

        model.summary()

        return model

    def fit_neural_network(self, model, X_train, y_train, reaction_time, X_val=[], y_val=[]):
        modelName = "neural_network_model" + str(reaction_time) + ".keras"
        if len(X_val) == 0:
            early_stopping = keras.callbacks.EarlyStopping(
                monitor='root_mean_squared_error', verbose=1, patience=7)
            model.fit(X_train, y_train, epochs=10, batch_size=16,
                      verbose=1, callbacks=[early_stopping])
            model.save(modelName, overwrite=True,)
        else:
            save_callback = keras.callbacks.ModelCheckpoint(
                modelName, save_best_only=True)
            early_stopping = keras.callbacks.EarlyStopping(
                monitor='root_mean_squared_error', verbose=1, patience=7)
            history = model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1, validation_data=(
                X_val, y_val), callbacks=[save_callback, early_stopping])
            history_dict = history.history
            loss_values = history_dict['loss']   # training loss
            val_loss_values = history_dict['val_loss']  # validation loss

            # creates list of integers to match the number of epochs of training
            epochs = range(1, len(loss_values)+1)

            # code to plot the results
            plt.plot(epochs, loss_values, 'b', label="Training Loss")
            plt.plot(epochs, val_loss_values, 'r', label="Validation Loss")
            plt.title("Training and Validation Loss")
            plt.xlabel("Epochs")
            plt.xticks(epochs)
            plt.ylabel("Loss")
            plt.legend()
            plt.show()
            # As above, but this time we want to visualize the training and validation accuracy
            acc_values = history_dict['root_mean_squared_error']
            val_acc_values = history_dict['val_root_mean_squared_error']

            plt.plot(epochs, acc_values, 'b', label="Training RMSE")
            plt.plot(epochs, val_acc_values, 'r', label="Validation RMSE")
            plt.title("Training and Validation RMSE")
            plt.xlabel("Epochs")
            plt.xticks(epochs)
            plt.ylabel("RMSEs")
            plt.legend()
            plt.show()
        return model

    def plot_prediction(self, df, col_x, predicted_y, actual_y, name, time_frame, modelname):
        plt.figure(figsize=(11, 9))
        label1 = "Actual" + str(name) + "Value"
        label2 = "Predicted" + str(name) + "Value"
        title_value = str(modelname) + str(name) + \
            " : Actual vs Fitted Values for Reaction Time: " + str(time_frame)
        ax = sns.lineplot(x=df[col_x], y=df[actual_y], color="r", label=label1)
        sns.lineplot(x=df[col_x], y=df[predicted_y],
                     color="b", label=label2, ci=None)
        plt.title(title_value)
        plt.show()
        plt.close()
        return None

    def fit_and_run_KNN(self, train_df, test_df, delta_time):
        shift_instance = delta_time*10

        train_df, test_df, X_train, y_train, X_test, y_test, val_df, X_val, y_val = self.preprocessing(
            train_df, test_df, shift_instance)

        model = self.define_fit_KNN(X_train, y_train)
        model_name = 'knn' + str(delta_time) + '.pkg'
        pickle.dump(model, open(model_name, 'wb'))

        predict_on_pair = self.prediction_test_pairs(test_df, 10, 12)
        predict_on_pair
        print(f"Prediction being done on :{predict_on_pair[0]}")
        target_variable = 'nextframeAcc'
        predicted_data = self.prediction(
            test_df, predict_on_pair, target_variable, model, delta_time)
        prediction_1 = predicted_data[predicted_data["L-F_Pair"]
                                      == predict_on_pair[0]]
        self.display_prediction_plots(prediction_1, delta_time, 'KNN ')

        return train_df, test_df, X_train, y_train, X_test, y_test, predicted_data, model

    def define_fit_KNN(self, X_train, y_train):
        model = KNeighborsRegressor(n_neighbors=5)
        model.fit(X_train, y_train)

        return model

    def fit_and_run_Random_Forest(self, train_df, test_df, delta_time, number_of_estimators):
        shift_instance = delta_time*10

        train_df, test_df, X_train, y_train, X_test, y_test, val_df, X_val, y_val = self.preprocessing(
            train_df, test_df, shift_instance)

        model = self.define_fit_RF(X_train, y_train, number_of_estimators)
        model_name = 'randomForest' + str(delta_time) + '.pkg'
        pickle.dump(model, open(model_name, 'wb'))

        predict_on_pair = self.prediction_test_pairs(test_df, 10, 12)
        # predict_on_pair
        print(f"Prediction being done on :{predict_on_pair[0]}")
        target_variable = 'nextframeAcc'
        predicted_data = self.prediction(
            test_df, predict_on_pair, target_variable, model, delta_time)
        prediction_1 = predicted_data[predicted_data["L-F_Pair"]
                                      == predict_on_pair[0]]
        self.display_prediction_plots(
            prediction_1, delta_time, 'Random Forest ')

        return train_df, test_df, X_train, y_train, X_test, y_test, predicted_data, model

    def define_fit_RF(self, X_train, y_train, number_of_estimators):
        model = RandomForestRegressor(
            n_estimators=number_of_estimators, n_jobs=-1)
        model.fit(X_train, y_train)

        return model

    def prediction(self, test_df, test_range, target_variable, model, time_frame):

        delta_time = 0.1
        predicted_df = []

        # this loop runs for each pair required predictions.
        for current_pair in test_range:

            # Assign shape of the predictions
            input_df = []
            input_df = test_df[test_df['L-F_Pair'] == current_pair]
            spacing = np.zeros(input_df.shape[0])
            local_y_subject = np.zeros(input_df.shape[0])
            local_y_preceding = np.zeros(input_df.shape[0])
            dv = np.zeros(input_df.shape[0])
            vel = np.zeros(input_df.shape[0])
            jerk = np.zeros(input_df.shape[0])
            pred_acc = np.zeros(input_df.shape[0])
            s_subject = np.zeros(input_df.shape[0])
            # updating the values for first Predictions
            vel[0] = input_df.iloc[0]['v_Vel']
            spacing[0] = input_df.iloc[0]['Rear_to_Front_Space_Headway']
            dv[0] = input_df.iloc[0]['Velocity Difference_Following-Preceding']
            jerk[0] = 0
            local_y_subject[0] = input_df.iloc[0]['Local_Y']

            local_y_preceding[0] = input_df.iloc[0]['preceding_Local_Y']
            preceding_vehicle_class = input_df.iloc[0]['preceding_v_Class']
            vehicle_class = input_df.iloc[0]['v_Class']
            length_preceding_vehicle = input_df.iloc[0]['preceding_vehicle_length']

            predict_for_input = np.array(
                [spacing[0], preceding_vehicle_class, vehicle_class, dv[0], vel[0]]).reshape(1, -1)
            pred_acc[0] = model.predict(predict_for_input)
            # print(
            #    f"j: {0} input:{predict_for_input},subject localy:{local_y_subject[0]},preceding_local_y:{local_y_preceding[0]},spacing:{spacing[0]} pred_acc: {pred_acc[0]}")
            vel[1] = vel[0]+(pred_acc[0] * delta_time)
            if vel[1] < 0:
                vel[1] = 0

            dv[1] = vel[1] - input_df.iloc[1]['preceding_Vehicle_Velocity']

            s_subject[0] = ((vel[0] * delta_time) +
                            (0.5*pred_acc[0]*pow(delta_time, 2)))

            #print(f"row 0=s_subject:{s_subject[0]}")
            local_y_subject[1] = local_y_subject[0] + s_subject[0]
            local_y_preceding[1] = input_df.iloc[1]['preceding_Local_Y']

            spacing[1] = local_y_preceding[1] - \
                local_y_subject[1] - length_preceding_vehicle

            for j in range(1, len(input_df)):
                predict_for_input = np.array(
                    [spacing[j], preceding_vehicle_class, vehicle_class, dv[j], vel[j]]).reshape(1, -1)

                pred_acc[j] = model.predict(predict_for_input)
                jerk[j] = (pred_acc[j] - pred_acc[j-1]) / delta_time
                if j == len(input_df)-1:
                    break

                vel[j+1] = vel[j]+(pred_acc[j]*delta_time)

                if vel[j+1] < 0:
                    vel[j+1] = 0

                dv[j+1] = vel[j+1] - input_df.iloc[j +
                                                   1]['preceding_Vehicle_Velocity']

                s_subject[j] = ((vel[j]*delta_time) +
                                (0.5*pred_acc[j]*pow(delta_time, 2)))

                local_y_subject[j+1] = local_y_subject[j] + s_subject[j]
                local_y_preceding[j+1] = input_df.iloc[j +
                                                       1]['preceding_Local_Y']

                spacing[j+1] = local_y_preceding[j+1] - \
                    local_y_subject[j+1] - length_preceding_vehicle

                # print(
                #    f"j: {j} input:{predict_for_input},subject localy:{local_y_subject[j]},preceding_local_y:{local_y_preceding[j]},spacing:{spacing[j]} pred_acc: {pred_acc[j]}")

            input_df['predicted_acceleration'] = pred_acc
            input_df['predicted_velocity'] = vel
            input_df['predicted_Local_Y'] = local_y_subject
            input_df['predicted_spacing'] = spacing
            input_df['predicted_jerk'] = jerk
            input_df['preceding_Local_Y_used'] = local_y_preceding
            input_df['s_subject'] = s_subject
            predicted_df.append(input_df)

        result = pd.concat(predicted_df)
        return result

    def prediction_preprocessing(self, df, time_frame):
        shift_instance = time_frame*10
        df = self.create_prediction_columns(df, shift_instance)

        return df

    def display_prediction_plots(self, prediction, delta_time, modelname):

        self.plot_prediction(prediction, 'pair_Time_Duration',
                             'predicted_acceleration', 'nextframeAcc', 'Acceleration', delta_time, modelname)
        self.plot_prediction(prediction, 'pair_Time_Duration',
                             'predicted_velocity', 'nextframesvel', 'Velocity', delta_time, modelname)
        self.plot_prediction(prediction, 'pair_Time_Duration',
                             'predicted_spacing', 'nextFrameSpacing', 'Spacing', delta_time, modelname)
        self.plot_prediction(prediction, 'pair_Time_Duration',
                             'predicted_jerk', 'nextframejerk', 'Jerk', delta_time, modelname)
        return None

    def accuracy(self, df, actual, predicted):
        R2_score = r2_score(df[actual], df[predicted])
        RMSE = np.sqrt(mean_squared_error(df[actual], df[predicted]))
        return R2_score, RMSE

    def predict_test_dataset(self, file_name, delta_time, model_name):
        string_delta_time = str(delta_time).replace('.', '_')
        file_name = file_name + string_delta_time
        print(
            f"Running test Set on :{file_name}, Reaction Time {delta_time} for Model: {model_name}")
        file = FileProcessing.FileProcessing()
        trajectory_display = file.read_input(file_name)
        target_variable = 'nextframeAcc'
        if model_name == 'neural_network_model':
            model = file.read_model(model_name, delta_time, neural=True)
        else:
            model = file.read_model(model_name, delta_time)
        predict_data = self.prediction_preprocessing(
            trajectory_display, delta_time)
        predict_on_pair = self.prediction_test_pairs(predict_data, 0, 9999)
        print(
            f"Total number of unique Pairs in Test Dataset: {len(predict_on_pair)}")
        predicted_data = self.prediction(
            predict_data, predict_on_pair, target_variable, model, delta_time)
        # predicted_data.columns
        r_square, rmse = self.accuracy(
            predicted_data, 'nextframeAcc', 'predicted_acceleration')
        print(f"\n")
        print(f"{model_name}, Reaction Time:{delta_time} Statistics Below:")
        print(f"r_square: {r_square}")
        print(f"rmse: {rmse}")
        prediction_1 = predicted_data[predicted_data["L-F_Pair"]
                                      == predict_on_pair[0]]
        self.display_prediction_plots(
            prediction_1, delta_time, model_name)

        predicted_data_file_name = model_name + '_' + \
            'predicted_Test_Set_' + str(delta_time)
        file.export_file(predicted_data, predicted_data_file_name)

        return r_square, rmse

    def predict_knn_rf_cnn(self, predict_data, predict_on_pair, delta_time):
        model_name = 'knn'
        target_variable = 'nextframeAcc'
        model = self.file.read_model(model_name, delta_time)
        predicted_data = self.prediction(
            predict_data, predict_on_pair, target_variable, model, delta_time)
        predicted_data.rename(columns={'predicted_acceleration': 'knn_predicted_acceleration',
                                       'predicted_velocity': 'knn_predicted_velocity',
                                       'predicted_Local_Y': 'knn_predicted_Local_Y',
                                       'predicted_jerk': 'knn_predicted_jerk',
                                       'predicted_spacing': 'knn_predicted_spacing'}, inplace=True)
        model_name = 'randomForest'
        model = self.file.read_model(model_name, delta_time)
        predicted_data = self.prediction(
            predicted_data, predict_on_pair, target_variable, model, delta_time)
        predicted_data.rename(columns={'predicted_acceleration': 'rf_predicted_acceleration',
                                       'predicted_velocity': 'rf_predicted_velocity',
                                       'predicted_Local_Y': 'rf_predicted_Local_Y',
                                       'predicted_jerk': 'rf_predicted_jerk',
                                       'predicted_spacing': 'rf_predicted_spacing'}, inplace=True)
        model_name = 'neural_network_model'
        model = self.file.read_model(model_name, delta_time, neural=True)
        predicted_data = self.prediction(
            predicted_data, predict_on_pair, target_variable, model, delta_time)
        predicted_data.rename(columns={'predicted_acceleration': 'cnn_predicted_acceleration',
                                       'predicted_velocity': 'cnn_predicted_velocity',
                                       'predicted_Local_Y': 'cnn_predicted_Local_Y',
                                       'predicted_jerk': 'cnn_predicted_jerk',
                                       'predicted_spacing': 'cnn_predicted_spacing'}, inplace=True)
        return predicted_data

    def plot_all_predictions(self, prediction_1, delta_time):

        prediction_1["Time Duration (s)"] = prediction_1["pair_Time_Duration"]
        prediction_1["Velocity (m/s)"] = prediction_1["nextframesvel"]
        prediction_1["Acceleration (m/s^2)"] = prediction_1["nextframeAcc"]
        prediction_1["Jerk (m/s^3)"] = prediction_1["nextframejerk"]
        prediction_1["Space (m)"] = prediction_1["nextFrameSpacing"]
        self.combined_predicted_plots(prediction_1, 'Time Duration (s)', 'knn_predicted_acceleration', 'rf_predicted_acceleration', 'cnn_predicted_acceleration',
                                      'Acceleration (m/s^2)', 'Acceleration', delta_time, '', 'KNN')
        self.combined_predicted_plots(prediction_1, 'Time Duration (s)', 'knn_predicted_acceleration', 'rf_predicted_acceleration', 'cnn_predicted_acceleration',
                                      'Acceleration (m/s^2)', 'Acceleration', delta_time, '', 'RF')
        self.combined_predicted_plots(prediction_1, 'Time Duration (s)', 'knn_predicted_acceleration', 'rf_predicted_acceleration', 'cnn_predicted_acceleration',
                                      'Acceleration (m/s^2)', 'Acceleration', delta_time, '', 'CNN')

        self.combined_predicted_plots(prediction_1, 'Time Duration (s)', 'knn_predicted_velocity', 'rf_predicted_velocity', 'cnn_predicted_velocity',
                                      'Velocity (m/s)', 'Velocity', delta_time, '')
        self.combined_predicted_plots(prediction_1, 'Time Duration (s)', 'knn_predicted_spacing', 'rf_predicted_spacing', 'cnn_predicted_spacing',
                                      'Space (m)', 'Space', delta_time, '')
        self.combined_predicted_plots(prediction_1, 'Time Duration (s)', 'knn_predicted_jerk', 'rf_predicted_jerk', 'cnn_predicted_jerk',
                                      'Jerk (m/s^3)', 'Jerk', delta_time, '', 'KNN')
        self.combined_predicted_plots(prediction_1, 'Time Duration (s)', 'knn_predicted_jerk', 'rf_predicted_jerk', 'cnn_predicted_jerk',
                                      'Jerk (m/s^3)', 'Jerk', delta_time, '', 'RF')
        self.combined_predicted_plots(prediction_1, 'Time Duration (s)', 'knn_predicted_jerk', 'rf_predicted_jerk', 'cnn_predicted_jerk',
                                      'Jerk (m/s^3)', 'Jerk', delta_time, '', 'CNN')

        return None

    def combined_predicted_plots(self, df, col_x, predicted_y_knn, predicted_y_rf, predicted_y_cnn, actual_y, name, time_frame, modelname, type=[]):
        plt.figure(figsize=(15, 9))
        label1 = "Actual " + str(name)
        predicted_y1 = 'KNN ' + " Prediction"
        predicted_y2 = 'RF ' + " Prediction"
        predicted_y3 = 'CNN ' + " Prediction"
        title_value = str(modelname) + str(name) + \
            " : Actual vs Fitted Values for Reaction Time: " + str(time_frame)
        plt.title(title_value, color="black", size=18)
        plt.xticks(range(0, 60, 5))
        plt.rc('xtick', labelsize=15)
        plt.rc('ytick', labelsize=15)
        plt.xlabel('Time Duration (s)', color="black", size=18)

        #plt.ylabel("Velocity (m/s)", color = "black", size = 18)
        ax = sns.lineplot(x=df[col_x], y=df[actual_y],
                          color="black", label=label1)
        if len(type) > 0:
            if type == 'KNN':
                sns.lineplot(x=df[col_x], y=df[predicted_y_knn],
                             color="b", label=predicted_y1, ci=None)
            elif type == 'RF':
                sns.lineplot(x=df[col_x], y=df[predicted_y_rf],
                             color="g", label=predicted_y2, ci=None)
            elif type == 'CNN':
                sns.lineplot(x=df[col_x], y=df[predicted_y_cnn],
                             color="orange", label=predicted_y3, ci=None)
        else:
            sns.lineplot(x=df[col_x], y=df[predicted_y_knn],
                         color="b", label=predicted_y1, ci=None)
            sns.lineplot(x=df[col_x], y=df[predicted_y_rf],
                         color="g", label=predicted_y2, ci=None)
            sns.lineplot(x=df[col_x], y=df[predicted_y_cnn],
                         color="orange", label=predicted_y3, ci=None)
            plt.ylabel(predicted_y3, color="black", size=18)
        plt.ylabel(actual_y, color="black", size=18)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.tick_params(left=False, bottom=False)
        plt.title(title_value)
        plt.show()
        plt.close()
        return None
