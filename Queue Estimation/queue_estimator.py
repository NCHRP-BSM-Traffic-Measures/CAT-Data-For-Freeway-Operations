import pandas as pd
import numpy as np
import json
from xgboost import XGBRegressor
import queue_fx_new4 as qfx
from datetime import timedelta

"""
Create predicted queue estimations from BSM and sensor data by 30 second timestep. Requires json control file
and csvs for vehicle lengths, stoplines, neighboring links, sensorIds and neighboring sensors. 
"""

class QueueEstimator(object):
    def __init__(self, control_file):
        """
        Define the queue estimator with the given json control file, 
        """
        with open(control_file) as in_f:
            self.control = json.load(in_f)

        # Gives initialized data for sensors, empty dataframe at beginning
        self.sensor_data = self.initialize_sensor_data()

        # Load in static data as dataframes
        try:
            self.veh_len_df = qfx.read_veh_lengths_file(self.control['veh_lengths_filepath'])
            self.stoplines_df = qfx.read_stoplines_file(self.control['stoplines_filepath'])
            self.up_and_downstream_ramp_links_df = qfx.read_up_and_downstream_ramp_links(self.control['neighboring_links_filepath'])
            self.sensorIDs_df = qfx.read_sensorIDs(self.control['sensorIDs_input_filepath'])
            self.neighborIDs_df = qfx.read_neighborSensorIDs(self.control['neighboringSensors_filepath'])
            self.all_ramp_lanes_df = qfx.read_all_ramp_lanes(self.control['all_ramp_lanes_filepath'])
        except (TypeError, KeyError):
            print('Static data files could not be properly loaded.')

        # Gives initialed data of predictions, assuming zero at all ramps
        self.y_df = self.initialize_queue_predictions()

        # Load in XGB model
        self.model_xgb = qfx.import_trained_XGB_model(self.control['xgb_model_filepath'])

        # create the avg stoplines X,Y df
        self.stopline_avg_df = qfx.create_avg_stoplines_df(self.stoplines_df)

        # add columns to neighborIDs with the Ramp_ID, upstream_mainline_ID, and downstream_mainline_ID
        self.neighborIDs_new_df = qfx.join_rampID_to_neighborSensorsID(self.neighborIDs_df, self.sensorIDs_df)

        # create list of the upstream and downstream mainline links for the on-ramps for the BSMs (different from the sensors)
        self.UpLinkIDs_list = list(self.up_and_downstream_ramp_links_df['Upstream_LinkID'])
        self.DownLinkIDs_list = list(self.up_and_downstream_ramp_links_df['Downstream_LinkID'])
        self.upDownLinkIDs_list = list(set(self.UpLinkIDs_list).union(set(self.DownLinkIDs_list)))

    def process_time_step(self, bsm_dict, occupancy_dict, flow_dict, timestep_start_seconds):
        """
        Main method that processes a given time step of bsm and sensor data and returns ramp queue predictions.
        Predictions are given as a list of dictionaries, with each dictionary being a row of data.
        Keys to these dictionaries are ramp, lane, time_30, queue_count_max (estimated queue count), and queue_len_max (estimated queue length)
        ::bsm_dict:: a list of dictionaries, each representing a processed bsm message with keys of bsm_tmp_id, time
            Speed, X, Y, transtime, transTo, Acceleration, brakeStatus, brakePressure, hardBraking, transmission_received_time
            Heading, link, lane, type
        ::occupancy_dict:: a dictionary with the sensor id as keys and occupancy as values
        ::flow_dict:: a dictionary with the sensor id as keys and flow as values
        ::timestep_start_seconds:: Beginning time step in seconds from start of simulation. Should be multiple of 30.
        """

        # Read in bsm data
        df = qfx.read_BSM_data(bsm_dict)
        df = df.rename(columns={'Time': 'time', 'speed': 'Speed', 'x': 'X', 'y': 'Y', 
            'acceleration': 'Acceleration', 'heading': 'Heading', 'type': 'Type'})

        # format the time
        df.transtime = df.transtime.apply(qfx.format_result)

        # create a new column that assigns BSM to 30 second time interval
        df['transtime_30sec'] = df['transtime'].dt.floor('30S')

        # join columns from veh len to main BSMs df 
        df = qfx.join_veh_len_to_BSM_df(df, self.veh_len_df)

        # continue with assigning BSMS to appropriate on-ramp, identifying upstream and downstream mainline links, 
        # on-ramp sensor, and upstream and downstream sensors on the link (edge)
        df_on_ramps = qfx.assign_BSMs_to_ramp_links(df, self.stoplines_df, self.up_and_downstream_ramp_links_df, self.neighborIDs_new_df)

        # create df of BSMs just for the upstream and downstream mainline links for the on-ramps
        df_up_downstream_ramps = qfx.assign_BSMs_to_ramp_up_and_downstream_links(df, self.upDownLinkIDs_list)

        # Read sensor data using flow dictionary, occupancy dictionary, and start of 30 sec timestep in seconds
        sensor_data_avg = qfx.read_sensor_data_live(flow_dict, occupancy_dict, start_seconds=timestep_start_seconds)

        # Concatinate new sensor data with loaded sensor data
        self.sensor_data = pd.concat([self.sensor_data, sensor_data_avg])

        # Only need 5 timesteps of sensor data in memory, pop earliest timestep if there are more than five
        self.pop_sensor_data()

        # Engineer the aggregated BSM features by assigned on-ramp, 30 secs, and lane base_df, base1_df, 
        base_df, base1_df = qfx.feature_engineering(df_on_ramps, df_up_downstream_ramps, self.stopline_avg_df)

        # Get dummy y values for most recent timestep given by BSM data
        y_df_new = qfx.y_dummy_append(self.y_df, self.all_ramp_lanes_df, start_seconds=timestep_start_seconds)

        # Join all features and labels
        df_xy = qfx.join_features_and_labels(base_df, y_df_new, base1_df, self.sensor_data, self.up_and_downstream_ramp_links_df, self.neighborIDs_new_df)

        # Add columns to the features for the previous 4 time steps' queue count, occupancy, upstream mainline link's occupancy, and 
        # downstream mainline link's occupancy for each ramp
        df_xy = qfx.add_previous_time_queue_count_col(df_xy)

        # Drop 'Ramp_Sensor_ID','Immediately_upstream_mainline_sensor_ID', and'Immediately_downstream_mainline_sensor_ID' columns since they are just labels
        df_xy = df_xy.drop(['Ramp_Sensor_ID','Immediately_upstream_mainline_sensor_ID','Immediately_downstream_mainline_sensor_ID'], axis=1)
        
        # Handle any missing values and encode_categorical_features
        df_xy = qfx.label_encode_categorical_features(qfx.handle_missing_data(df_xy, df_on_ramps))

        # split into X features and y labels
        X,y = qfx.split_into_X_and_Y(df_xy)

        # Make predictions using xgb model
        y_pred = self.model_xgb.predict(X)

        # Cannot be negative values of queue
        y_pred = [0 if y < 0 else y for y in y_pred]

        # Replace by replacing dummy values (-1) of y with predicted values
        y_updated = qfx.replace_dummy_y_with_predictions(y, y_pred)

        # Update original df_xy with new predictions
        df_xy_update = qfx.update_dfxy_with_predictions(df_xy, y_updated)

        # Calculate predicted queue lengths from predicted queue counts and avg vehicle length
        df_xy_update1 = qfx.derive_queue_len_from_count(df_xy_update)

        # Change to format needed to continue cycling through predictions
        y_updated2 = qfx.reformat_df_predictions(df_xy_update1, self.control['ramp_dictionary_filepath'])

        # Update y data by appending new predictions, only keeping 4 timesteps in data frame
        self.pop_y_data(y_updated2, timestep_start_seconds)

        # Only return data from the prediction of the current timestep
        y_updated3 = qfx.y_data_timestep(y_updated2, timestep_start_seconds)

        # Return predictions as dictionary
        return y_updated3.to_dict(orient='records')
        
    def pop_sensor_data(self):
        '''Checks to see if sensor data has more than 5 time steps. If so, returns only the most recent 5'''

        time_steps = list(set(self.sensor_data['time_30']))
        time_steps.sort()
        if len(time_steps) > 5:
            last_5 = time_steps[-5]
            self.sensor_data = self.sensor_data[self.sensor_data.time_30 >= last_5]

    def pop_y_data(self, y_update, start_second):
        '''Appends new updated predictions to predictions data
        Checks to see if predictions data has more than 4 time steps. If so, returns only the most recent 4 

        Keyword arguments:
        y_update -- new y predictions 
        start_second -- timestep start in seconds'''

        y_update = y_update.rename(columns={'time_30':'time'})
        y_update['queue_indicator'] = 0
        y_update['queue_count_binned'] = 0
        new_time = str(timedelta(seconds = int(start_second), hours = 13))

        if start_second == 0:
            self.y_df = y_update
        else:
            y_update = y_update[y_update.time==new_time]
            new_y = pd.concat([self.y_df, y_update])
            time_steps = list(set(new_y['time']))
            time_steps.sort()
            if len(time_steps) > 4:
                last_4 = time_steps[-4]
                self.y_df = new_y[new_y.time >= last_4]
                self.y_df = self.y_df.drop_duplicates().reset_index(drop=True)
            else:
                self.y_df = new_y
                self.y_df = self.y_df.drop_duplicates().reset_index(drop=True)

    def initialize_sensor_data(self):
        '''Creates data which will track occupancy and flow data
        '''
        columns = ['sensorID', 'flow', 'occupancy', 'time_30']
        df_sensor = pd.DataFrame(columns=columns)
        return df_sensor

    def initialize_queue_predictions(self):
        '''
        Initializes queue predictions dataframe
        '''

        # Start with all ramp/lane combos
        initial_queues = self.all_ramp_lanes_df.copy()

        # Starts at 1 pm
        initial_queues['time'] = '13:00:00'

        # Give -1 as dummy value for queue measures
        initial_queues['queue_count_max'] = -1
        initial_queues['queue_len_max'] = -1
        
        return initial_queues
