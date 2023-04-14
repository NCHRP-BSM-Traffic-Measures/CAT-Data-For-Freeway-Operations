import pandas as pd
import numpy as np
import json
import queue_fx_new4 as qfx

# Use Control File to Point to Input Data
control_file = 'control.json'
with open(control_file) as in_f:
	control = json.load(in_f)

# Read in data files
veh_len_df = qfx.read_veh_lengths_file(control['veh_lengths_filepath'])
stoplines_df = qfx.read_stoplines_file(control['stoplines_filepath'])
up_and_downstream_ramp_links_df = qfx.read_up_and_downstream_ramp_links(control['neighboring_links_filepath'])
sensorIDs_df = qfx.read_sensorIDs(control['sensorIDs_input_filepath'])
neighborIDs_df = qfx.read_neighborSensorIDs(control['neighboringSensors_filepath'])
all_ramp_lanes_df = qfx.read_all_ramp_lanes(control['all_ramp_lanes_filepath'])
y_df = qfx.format_queues(qfx.read_max_queues_Y_file(control['ground_truth_filepath']))
sensor_df = qfx.read_sensor_data_offline(control['flow_data_filepath'], control['occupancy_data_filepath'])
df = qfx.read_BSMs_file(control['bsm_data_filepath'])

# create the avg stoplines X,Y df
stopline_avg_df = qfx.create_avg_stoplines_df(stoplines_df)

# format the time
df.transtime = df.transtime.apply(qfx.format_result)
# create a new column that assigns BSM to 30 second time interval
df['transtime_30sec'] = df['transtime'].dt.floor('30S')

# join columns from veh len to main BSMs df 
df = qfx.join_veh_len_to_BSM_df(df, veh_len_df)

# add columns to neighborIDs with the Ramp_ID, upstream_mainline_ID, and downstream_mainline_ID
neighborIDs_new_df = qfx.join_rampID_to_neighborSensorsID(neighborIDs_df, sensorIDs_df)

# create list of the upstream and downstream mainline links for the on-ramps for the BSMs (different from the sensors)
UpLinkIDs_list = list(up_and_downstream_ramp_links_df['Upstream_LinkID'])
DownLinkIDs_list = list(up_and_downstream_ramp_links_df['Downstream_LinkID'])
upDownLinkIDs_list = list(set(UpLinkIDs_list).union(set(DownLinkIDs_list)))

# continue with assigning BSMS to appropriate on-ramp, identifying upstream and downstream mainline links, on-ramp sensor, and upstream and downstream sensors
# on the link (edge)
df_on_ramps = qfx.assign_BSMs_to_ramp_links(df, stoplines_df, up_and_downstream_ramp_links_df, neighborIDs_new_df)

# create df of BSMs just for the upstream and downstream mainline links for the on-ramps
df_up_downstream_ramps = qfx.assign_BSMs_to_ramp_up_and_downstream_links(df, upDownLinkIDs_list)

# Engineer the aggregated BSM features by assigned on-ramp, 30 secs, and lane
base_df, base1_df = qfx.feature_engineering(df_on_ramps, df_up_downstream_ramps, stopline_avg_df)

# Join all features and labels
df_xy = qfx.join_features_and_labels(base_df, y_df, base1_df, sensor_df, up_and_downstream_ramp_links_df, neighborIDs_new_df)

# Add columns to the features for the previous 4 time steps' queue count, occupancy, upstream mainline link's occupancy, and 
# downstream mainline link's occupancy for each ramp
df_xy = qfx.add_previous_time_queue_count_col(df_xy)

# Drop 'Ramp_Sensor_ID','Immediately_upstream_mainline_sensor_ID', and'Immediately_downstream_mainline_sensor_ID' columns since they are just labels
df_xy = df_xy.drop(['Ramp_Sensor_ID','Immediately_upstream_mainline_sensor_ID','Immediately_downstream_mainline_sensor_ID'], axis=1)

# Handle any missing values and encode_categorical_features
df_xy = qfx.label_encode_categorical_features(qfx.handle_missing_data(df_xy, df_on_ramps))

# Write processed data to csv
df_xy.to_csv(control['data_for_ML_filepath'], index=False)
