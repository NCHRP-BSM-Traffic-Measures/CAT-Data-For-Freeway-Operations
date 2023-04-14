"""This python script contains the libraries and functions needed to run queue_estimator.py and Process_Data_For_Training.py
"""
# load libraries 
import numpy as np
import pandas as pd
from datetime import timedelta
from xgboost import XGBRegressor
from sklearn import metrics
from sklearn import preprocessing
import joblib
import time
import json


''' Queue values from definition.
    Can be changed if altering definitions of queue.
'''
QUEUE_START_SPEED = 0.00 
QUEUE_FOLLOWING_SPEED = (10.0 *0.681818) # convert ft/sec to mph
QUEUE_HEADWAY_DISTANCE = 20.0 #ft, in queue definition
QUEUE_DISTANCE_WITHIN_STOP_POINT = 20 #ft 

# 
CAR_LENGTH_AVG = 16.4041995 # in ft

def convert_meters_to_feet(value_in_meters):
    """Convert values in meters to values in feet

    Keyword arguments:
    value_in_meters -- meters to be converted to ft
    """
    value_in_ft = value_in_meters/0.3048
    return value_in_ft

def read_BSM_data(bsm_data):
    ''' Read BSMs as list of dictionaries, returns as pandas dataframe

    Keyword arguments:
    bsm_data -- dictionary object that comes from bsm_emulator'''
    df = pd.DataFrame(bsm_data)
    return df

def read_veh_lengths_file(veh_lengths_filename):
    """Read vehicle lengths by type csv file and store in pandas dataframe (veh_len_df)

    Keyword arguments:
    veh_lengths_filename -- filename for vehicle lengths csv
    """
    veh_len_df = pd.read_csv(veh_lengths_filename)
    return veh_len_df

def read_stoplines_file(stoplines_filename):
    """Read stop lines by lane and link csv file and store in pandas dataframe (stopline_df)

    Keyword arguments:
    stoplines_filename -- filename for stoplines csv
    """
    stoplines_df = pd.read_csv(stoplines_filename)
    # drop lat and lon columns
    stoplines_df = stoplines_df.drop(columns =['lat', 'lon'])
    
    # convert stoplines_df values from meters to feet and then drop columns that are in meters
    stoplines_df['stopline_X'] = convert_meters_to_feet(stoplines_df['stopline_X'])
    stoplines_df['stopline_Y'] = convert_meters_to_feet(stoplines_df['stopline_Y'])
    stoplines_df['ramp_length_ft'] = convert_meters_to_feet(stoplines_df['ramp_length_m'])
    stoplines_df['edge_length_ft'] = convert_meters_to_feet(stoplines_df['edge_length_m'])
    stoplines_df['ramp_width_ft'] = convert_meters_to_feet(stoplines_df['ramp_width_m'])
    stoplines_df['dist_to_stop_bar'] = convert_meters_to_feet(stoplines_df['dist_to_stop_bar'])
    stoplines_df = stoplines_df.drop(columns =['ramp_length_m', 'edge_length_m', 'ramp_width_m'])
    
    return stoplines_df

def create_avg_stoplines_df(stoplines_df_name):
    """Create an aggregated stoplines_avg_df with the average stopline X, Y for each ramp and all lanes since 
    they are the same for all lanes in the test case 

    Keyword arguments:
    stoplines_df_name -- stopline df type created by read_stoplines_file
    """
    
    stopline_avg_df = stoplines_df_name.groupby(['Ramp_ID'])['stopline_X'].mean().reset_index(name='mean_X')
    stopline_avg_df['mean_Y'] = stoplines_df_name.groupby(['Ramp_ID'])['stopline_Y'].mean().reset_index(name='mean_Y').iloc[:,1]
    stopline_avg_df['n_lanes'] = stoplines_df_name.groupby('Ramp_ID')['Lane'].nunique().reset_index().iloc[:,1]
    
    return stopline_avg_df

def read_up_and_downstream_ramp_links(neighboringLinks_filename):
    """Read upstream and downstream links for each on-ramp file and store in pandas dataframe

    Keyword arguments:
    neighboringLinks_filename -- filename for neighboring links csv
    """
    up_and_downstream_ramp_links_df = pd.read_csv(neighboringLinks_filename)
    
    return up_and_downstream_ramp_links_df
    
def assign_BSMs_to_ramp_links(df_BSM_name, stoplines_filename, up_and_downstream_ramp_links_df_name, neighborIDs_new_df_name):
    """Use the BSM link and the stoplines to assign BSMs to ramps, then use BSM ramp ID and neighbors to assign upstream and
    downstream mainline link ids, and then use sensor neighborIDs to assign the on-ramp sensor, upstream sensor, and downstream
    sensor IDs

    Keyword arguments:
    df_BSM_name -- df holding BSM data for timestep
    stoplines_filename -- df stoplines file
    up_and_downstream_ramp_links_df_name -- df up and downstream ramp links
    neighborIDs_new_df_name -- df holding IDs of neighbors

    """ 
    df = df_BSM_name.merge(stoplines_filename[['Ramp_ID', 'Link_ID']], how='left', left_on='link', right_on='Link_ID')
    df = df.drop(['Link_ID'], axis=1)
    df = df.dropna()
    
    df = df.merge(up_and_downstream_ramp_links_df_name[['RampID','Upstream_LinkID','Downstream_LinkID']], how='left',
                 left_on='Ramp_ID', right_on='RampID')
    df = df.drop(['RampID'], axis = 1)
    
    df = df.merge(neighborIDs_new_df_name[['Ramp_Sensor_ID', 'Immediately_upstream_mainline_sensor_ID','Immediately_downstream_mainline_sensor_ID',
                                          'Ramp']], how = 'left', left_on = 'Ramp_ID', right_on = 'Ramp')
    df = df.drop(['Ramp'], axis = 1)
    
    return df

def assign_BSMs_to_ramp_up_and_downstream_links(df_BSM_name, upDownLinkIDs_list_name):
    """Use the BSM link and the list of all on-ramps' upstream and downstream links to subset BSMs to just include 
    relevant ones

    Keyword arguments:
    df_BSM_name -- df holding BSM data for timestep
    upDownLinkIDs_list_name -- list with up and downstream link IDs
    """
    df = df_BSM_name.loc[df_BSM_name['link'].isin(upDownLinkIDs_list_name)]
    return df
    
def read_sensorIDs(sensorIDs_filename):
    """Read sensor IDs for by relevant link/edge and ramp ID csv file and store in pandas dataframe (sensorIDs_df)

    Keyword arguments:
    sensorIDs_filename -- filename for sensor IDs"""

    sensorIDs_df = pd.read_csv(sensorIDs_filename)
    return sensorIDs_df

def read_neighborSensorIDs(neighborIDs_filename):
    """Read neighboring upstream and downstream sensor IDs for each ramp ID csv file and store in pandas dataframe (neighborIDs_df)

    Keyword arguments:
    neighborIDs_filename -- filename for neighbor IDs"""
    neighborIDs_df = pd.read_csv(neighborIDs_filename)
    return neighborIDs_df

def read_all_ramp_lanes(all_ramp_lanes_filename):
    """ Read pandas dataframe for all ramp/lane combinations 
    Keyword arguments:
    all_ramp_lanes_filename -- filename for all ramps and lane combos"""

    ramp_lanes_df = pd.read_csv(all_ramp_lanes_filename)
    return ramp_lanes_df

def join_rampID_to_neighborSensorsID(neighborIDs_df_name, sensorIDs_df_name):
    """Join the ramp sensor ID, upstream mainline link ID, and downstream mainline link ID
    from neighbor IDs df with the ramp ID, upstream ID, and downstream ID from the sensorIDs df

    Keyword arguments:
    neighborIDs_df_name -- df with neighbor IDs
    sensorIDs_df_name -- df with sensor IDs"""

    # Join ramp ID column to neighbor IDs df Ramp_Sensor_ID
    new_neighborIDs_df = neighborIDs_df_name.merge(sensorIDs_df_name[['Sensor_ID','Ramp']], how='left', left_on='Ramp_Sensor_ID', right_on='Sensor_ID')
    new_neighborIDs_df = new_neighborIDs_df.drop(['Sensor_ID'], axis=1)
    
    # Join upstream ID column to new neighbor IDs df and Link_Edge_ID
    # Immediately_upstream_mainline_sensor_ID
    new_neighborIDs_df = new_neighborIDs_df.merge(sensorIDs_df_name[['Sensor_ID','Link_Edge_ID']], how='left', left_on='Immediately_upstream_mainline_sensor_ID', right_on='Sensor_ID')
    new_neighborIDs_df = new_neighborIDs_df.drop(['Sensor_ID'], axis=1)
    new_neighborIDs_df = new_neighborIDs_df.rename(columns = {'Link_Edge_ID':'upstream_mainline_link_ID'})
    
    # Join downstream ID column to new neighbor IDs df and Link_Edge_ID
    # Immediately_downstream_mainline_sensor_ID
    new_neighborIDs_df = new_neighborIDs_df.merge(sensorIDs_df_name[['Sensor_ID','Link_Edge_ID']], how='left', left_on='Immediately_downstream_mainline_sensor_ID', right_on='Sensor_ID')
    new_neighborIDs_df = new_neighborIDs_df.drop(['Sensor_ID'], axis=1)
    new_neighborIDs_df = new_neighborIDs_df.rename(columns = {'Link_Edge_ID':'downstream_mainline_link_ID'})
    
    return new_neighborIDs_df

def format_result(result):
    """Format result of simulation time float into datetime, 
    add 13 hours for I-210 EB simulation data because 0.0 corresponds to 1:00 PM,
    output is time in HH:MM:SS.microseconds

    Keyword arguments:
    result -- time float field """

    seconds = int(result)
    microseconds = (result * 1000000) % 1000000
    output = timedelta(0, seconds, microseconds) + timedelta(hours=13)
    return output

def distance_between(origin_x, origin_y, destination_x, destination_y):
    """Calculate the distance between two points. This distance does not account for the on-ramp roadway curvature

    Keyword arguments:
    origin_x -- x coordinate for BSM
    origin_y -- y coordinate for BSM
    destination_x -- x coordinate for stop bar
    destination_y -- y coordinate for stop bar
    """

    return ((origin_x - destination_x)**2 + (origin_y - destination_y)**2)**.5

def min_dist_to_avg_stopbar(group_row, stopline_avg_df_name):
    """Calculate the distance between each grouping min X,Y and the avg stopline X,Y for that on-ramp.
    This distance is an approximation and depends on the direction and curvature of the on-ramp.

    Keyword arguments:
    group_row -- grouping of BSMs
    stopline_avg_df_name -- df containing stopline averages
    """

    row_stop_X, row_stop_Y = stopline_avg_df_name.loc[stopline_avg_df_name['Ramp_ID']==group_row['Ramp_ID'],['mean_X','mean_Y']].values[0]
    row_dist = distance_between(group_row['max_X'], group_row['max_Y'], row_stop_X, row_stop_Y)
    return(row_dist)

def max_dist_to_avg_stopbar(group_row, stopline_avg_df_name):
    """Calculate the max distance between each grouping max X,Y and the avg stopline X,Y for that on-ramp.
    This distance is an approximation and depends on the direction and curvature of the on-ramp.

    Keyword arguments:
    group_row -- grouping of BSMs
    stopline_avg_df_name -- df containing stopline averages
    """

    row_stop_X, row_stop_Y = stopline_avg_df_name.loc[stopline_avg_df_name['Ramp_ID']==group_row['Ramp_ID'],['mean_X','mean_Y']].values[0]
    row_dist = distance_between(group_row['min_X'], group_row['min_Y'], row_stop_X, row_stop_Y)
    return(row_dist)
    
def to_seconds(s):
    """Convert the 30 second time string to a float

    Keyword arguments:
    s -- seconds as string"""

    hr, minute, sec = [float(x) for x in s.split(':')]
    total_seconds = hr*3600 + minute*60 + sec
    return total_seconds

def join_veh_len_to_BSM_df(df_BSM_name, veh_len_df_name):
    """Join vehicle length column to main BSM df
    
    Keyword arguments:
    df_BSM_name -- df containing BSM group data
    veh_len_df_name -- df containing vehicle lengths
    """

    df = df_BSM_name.merge(veh_len_df_name[['Type_ID','Length (ft)']], how='left', left_on='Type', right_on='Type_ID')
    df = df.drop(['Type_ID'], axis=1)
    return df

def join_up_and_downstream_linkIDs_to_base_df(up_and_downstream_ramp_links_df_name, base_df_name):
    """Join upstream_linkID and downstream_linkID with base_df

    Keyword arguments:
    up_and_downstream_ramp_links_df_name -- df containing upstream and downstream road links
    base_df_name -- df used for connecting ramps to road links
    """
    base2_df = base_df_name.merge(up_and_downstream_ramp_links_df_name[['RampID', 'Upstream_LinkID','Downstream_LinkID']], how = 'left',#how = 'inner'
                                 left_on ='ramp', right_on = 'RampID')
    base2_df = base2_df.drop(['RampID'], axis=1)
    return base2_df

def join_base1_df_to_df_xy(df_xy_name, base1_df_name):
    """Join new features from base1_df (upstream and downstream links from on-ramp) with base_df (for on-ramp)

    Keyword arguments:
    df_xy_name -- df used for connecting ramps to road links
    base1_df_name -- df used for connecting ramps to road links
    """
    base2_df = df_xy_name.copy()


    ''' If empty dataframe for base1_df (no upstream or downstream bsms) just add empty columns and return a df full of NA values.
        This happens in some cases with low CV market penetration and uneven ingress of BSMs.
    '''
    extra_columns = ['upstream_bsm_tmp_id_Count',
           'upstream_num_BSMs_0speed', 'upstream_num_BSMs_0_to_following_speed',
           'upstream_num_BSMs_above_following_speed', 'upstream_speed_stddev',
           'upstream_speed_max', 'upstream_accel_stddev',
           'upstream_num_BSMs_neg_accel', 'downstream_bsm_tmp_id_Count',
           'downstream_num_BSMs_0speed',
           'downstream_num_BSMs_0_to_following_speed',
           'downstream_num_BSMs_above_following_speed', 'downstream_speed_stddev',
           'downstream_speed_max', 'downstream_accel_stddev',
           'downstream_num_BSMs_neg_accel']

    # Return NA values df
    if base1_df_name.empty:
        for x in extra_columns:
            base2_df[x] = np.nan
        return base2_df

    # Procedure when BSMs are ingressed normally
    else:
        '''make two new dataframes from base1_df_name, one for upstream links and one for downstream links and then rename columns'''
        upstream_temp_df = base1_df_name.copy()
        upstream_temp_df = upstream_temp_df.rename(columns = {'bsm_tmp_id_Count':'upstream_bsm_tmp_id_Count', 
                                                             'num_BSMs_0speed':'upstream_num_BSMs_0speed',
                                                             'num_BSMs_0_to_following_speed': 'upstream_num_BSMs_0_to_following_speed',
                                                             'num_BSMs_above_following_speed':'upstream_num_BSMs_above_following_speed',
                                                             'speed_stddev': 'upstream_speed_stddev', 'speed_max': 'upstream_speed_max',
                                                             'accel_stddev': 'upstream_accel_stddev', 
                                                              'num_BSMs_neg_accel':'upstream_num_BSMs_neg_accel'})
        
        downstream_temp_df = base1_df_name.copy()
        downstream_temp_df = downstream_temp_df.rename(columns = {'bsm_tmp_id_Count':'downstream_bsm_tmp_id_Count', 
                                                             'num_BSMs_0speed':'downstream_num_BSMs_0speed',
                                                             'num_BSMs_0_to_following_speed': 'downstream_num_BSMs_0_to_following_speed',
                                                             'num_BSMs_above_following_speed':'downstream_num_BSMs_above_following_speed',
                                                                  'speed_stddev': 'downstream_speed_stddev', 'speed_max': 'downstream_speed_max',
                                                             'accel_stddev': 'downstream_accel_stddev', 
                                                              'num_BSMs_neg_accel':'downstream_num_BSMs_neg_accel'})
        
        
        """join the upstream link features with base2_df"""
        base2_df = base2_df.merge(upstream_temp_df[['time_30','link','upstream_bsm_tmp_id_Count','upstream_num_BSMs_0speed',
                                                 'upstream_num_BSMs_0_to_following_speed','upstream_num_BSMs_above_following_speed', 
                                                    'upstream_speed_stddev', 'upstream_speed_max', 'upstream_accel_stddev',
                                                   'upstream_num_BSMs_neg_accel']], how = 'left', 
                                     left_on =['Upstream_LinkID','time'], right_on = ['link','time_30'])
        base2_df = base2_df.drop(['link', 'time_30'], axis=1)
        
        """join the downstream link features with base2_df"""
        base2_df = base2_df.merge(downstream_temp_df[['time_30','link','downstream_bsm_tmp_id_Count','downstream_num_BSMs_0speed',
                                                 'downstream_num_BSMs_0_to_following_speed','downstream_num_BSMs_above_following_speed', 
                                                    'downstream_speed_stddev', 'downstream_speed_max', 'downstream_accel_stddev',
                                                   'downstream_num_BSMs_neg_accel']], how = 'left', 
                                     left_on =['Downstream_LinkID','time'], right_on = ['link','time_30'])
        base2_df = base2_df.drop(['link', 'time_30'], axis=1)
        
        return base2_df

def join_number_of_lanes_on_ramp(stopline_avg_df_name, base_new_df_name):
    """join the number of on-ramp lanes to base_new_df
    
    Keyword arguments:
    stopline_avg_df_name -- df with stopline average data
    base_new_df_name -- df used for connecting ramps to road segments
    """

    base_temp = base_new_df_name.merge(stopline_avg_df_name[['Ramp_ID', 'n_lanes']], how = 'left', left_on = 'Ramp_ID',
                                      right_on = 'Ramp_ID')
    
    return base_temp

def join_sensor_id_for_on_ramp_and_updown_sensors(base_new_df_name, neighborIDs_new_df_name):
    """join the on-ramp sensor ID and the closest upstream and downstream sensor ID to base_new_df

    Keyword arguments:
    base_new_df_name -- df used for connecting ramps to road segments
    neighborIDs_new_df_name -- df with neighbor IDs
    """

    base_new_temp = base_new_df_name.merge(neighborIDs_new_df_name[['Ramp', 'Ramp_Sensor_ID', 
                                                                   'Immediately_upstream_mainline_sensor_ID',
                                                                  'Immediately_downstream_mainline_sensor_ID']], 
                                          how = 'left', left_on = 'ramp', right_on = 'Ramp')
    base_new_temp = base_new_temp.drop(['Ramp'], axis=1)
    return base_new_temp

def join_sensor_occupancy_flow_features(base_new1_df_name, df_sensor_data_new_name):
    """join the sensor occupancy and flow for the on-ramp sensor, closest mainline link upstream sensor, 
    and closest mainline link downstream sensor

    Keyword arguments:
    base_new1_df_name -- df used for connecting ramps to road segments
    df_sensor_data_new_name -- df containing sensor data
    """
    
    # for on-ramp sensor
    on_ramp_df_sensor_data_new_name = df_sensor_data_new_name.copy() 
    on_ramp_df_sensor_data_new_name = on_ramp_df_sensor_data_new_name.rename(columns = {'occupancy':'onramp_occupancy'})
    on_ramp_df_sensor_data_new_name = on_ramp_df_sensor_data_new_name.rename(columns = {'flow':'onramp_flow'})
    
    base_new_temp = base_new1_df_name.merge(on_ramp_df_sensor_data_new_name[['onramp_occupancy', 'onramp_flow', 'sensorID','time_30']],
                                           how = 'left', left_on = ['Ramp_Sensor_ID', 'time'], right_on = ['sensorID', 'time_30'])
    base_new_temp = base_new_temp.drop(['sensorID', 'time_30'], axis=1)
    
    # for upstream sensor
    up_df_sensor_data_new_name = df_sensor_data_new_name.copy()
    up_df_sensor_data_new_name = up_df_sensor_data_new_name.rename(columns = {'occupancy':'upstream_occupancy'})
    up_df_sensor_data_new_name = up_df_sensor_data_new_name.rename(columns = {'flow':'upstream_flow'})
    
    base_new_temp = base_new_temp.merge(up_df_sensor_data_new_name[['upstream_occupancy', 'upstream_flow', 'sensorID','time_30']],
                                           how = 'left', left_on = ['Immediately_upstream_mainline_sensor_ID', 'time'], right_on = ['sensorID', 'time_30'])
    base_new_temp = base_new_temp.drop(['sensorID', 'time_30'], axis=1)
    
    # for downstream sensor
    down_df_sensor_data_new_name = df_sensor_data_new_name.copy()
    down_df_sensor_data_new_name = down_df_sensor_data_new_name.rename(columns = {'occupancy':'downstream_occupancy'})
    down_df_sensor_data_new_name = down_df_sensor_data_new_name.rename(columns = {'flow':'downstream_flow'})

    
    base_new_temp = base_new_temp.merge(down_df_sensor_data_new_name[['downstream_occupancy', 'downstream_flow', 'sensorID','time_30']],
                                           how = 'left', left_on = ['Immediately_downstream_mainline_sensor_ID', 'time'], right_on = ['sensorID', 'time_30'])
    base_new_temp = base_new_temp.drop(['sensorID', 'time_30'], axis=1)
    
    return base_new_temp

# feature_engineering(df_on_ramps, df_up_downstream_ramps, avg_sensor_data_df, stopline_avg_df)
def feature_engineering(df_BSM_name, df_BSM_upstream_downstream, stopline_avg_df_name):
    """Create grouped df with new aggregated features based on BSMs.

    Keyword arguments:
    df_BSM_name -- df with BSM data
    df_BSM_upstream_downstream -- df containing upstream and downstream ramp data
    stopline_avg_df_name -- df containing stopline averages

    """

    # Our main group by object (on-ramp, lane, 30 second time chunk) - lane may not work or be needed

    # Converting to int because of needed format
    df_BSM_name['brakeStatus'] = df_BSM_name['brakeStatus'].apply(lambda x: int(x))

    if df_BSM_upstream_downstream.empty:
        gb_updown_links = pd.DataFrame(columns=['bsm_tmp_id', 'transtime_30sec', 'link']).groupby(['transtime_30sec', 'bsm_tmp_id']).count()
    else:
        gb_updown_links = df_BSM_upstream_downstream.groupby(['transtime_30sec','link'])[['bsm_tmp_id']].count()


    # Check for no BSM messages # NEW
    if df_BSM_name.empty: 
        base_df = pd.DataFrame(columns=['transtime_30sec', 'Ramp_ID', 'lane', 'bsm_tmp_id_Count',
                                    'num_BSMs_0speed', 'num_BSMs_0_to_following_speed',
                                    'num_BSMs_above_following_speed', 'num_BSMs_len_above_avg',
                                    'num_BSMs_len_below_avg', 'veh_len_avg_in_group',
                                    'veh_len_med_in_group', 'speed_stddev', 'speed_max', 'accel_stddev',
                                    'num_BSMs_neg_accel', 'max_X', 'max_Y', 'min_X', 'min_Y',
                                    'max_distance_between_BSMs', 'min_dist_to_stopbar',
                                    'max_dist_to_stopbar', 'num_braking', 'num_braking_hard',
                                    'hard_braking', 'time_30', 'n_lanes'])
    else:
    
        # do this for both on-ramps and up and downstream mainline links (these will later be joined)
        gb_main = df_BSM_name.groupby(['transtime_30sec','Ramp_ID', 'lane'])[['bsm_tmp_id']].count()
        
        # creating the base aggregated DF to add columns to
        base_df = gb_main.add_suffix('_Count').reset_index()
        gb = df_BSM_name.groupby(['transtime_30sec','Ramp_ID', 'lane'])

        # get the value of the average vehicle length across all BSMs on on-ramps
        avg_veh_len = df_BSM_name["Length (ft)"].mean()
        median_veh_len = df_BSM_name["Length (ft)"].median()

        # count # of BSMs in 30 sec-ramp-lane grouping with 0 speed
        base_df['num_BSMs_0speed'] = gb['Speed'].apply(lambda x: (x==0).sum()).reset_index(name='sum').iloc[:,3]
        
        # number of BSMs with speed between 0 and QUEUE_FOLLOWING_SPEED
        base_df['num_BSMs_0_to_following_speed'] = gb['Speed'].apply(lambda x: ((x>0) & (x<=QUEUE_FOLLOWING_SPEED)).sum()).reset_index(name='sum').iloc[:,3]

        # number of BSMs greater than QUEUE_FOLLOWING_SPEED
        base_df['num_BSMs_above_following_speed'] = gb['Speed'].apply(lambda x: (x>QUEUE_FOLLOWING_SPEED).sum()).reset_index(name='sum').iloc[:,3]
        
        # number of BSMs with vehicle length above average length on all on-ramps
        base_df['num_BSMs_len_above_avg'] = gb["Length (ft)"].apply(lambda x: (x>avg_veh_len).sum()).reset_index(name='sum').iloc[:,3]

        # number of BSMs with vehicle length equal to or below average
        base_df['num_BSMs_len_below_avg'] = gb["Length (ft)"].apply(lambda x: (x<=avg_veh_len).sum()).reset_index(name='sum').iloc[:,3]

        # get AVG vehicle length per grouping
        base_df['veh_len_avg_in_group'] = gb["Length (ft)"].mean().reset_index(name='sum').iloc[:,3]

        # get the MEDIAN vehicle length per grouping
        base_df['veh_len_med_in_group'] = gb["Length (ft)"].median().reset_index(name='sum').iloc[:,3]

        # speed standard deviation 
        base_df['speed_stddev'] = gb["Speed"].std().reset_index().iloc[:,3]

        # max speed in grouping
        base_df['speed_max'] = gb["Speed"].max().reset_index().iloc[:,3]

        # acceleration standard deviation
        # could be called "Instant_Acceleration" or "Avg_Acceleration" instead of "Acceleration"
        base_df['accel_stddev'] = gb["Acceleration"].std().reset_index().iloc[:,3]

        # number of BSMs with negative acceleration
        base_df['num_BSMs_neg_accel'] = gb["Acceleration"].apply(lambda x: (x<=0).sum()).reset_index(name='sum').iloc[:,3]

        # These max and mins will be used to calculate distance features between BSMs and from the BSMs to the on-ramp stopbars
        # Max X per group
        base_df['max_X'] = gb["X"].max().reset_index(name='max').iloc[:,3]

        # Max Y per group
        base_df['max_Y'] = gb["Y"].max().reset_index(name='max').iloc[:,3]

        # Min X per group
        base_df['min_X'] = gb["X"].min().reset_index(name='max').iloc[:,3]

        # Min Y per group
        base_df['min_Y'] = gb["Y"].min().reset_index(name='max').iloc[:,3]

        # the distances are approximations and do not account for the curvature of the on-ramps
        # distance between Max X,Y and Min X,Y to indicate how far apart the BSMs are
        base_df['max_distance_between_BSMs'] = base_df.apply(lambda row: distance_between(row['max_X'],row['max_Y'],row['min_X'],row['min_Y']), axis=1)

        # the distances are approximations and do not account for the curvature of the on-ramps
        base_df['min_dist_to_stopbar'] = base_df.apply(lambda row: min_dist_to_avg_stopbar(row, stopline_avg_df_name), axis=1)

        base_df['max_dist_to_stopbar'] = base_df.apply(lambda row: max_dist_to_avg_stopbar(row, stopline_avg_df_name), axis=1)                                 

        # Create frequency of braking features
        base_df['num_braking'] = gb["brakeStatus"].apply(lambda x: (x>0).sum()).reset_index(name='sum').iloc[:,3]
        base_df['num_braking_hard'] = gb["hardBraking"].apply(lambda x: (x>0).sum()).reset_index(name='sum').iloc[:,3]
        # change it to 1/0 yes/no hard braking occurred
        base_df['hard_braking'] = 0
        mask_hardBrake = (base_df['num_braking_hard']>0)
        base_df.loc[mask_hardBrake,'hard_braking'] = 1

        # convert timedelta to string
        base_df['time_30'] = base_df['transtime_30sec'].astype(str).str[7:15]
        
        # add number of lanes for on-ramp to features
        base_df = join_number_of_lanes_on_ramp(stopline_avg_df_name, base_df)

    # creating the upstream and downstream links' aggregated DF to add columns to
    base1_df = gb_updown_links.add_suffix('_Count').reset_index()

    if df_BSM_upstream_downstream.empty:
        df_BSM_upstream_downstream = pd.DataFrame(columns=['Vehicle_ID', 'bsm_tmp_id', 'time', 
                                               'Speed', 'X', 'Y', 'transtime',
                                               'transTo', 'Acceleration', 'brakeStatus', 
                                               'brakePressure', 'hardBraking', 'transmission_received_time', 
                                               'Heading', 'link', 'lane',
                                               'Type', 'transtime_30sec', 'Length (ft)'])

    gb1 = df_BSM_upstream_downstream.groupby(['transtime_30sec','link'])
    
    # count # of BSMs in 30 sec-link grouping with 0 speed
    base1_df['num_BSMs_0speed'] = gb1['Speed'].apply(lambda x: (x==0).sum()).reset_index(name='sum').iloc[:,2]
        
    # number of BSMs on links with speed between 0 and QUEUE_FOLLOWING_SPEED
    base1_df['num_BSMs_0_to_following_speed'] = gb1['Speed'].apply(lambda x: ((x>0) & (x<=QUEUE_FOLLOWING_SPEED)).sum()).reset_index(name='sum').iloc[:,2]
    
    # number of BSMs on links with speed greater than QUEUE_FOLLOWING_SPEED
    base1_df['num_BSMs_above_following_speed'] = gb1['Speed'].apply(lambda x: (x>QUEUE_FOLLOWING_SPEED).sum()).reset_index(name='sum').iloc[:,2]
    
    # speed standard deviation 
    base1_df['speed_stddev'] = gb1["Speed"].std().reset_index().iloc[:,2]

    # max speed in grouping
    base1_df['speed_max'] = gb1["Speed"].max().reset_index().iloc[:,2]

    # acceleration standard deviation
    # could be called "Instant_Acceleration" or "Avg_Acceleration" instead of "Acceleration"
    base1_df['accel_stddev'] = gb1["Acceleration"].std().reset_index().iloc[:,2]

    # number of BSMs with negative acceleration
    base1_df['num_BSMs_neg_accel'] = gb1["Acceleration"].apply(lambda x: (x<=0).sum()).reset_index(name='sum').iloc[:,2]
    
    # convert timedelta to string ## ERROR LOCATION
    base1_df['time_30'] = base1_df['transtime_30sec'].astype(str).str[7:15]
    
    return base_df, base1_df

def handle_missing_data(df_xy_name, df_BSM_name):
    """Since python's scikit-learn will not accept rows with NA, this function replaces NAs with 0 for most columns except the veh len avg and median.
    Assumption: rows with NA for the BSM features did not see any BSMs sent from a CV in that link and time period. 
    Please note: Handling missing data is more an art than a science! You may want to handle NAs differently in your case.


    Keyword arguments:
    df_xy_name -- intermediate df in the feature engineering process
    df_BSM_name -- df containing BSM data
    """

    ## Handling NaN rows in df_xy
    #replace NaN with 0
    df_xy = df_xy_name.fillna(0)

    # get the value of the average vehicle length across all BSMs
    avg_veh_len = df_BSM_name["Length (ft)"].mean()
    median_veh_len = df_BSM_name["Length (ft)"].median()

    # replace 0 values for veh_len_avg_in_group with the average over all BSMs
    mask_veh_avg = (df_xy['veh_len_avg_in_group']==0)
    if pd.isnull(avg_veh_len): #NEW
        df_xy.loc[mask_veh_avg,'veh_len_avg_in_group'] = CAR_LENGTH_AVG
    else:
        df_xy.loc[mask_veh_avg,'veh_len_avg_in_group'] = avg_veh_len

    # replace 0 values for veh_len_med_in_group with the median over all BSMs
    mask_veh_med = (df_xy['veh_len_med_in_group']==0)
    if pd.isnull(median_veh_len):
        df_xy.loc[mask_veh_med,'veh_len_med_in_group'] = CAR_LENGTH_AVG
    else:
        df_xy.loc[mask_veh_med,'veh_len_med_in_group'] = median_veh_len
    return df_xy

def label_encode_categorical_features(df_xy_name):
    """Label encode categorical features for ML models
    Please note: encoding is also more of an art than a science. You could try different methods.

    Keyword arguments:
    df_xy_name -- intermediate df in the feature engineering process""" 

    # label encode the ramp IDs
    df_xy_name["ramp"] = df_xy_name["ramp"].astype('category')
    df_xy_name["ramp_encoded"] = df_xy_name["ramp"].cat.codes

    # now drop the original 'ramp' column (you don't need it anymore)
    df_xy_name.drop(['ramp'],axis=1, inplace=True)

    # label encode the on-ramp upstream and downstream mainline link IDs
    df_xy_name["Upstream_LinkID"] = df_xy_name["Upstream_LinkID"].astype('category')
    df_xy_name["Upstream_LinkID_encoded"] = df_xy_name["Upstream_LinkID"].cat.codes
    # now drop the original 'Upstream_LinkID' column (you don't need it anymore)
    df_xy_name.drop(['Upstream_LinkID'],axis=1, inplace=True)
    
    df_xy_name["Downstream_LinkID"] = df_xy_name["Downstream_LinkID"].astype('category')
    df_xy_name["Downstream_LinkID_encoded"] = df_xy_name["Downstream_LinkID"].cat.codes
    # now drop the original 'Downstream_LinkID' column (you don't need it anymore)
    df_xy_name.drop(['Downstream_LinkID'],axis=1, inplace=True)

    # needs to be numeric to work in sklearn
    df_xy_name['time_float'] = df_xy_name['time'].apply(to_seconds)

    return df_xy_name

def format_queues(y_df_name):
    """Bin the number of vehicles in queue into pairs. 
    Used for reading in ground truth labels for training.

    Keyword arguments:
    y_df_name -- df with the y label values from ground truth calculation
    """

    # Creating a queue indicator column
    # add a column to y_df that is an indicator 1/0 for queue at ramp yes/no
    if y_df_name.shape[0] == 0:
        cols_keep = y_df_name.columns.tolist()
        cols_keep.extend(['queue_indicator', 'queue_count_binned'])
        y_df_name = pd.DataFrame(columns=cols_keep)
        return y_df_name

    y_df_name['queue_indicator'] = 0
    mask_queue = (y_df_name['queue_count_max']>0)
    y_df_name.loc[mask_queue,'queue_indicator'] = 1
    # Creating a queue count binned column, pairs of # vehs
    # bin the queue counts into pairs as high as your max queue count observed in your training data
    # check for longest ramp divided by the smallest vehicle length for bin label possibilities
    binned_queues = [-np.inf,0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,
                     70,72,74,76,78,80,82,84,86,88,90,92,94,96,98,100,102,104,106,108,110,112,114,116,118,120,122,124,126,128,130,
                     132,134,136,138,140,142,144]
    bin_labels = ["no_queue","1-2", "3-4", "5-6", "7-8", "9-10", "11-12", "13-14", "15-16",
                "17-18", "19-20", "21-22", "23-24", "25-26", "27-28", "29-30", "31-32", "33-34", "35-36", "37-38", "39-40",
                "41-42", "43-44", "45-46", "47-48", "49-50", "51-52", "53-54", "55-56", "57-58", "59-60", "61-62", "63-64", 
                "65-66", "67-68", "69-70", "71-72", "73-74", "75-76", "77-78", "79-80", "81-82", "83-84", "85-86", "87-88",
                "89-90", "91-92", "93-94", "95-96", "97-98", "99-100", "101-102", "103-104", "105-106", "107-108", "109-110",
                "111-112", "113-114", "115-116", "117-118", "119-120", "121-122", "123-124", "125-126", "127-128", "129-130",
                "131-132", "133-134", "135-136", "137-138", "139-140", "141-142", "143-144"]

    # convert the categorically binned queue_count_binned column to int with .cat.codes
    y_df_name['queue_count_binned']=pd.cut(x=y_df_name['queue_count_max'], bins=binned_queues, 
                                    labels = bin_labels, include_lowest =True).cat.codes
    return y_df_name

def join_features_and_labels(base_df_name, y_df_name, base1_df_name, df_sensor_data_new_name, up_and_downstream_ramp_links_df_name, neighborIDs_new_df_name):
    """Join the aggregated features (created from BSMs and supporting files) and their labels (queue count and length).

    Keyword arguments:
    base_df_name -- intermediate df used for transformation
    y_df_name -- df with the y label values from ground truth calculation
    base1_df_name -- intermediate df used for transformation
    df_sensor_data_new_name -- df with sensor data
    up_and_downstream_ramp_links_df_name -- df with up and downstream links
    neighborIDs_new_df_name -- df with neighbor IDs
    """

    ''' join the labels (y_df) to the upstream and downstream mainline links, on-ramp sensorID, and upstream and downstream mainline sensorIDs,
    then join this with the on-ramp BSM (base_df), on-ramps' upstream and downstream mainline links' BSM (base1_df), and on-ramp plus 
    upsteam and downstream mainline sensors' (df_sensor_data_new) features '''
    
    df_xy = join_up_and_downstream_linkIDs_to_base_df(up_and_downstream_ramp_links_df_name, y_df_name)
    
    df_xy = join_sensor_id_for_on_ramp_and_updown_sensors(df_xy, neighborIDs_new_df_name)
    
    base_df_name['lane'] = base_df_name.lane.apply(lambda x: int(x))
    df_xy['lane'] = df_xy.lane.apply(lambda x: int(x))

    df_xy = df_xy.merge(base_df_name, how= 'left', left_on=['time','ramp', 'lane'],right_on=['time_30', 'Ramp_ID', 'lane'])
    df_xy = df_xy.drop(['Ramp_ID', 'time_30'], axis=1)
    
    # join the BSM data from the mainline links upstream and downstream of the on-ramps
    df_xy = join_base1_df_to_df_xy(df_xy, base1_df_name)
    
    # join the sensor data to df_xy for on-ramp sensor and upstream and downstream mainline sensor
    df_xy = join_sensor_occupancy_flow_features(df_xy, df_sensor_data_new_name)

    # df_xy = join_sensor_occupancy_features(df_xy, df_sensor_data_new_name)
    
    df_xy.drop('transtime_30sec',axis='columns', inplace=True)
    
    return df_xy

def add_previous_time_queue_count_col(df_xy_name):
    """Creating a column that captures the previous 4 timesteps' queue_counts, 
    occupancies for on-ramp sensor, sensor upstream of on-ramp, and occupancies for sensor downstream of on-ramp.

    Keyword arguments:
    df_xy_name -- intermidate df used for feature engineering
    """

    # to datetime
    df_xy_name['time_30_dt']= pd.to_datetime(df_xy_name['time'], format="%H:%M:%S")

    # add new columns for 4 previous timesteps (30 second increments) to the current time 
    df_xy_name['previous_time_30sec'] = df_xy_name['time_30_dt'] - timedelta(seconds=30)
    df_xy_name['previous_time_60sec'] = df_xy_name['previous_time_30sec'] - timedelta(seconds=30)
    df_xy_name['previous_time_90sec'] = df_xy_name['previous_time_60sec'] - timedelta(seconds=30)
    df_xy_name['previous_time_120sec'] = df_xy_name['previous_time_90sec'] - timedelta(seconds=30)

    # now remove the date from the datetime
    df_xy_name['time_30_dt'] = df_xy_name['time_30_dt'].dt.time
    df_xy_name['previous_time_30sec'] = df_xy_name['previous_time_30sec'].dt.time
    df_xy_name['previous_time_60sec'] = df_xy_name['previous_time_60sec'].dt.time
    df_xy_name['previous_time_90sec'] = df_xy_name['previous_time_90sec'].dt.time
    df_xy_name['previous_time_120sec'] = df_xy_name['previous_time_120sec'].dt.time
    
    # add previous time_30sec data to df_xy
    #  self left join, left on current time, right on previous time 30sec (same ramp and lane!)
    base = pd.merge(df_xy_name, df_xy_name, 
            left_on=['previous_time_30sec','ramp', 'lane'],
            right_on=['time_30_dt','ramp', 'lane'],
            how = 'left', copy=False, suffixes=('', '_previous30'))

    
    # columns to keep in base
    cols_keep = df_xy_name.columns.tolist()
    # cols_keep.append('queue_count_max_previous')
    cols_keep.extend(['queue_count_max_previous30', 'onramp_occupancy_previous30',
                     'upstream_occupancy_previous30', 'downstream_occupancy_previous30', 
                     'onramp_flow_previous30','upstream_flow_previous30', 'downstream_flow_previous30'])

    # keep only the original columns plus the queue_count_max_previous
    base = base.loc[:,base.columns.isin(['time','ramp', 'lane','queue_count_max_previous30',
                                         'onramp_occupancy_previous30','upstream_occupancy_previous30', 
                                         'downstream_occupancy_previous30', 'onramp_flow_previous30',
                                         'upstream_flow_previous30', 'downstream_flow_previous30'])]


    df_xy = df_xy_name.merge(base, how='left', 
                            left_on=['time','ramp', 'lane'], 
                            right_on=['time','ramp', 'lane'])
    
    df_xy.drop_duplicates(inplace=True)
    df_xy.reset_index(drop=True, inplace=True)

    df_xy.drop(['previous_time_30sec'], axis=1, inplace=True)
    
    # add previous time_60sec data to df_xy
    # self left join, left on current time, right on previous time 60sec (same ramp and lane!)
    base = pd.merge(df_xy, df_xy, 
            left_on=['previous_time_60sec','ramp', 'lane'],
            right_on=['time_30_dt','ramp', 'lane'],
            how = 'left', copy=False, suffixes=('', '_previous60'))
    
    # columns to keep in base
    cols_keep = df_xy.columns.tolist()
    # cols_keep.append('queue_count_max_previous')
    cols_keep.extend(['queue_count_max_previous60', 'onramp_occupancy_previous60',
                     'upstream_occupancy_previous60', 'downstream_occupancy_previous60',
                     'onramp_flow_previous60','upstream_flow_previous60', 'downstream_flow_previous60'])

    # keep only the original columns plus the queue_count_max_previous
    base = base.loc[:,base.columns.isin(['time','ramp', 'lane','queue_count_max_previous60',
                                         'onramp_occupancy_previous60','upstream_occupancy_previous60', 
                                         'downstream_occupancy_previous60', 'onramp_flow_previous60',
                                         'upstream_flow_previous60', 'downstream_flow_previous60'])]

    df_xy = df_xy.merge(base, how='left', 
                            left_on=['time','ramp', 'lane'], 
                            right_on=['time','ramp', 'lane'])

    df_xy.drop_duplicates(inplace=True)
    df_xy.reset_index(drop=True, inplace=True)
    
    df_xy.drop(['previous_time_60sec'], axis=1, inplace=True)
    
    # add previous time_90sec data to df_xy
    # self left join, left on current time, right on previous time 90sec (same ramp and lane!)
    base = pd.merge(df_xy, df_xy, 
            left_on=['previous_time_90sec','ramp', 'lane'],
            right_on=['time_30_dt','ramp', 'lane'],
            how = 'left', copy=False, suffixes=('', '_previous90'))
    
    # columns to keep in base
    cols_keep = df_xy.columns.tolist()
    # cols_keep.append('queue_count_max_previous')
    cols_keep.extend(['queue_count_max_previous90', 'onramp_occupancy_previous90',
                     'upstream_occupancy_previous90', 'downstream_occupancy_previous90', 
                     'onramp_flow_previous90','upstream_flow_previous90', 'downstream_flow_previous90'])

    # keep only the original columns plus the queue_count_max_previous
    base = base.loc[:,base.columns.isin(['time','ramp', 'lane','queue_count_max_previous90',
                                         'onramp_occupancy_previous90','upstream_occupancy_previous90', 
                                         'downstream_occupancy_previous90','onramp_flow_previous90',
                                         'upstream_flow_previous90', 'downstream_flow_previous90'])]

    df_xy = df_xy.merge(base, how='left', 
                            left_on=['time','ramp', 'lane'], 
                            right_on=['time','ramp', 'lane'])

    df_xy.drop_duplicates(inplace=True)
    df_xy.reset_index(drop=True, inplace=True)
    
    df_xy.drop(['previous_time_90sec'], axis=1, inplace=True)
    
    # add previous time_120sec data to df_xy
    # self left join, left on current time, right on previous time 120sec (same ramp and lane!)
    base = pd.merge(df_xy, df_xy, 
            left_on=['previous_time_120sec','ramp', 'lane'],
            right_on=['time_30_dt','ramp', 'lane'],
            how = 'left', copy=False, suffixes=('', '_previous120'))
    
    # columns to keep in base
    cols_keep = df_xy.columns.tolist()
    # cols_keep.append('queue_count_max_previous')
    cols_keep.extend(['queue_count_max_previous120', 'onramp_occupancy_previous120',
                     'upstream_occupancy_previous120', 'downstream_occupancy_previous120',
                     'onramp_flow_previous120','upstream_flow_previous120', 'downstream_flow_previous120'])

    # keep only the original columns plus the queue_count_max_previous
    base = base.loc[:,base.columns.isin(['time','ramp', 'lane','queue_count_max_previous120',
                                         'onramp_occupancy_previous120','upstream_occupancy_previous120', 
                                         'downstream_occupancy_previous120', 'onramp_flow_previous120',
                                         'upstream_flow_previous120', 'downstream_flow_previous120'])]

    df_xy = df_xy.merge(base, how='left', 
                            left_on=['time','ramp', 'lane'], 
                            right_on=['time','ramp', 'lane'])

    df_xy.drop_duplicates(inplace=True)
    df_xy.reset_index(drop=True, inplace=True)
    
    df_xy.drop(['previous_time_120sec', 'time_30_dt'], axis=1, inplace=True)    
    
    return df_xy

def split_into_X_and_Y(df_xy_name, label_selection = 'queue_count_max'):
    """Separate the features (X) and the labels (Y). The default label selection (Y) is queue_count_max. 

    Keyword arguments:
    df_xy_name -- intermediate df for feature engineering
    label_selection -- column to be considered label (default 'queue_count_max')

    """

    # Preparing X and y
    col_lst = ['queue_count_max', 'queue_len_max', 'queue_indicator', 'queue_count_binned','time']
    X = df_xy_name.loc[:,~df_xy_name.columns.isin(col_lst)]
    y = df_xy_name[label_selection]
    return X, y

def read_BSMs_file(BSMs_X_filename):
    """Read BSMs csv file and store in pandas dataframe (df). Used for training data processing.

    Keyword arguments:
    BSMs_X_filename -- filepath for X BSM data (features)"""
    columns = ['bsm_tmp_id', 'time', 'Speed', 'X', 'Y', 'transtime', 'transTo', 'Acceleration', 'brakeStatus', 'brakePressure', 
               'hardBraking', 'transmission_received_time', 'Heading', 'link', 'lane', 'Type']
    

    df = pd.read_csv(BSMs_X_filename, header = 0, names = columns)
    return df

def y_dummy_append(y_df_name, all_ramp_lanes, start_seconds):
    ''' Need to create dummy data to append to y dataframe
    Necessary for processing, put -1 as dummy values
    Will use the start of the run variable given by SUMO. 

    Keyword arguments:
    y_df_name -- df for labels
    all_ramp_lanes -- df that contains all ramp and lane combinations
    start_seconds -- start of simulation in seconds 
    '''

    new_timestep = start_seconds
    new_timestep_str = str(timedelta(seconds = int(start_seconds), hours = 13)).split()[-1]

    if new_timestep_str == 0:
        return y_df_name

    # Get all combos of ramps and lanes
    temp_ydf = all_ramp_lanes.copy()
    temp_ydf.columns = ['ramp', 'lane']

    # Include time column with new timestep
    temp_ydf['time'] = timedelta(seconds = int(start_seconds), hours = 13)
    temp_ydf['time'] = temp_ydf.time.apply(lambda x: str(x).split()[-1])

    # Dummy value of -1 for all queue count metrics
    temp_ydf['queue_count_max'] = -1
    temp_ydf['queue_len_max'] = -1

    # Append to original y dataframe
    y_df_new = pd.concat([y_df_name, temp_ydf])
    return y_df_new

def import_trained_XGB_model(model_filename):
     """Import trained ML model pkl file and store in joblib_model_xgb

    Keyword arguments:
    model_filename -- scikitlearn pickled model file location """

     joblib_model_xgb = joblib.load(model_filename)
     return joblib_model_xgb

def replace_dummy_y_with_predictions(y_real_name, y_pred_name):
    '''Takes two vectors, y real and y predicted, of equal length. Replaces dummy values (-1) with predicted values of y

    Keyword arguments:
    y_real_name -- list with ongoing y values
    y_pred_name -- list with new y predictions '''

    temp_y = list()
    for i in range(len(y_real_name)):
        if y_real_name[i] == -1:
            temp_y.append(y_pred_name[i])
        else:
            temp_y.append(y_real_name[i])
    return temp_y

def update_dfxy_with_predictions(df_xy_name, y_updated_name, label_selection = 'queue_count_max'):
    ''' Inserts the new updated y (with predictions) into df_xy 

    Keyword arguments:
    df_xy_name -- intermediate df with data
    y_updated_name -- y df updated by replace_dummy_y_with_predictions
    '''

    df_xy_name[label_selection] = np.rint(y_updated_name)
    return df_xy_name

def derive_queue_len_from_count(df_xy_name):
    """Use the ML classifications of queue count to estimate queue length for each link

    Keyword arguments:
    df_xy_name -- intermediate df with data"""

    df_xy_name['queue_len_max'] = df_xy_name.apply(lambda x: x['queue_count_max'] * x['veh_len_avg_in_group'], axis=1)
    return df_xy_name

def reformat_df_predictions(df_preds_name, ramp_dict_path):
    '''Reformats the predicted ys to how they have to be read in for the next round

    Keyword arguments:
    df_preds_name -- df_xy with predictions incorporated
    ramp_dict_path -- filepath for ramp encodings'''
    
    # Reading in ramp dictionary
    with open(ramp_dict_path) as json_file:
        ramp_dict = json.load(json_file)

    # Need integer keys
    ramp_dict = {int(k):v for k,v in ramp_dict.items()}    

    # map dictionary to encoding
    df_preds_name['ramp'] = df_preds_name.ramp_encoded.map(ramp_dict)

    # Need a column called time_30 for sake of how preprocessing was set up
    # df_preds_name['time_30'] = df_preds_name.time.apply(lambda x: '0 days ' + x)
    df_preds_name['time_30'] = df_preds_name.time

    # Make lane a string
    df_preds_name['lane'] = df_preds_name.lane.apply(lambda x: str(x))

    # Return only necessary columns for matching y_format
    y_return = df_preds_name[['ramp', 'lane', 'time_30', 'queue_count_max', 'queue_len_max']]
    
    return y_return

def read_sensor_data_live(flow_dict_name, occupancy_dict_name, start_seconds):
    '''
    Takes two dictionaries of values. One for flows and one for occupancy sensor data.
    Returns data frame with the two merged.
    Also takes a time in the form of the beginning time of interval, in seconds

    Keyword arguments:
    flow_dict_name -- dictionary with sensor flows
    occupancy_dict_name -- dictionary with occupancy flows
    start_seconds -- start time in seconds of the timestep being processed
    '''

    # Covert to one row df
    sensor_flow_df1r = pd.DataFrame(flow_dict_name, index=[0])
    sensor_occ_df1r = pd.DataFrame(occupancy_dict_name, index=[0])

    # Convert columns to rows
    sensor_flow_df1r = sensor_flow_df1r.melt(var_name="sensor", 
                                             value_name="flow")
    sensor_occ_df1r = sensor_occ_df1r.melt(var_name="sensor", 
                                             value_name="occupancy")
    # group by ramp sensor, take averages
    sensor_flow_df1r['sensorID'] = sensor_flow_df1r.sensor.apply(lambda x: '_'.join(x.split('_')[:-1]))
    flow_avg_df1r = sensor_flow_df1r[['flow', 'sensorID']].groupby(['sensorID']).mean('flow').reset_index()

    sensor_occ_df1r['sensorID'] = sensor_occ_df1r.sensor.apply(lambda x: '_'.join(x.split('_')[:-1]))
    occ_avg_df1r = sensor_occ_df1r[['occupancy', 'sensorID']].groupby(['sensorID']).mean('occupancy').reset_index()

    # Merge the two datasets
    sensor_data_df1r = flow_avg_df1r.merge(right=occ_avg_df1r, how = 'inner', on='sensorID')
    
    # Include timestamp
    sensor_data_df1r['time_30'] = timedelta(seconds = int(start_seconds), hours = 13)
    sensor_data_df1r['time_30'] = sensor_data_df1r['time_30'].dt.floor('30S')
    sensor_data_df1r['time_30'] = sensor_data_df1r.time_30.apply(lambda x: str(x).split()[-1])
    
    return sensor_data_df1r

def read_sensor_data_offline(flow_filepath_name, occupancy_filepath_name):
    '''
    Takes filepaths for full flow and occupancy data and returns DataFrame with all sensor data averaged by timestep and across ramps.
    Used for data processing for training.

    Keyword arguments:
    flow_filepath_name -- filepath that contains all flow sensor data for a run
    occupancy_filepath_name -- filepath that contains all occupancy sensor data for a run
    '''

    sensor_flow_df = pd.read_csv(flow_filepath_name)
    sensor_occ_df = pd.read_csv(occupancy_filepath_name)

    # Convert columns to rows
    sensor_flow_df = sensor_flow_df.melt(id_vars = ['begin', 'end'],
                                         var_name="sensor", 
                                         value_name="flow")
    sensor_occ_df = sensor_occ_df.melt(id_vars = ['begin', 'end'],
                                       var_name="sensor", 
                                       value_name="occupancy")
    # group by ramp sensor, take averages
    sensor_flow_df['sensorID'] = sensor_flow_df.sensor.apply(lambda x: '_'.join(x.split('_')[:-1]))
    flow_avg = sensor_flow_df.groupby(['begin', 'end', 'sensorID']).mean('flow').reset_index()

    sensor_occ_df['sensorID'] = sensor_occ_df.sensor.apply(lambda x: '_'.join(x.split('_')[:-1]))
    occ_avg = sensor_occ_df.groupby(['begin', 'end', 'sensorID']).mean('flow').reset_index()

    # Merge the two datasets
    sensor_data_df = flow_avg.merge(right=occ_avg, how = 'inner', on=['begin', 'end', 'sensorID'])

    # Include timestamp
    sensor_data_df['time_30'] = sensor_data_df.begin.apply(lambda x: timedelta(seconds = int(x), hours = 13))
    sensor_data_df['time_30'] = sensor_data_df.time_30.apply(lambda x: str(x).split()[-1])

    # Remove unused time columns
    sensor_data_df.drop(['begin', 'end'], axis=1, inplace=True)

    return sensor_data_df


def y_data_timestep(df_y, start_seconds):
    '''Takes in a predictions dataframe and a integer of seconds, returns a dataframe only of that timestep

    Keyword arguments:
    df_y -- df with predictions
    start_seconds -- timestep in seconds'''

    new_timestep_str = str(timedelta(seconds = int(start_seconds), hours = 13)).split()[-1]
    df_temp_y = df_y[df_y.time_30 == new_timestep_str].reset_index(drop=True)
    return df_temp_y