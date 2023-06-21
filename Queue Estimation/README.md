# Summary

This documentation presents a comprehensive overview of the source code for the  Queue Estimator Machine Learning module for the NCHRP 08-145 Project, entitled "Utilizing Cooperative Automated Transportation (CAT) Data for Freeway Operational Strategies." The source code was developed for one of the freeway operational strategies selected by the Project Panel, ramp metering, as a case study. The Queue Estimator is as an input to Queue-informed freeway operation strategies.

This code was developed using Python 3.8

# Prerequisites

- scikit-learn == 1.1.1

Related functions:

- queue_estimator
- queue_estimator_trainer

# Code Structure Description

* **Queue estimation functions (Noblis)**: Modules, including queue_estimator.py, queue_fx.py, queue_estimator_trainer.py, ground_truth_max_queue_counts_and_lengths.py, process_data_for_training.py,  control.json, xgb_parameters.json, queueEstimator.json, that perform queue estimation functions.

# Key Input Files Preparation
- the python package queue_estimator.py
- the python package queue_fx.py
- the python package queue_estimator_trainer.py

# Usage

**Calculating Ground Truths from Trajectory Files**

This script converts trajectory files output by SUMO into the ground truths that are used as labels for training ML models. It is called from a command line interface.

- Ensure that appropriate files (trajectories, stoplines, vehicle lengths) are in a folder specified by the supporting_files_path variable at the beginning of the script *ground_truth_max_queue_counts_and_lengths.py*
- Run script *ground_truth_max_queue_counts_and_lengths.py* with the following parameters: trajectories_filename, veh_lengths_filename, stoplines_filename,  --out output_csv_filename. For example:

```
python ground_truth_max_queue_counts_and_lengths.py trajSample_7hours_Scenario2_test.csv vehicle_length_by_type_file_1.csv stopBar_new_test_calcs_for_known_subset.csv --out test_outputs_max_ground_truth_7hrs_scenariotest_new.csv
```
**Processing Data for Training ML Model**

This script uses BSM data from the bsm_emulator and ground truths from ground_truth_max_queue_counts_and_lengths.py to transform data into features and labels ready for ML training. It is called from a command line interface.

- Ensure that a control.json file includes input file paths for vehicle_lengths, stoplines, neighboring_links, sensorIDs_input, neighboring_sensors, all_ramp_lanes, ground_truths, flow_data, occupancy_data, and bsm_data.
- Ensure that the control.json file includes an output file path for the processed data as 'data_for_ML_filepath'
- Run the script, as seen below:

```
python Process_Data_For_Training.py
```

**Training ML Model**

The class QueueEstimatorTrainer() provides functionality to train and save ML models using a temporal validation approach.
- Ensure that control.json includes 'data_for_ML_filepath' created in the step above.
- Invoke an instance of the QueueEstimatorTrainer class

```
>>> import queue_estimator_trainer
>>> trainer = queue_estimator_trainer.QueueEstimatorTrainer('control.json', 'xgb_parameters.json')
```

- The main function of the QueueEstimatorTrainer class is xgboost_temporal_grid_search(). This may take significant time if there are a large number of combinations of parameters. It will keep the model associated with hyperparameters with the lowest mean squared error.

```
>>> trainer.xgboost_temporal_grid_search()
Combination 1 / 486, 0min, 0sec
MSE: 6.353795806936997
Combination 2 / 486, 0min, 2sec
```

After training and validation, the best performing model can be pickled for later use in the QueueEstimator

```
>>> trainer.save_best_model('best_model.pkl')
```

**Queue Estimator**

Creates predicted queue estimations from BSM and sensor data by 30 second timestep. Requires json control file and csvs for vehicle lengths, stoplines, neighboring links, sensorIds and neighboring sensors.

- Ensure that in addition to all other standard file paths used in the control.json file, 'xgb_model_filepath' is referring to the model that you want to use as the basis of the Queue Estimator.

- The main function of the QueueEstimator class is process_time_step(). For a timestep, it receives all BSMs, occupancy sensor data, flow sensor data, and the start of the timestep in seconds.
