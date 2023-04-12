# NCHRP 08-145: Ramp metering Code Package Documentation

This folder contains the source code for ramp metering for the NCHRP 08-145 Project: Utilizing Cooperative Automated Transportation (CAT) Data for Freeway Operational Strategies. A detailed description of the methodologies can be found in the project final report. The source code contains the main file (run.py) that runs the simulation I-210 under the CAT-enhanced system-wide ramp metering algorithm, namely queue-informed HERO, in Scenario 1.

# README Outline:
* Project Description
* Prerequisites
* Usage

# Project Description

This folder contains the source codes 1) running the simualtion of I-210 (from postmile 28 to postmile 38) under the queue-informed system-wide ramp metering enhanced by the CAT data (Scenario 1) and 2) saving the results.

# Prerequisites
- Python 3.6 (or higher)

Python Package Requirements:
- numpy==1.19.5
- pandas == 1.1.5
- xgboost==1.6.0
- freewayControl
- connectedEnv
- TCARandom
- bsm_emulator
- queue_estimator
- traci==1.14.1
- sumolib==1.14.1

# Usage
Before running the simualtion, make sure that the following files/directoris exist:

- a simulation project directory containing necessary files, such as main simulation file ending with sumocfg.xml, route file, network file and so on;
- a list of json files of sensor description including senosr IDs and period; 
- a json file of of meter description including phase IDs, green time, minimum red time, maximum red time and so on;
- a json file storing parameters of the queue estimator;
- a json file specifying parameters of connected vehicles, such as communication error, equipment failure error, and so on;
- a csv file specifying locations of RSUs;
- the python package freewayControl.py;
- the python package connectedEnv.py;
- the python package TCARandom.py;
- the python package bsm_emulator.py;
- the python package queue_estimator.


To start the simulation, run the following script:
```
python run.py
```