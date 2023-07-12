# Summary：

This documentation presents a comprehensive overview of the source code developed as part of the NCHRP 08-145 Project, entitled "Utilizing Cooperative Automated Transportation (CAT) Data for Freeway Operational Strategies." The objective of this research is to assess operational scenarios and use cases where freeway operational strategies could be improved through the transmission of data between a transportation management system (TMS) and the larger CAT. The source code was developed for one of the freeway operational strategies selected by the Project Panel, ramp metering, as a case study. The source code includes both the benchmark (baseline scenario without integration of CAT data) and alternative scenarios (Scenarios 1, 2, and 3) with enhanced ramp metering control incorporating CAT data. The source code also includes the BSM Emulator used to create emulated Basic Safety Messages to use as the CAT data input, and Machine Learning Queue Estimation using CAT data for input to the enhanced ramp metering. The ramp metering code was developed using Python 3.6 and the BSM Emulator and Queue Estimation were developed using Python 3.10. The ramp metering code is divided into 16 distinct sub-groups, each focusing on a unique combination of ramp metering control algorithm, incident frequency, and scenario variation, as outlined below:

- Peak-hour with no/few incident-ALINEA-benchmark
- Peak-hour with no/few incident-Queue-informed ALINEA-Scenario 1
- Peak-hour with no/few incident-Queue-informed ALINEA-Scenario 2
- Peak-hour with no/few incident-Queue-informed ALINEA-Scenario 3
- Peak-hour with no/few incident-HERO-benchmark
- Peak-hour with no/few incident-Queue-informed HERO-Scenario 1
- Peak-hour with no/few incident-Queue-informed HERO-Scenario 2
- Peak-hour with no/few incident-Queue-informed HERO-Scenario 3
- Peak-hour with many incident-ALINEA-benchmark
- Peak-hour with many incident-Incident-aware ALINEA-Scenario 1
- Peak-hour with many incident-Incident-aware ALINEA-Scenario 2
- Peak-hour with many incident-Incident-aware ALINEA-Scenario 3
- Peak-hour with many incident-HERO-benchmark
- Peak-hour with many incident-Incident-aware HERO-Scenario 1
- Peak-hour with many incident-Incident-aware HERO-Scenario 2
- Peak-hour with many incident-Incident-aware HERO-Scenario 3

For an in-depth exploration of the methodologies employed, please refer to the project's final report. 

This software is offered as is, without warranty or promise of support of any kind either expressed or implied. Under no circumstance will the National Academy of Sciences or the Transportation Research Board (collectively “TRB”) be liable for any loss or damage caused by the installation or operation of this product. TRB makes no representation or warranty of any kind, expressed or implied, in fact or in law, including without limitation, the warranty of merchantability or the warranty of fitness for a particular purpose, and shall not in any case be liable for any consequential or special damages.


# Document Outline:
* Project Title
* Release Notes
* Getting Started
* Prerequisites
* Code Structure Description
* Key Input Files Preparation
* Usage
* Additional notes
* License
* Authors
* Acknowledgements

# Project Title

*NCHRP 08-145 - Utilizing Cooperative Automated Transportation (CAT) Data to Enhance Freeway Operational Strategies. For more information, please visit: https://apps.trb.org/cmsfeed/TRBNetProjectDisplay.asp?ProjectID=4956*

## Release Notes

#### Release 1.0.0 (April 11, 2023)
- Initial release

## Getting Started

*Download the source code files. The source code was developed for the Interstate 210 (I-210) Eastbound freeway network built in the open-sourced microscopic simulation software SUMO version 1.15.0. Modifications to the code may be necessary if other simulation models or simulation software are used.*

# Prerequisites
- Python 3.6 (or higher)

Python Package Requirements:

- numpy == 1.19.5
- pandas == 1.1.5
- traci == 1.14.1
- sumolib == 1.14.1
- xgboost == 1.6.0

Related functions:

- freewayControl
- connectedEnv
- TCARandom
- bsm_emulator
- queue_estimator


# Code Structure Description
The primary components of the source code comprise:

* **run.py** : The principal file responsible for executing the traffic simulation of I-210 network between milepost 28 and milepost 38 under the selected ramp metering control algorithm.

* **connectedEnv.py** : A module containing information collection functions, featuring essential classes for utilizing Basic Safety Messages (BSMs) to extract traffic data in SUMO. The classes are divided into two categories: 1) gathering trajectory data from BSMs, and 2) employing trajectory data to estimate bottleneck-related information, such as travel time from upstream on-ramps to bottlenecks and traffic flow and density surrounding bottlenecks.

* **freewayControl.py** : A module encompassing control algorithm functions, which houses vital classes for implementing ramp metering within the microscopic simulation software SUMO. The classes are organized into two sections: 1) encapsulating detectors, including E1, E2, E3 detectors, and signal lights within SUMO, and 2) executing ramp metering algorithms such as fixed-rate, ALINEA, queue-informed ALINEA, Incident-aware ALINEA (feedforward ALINEA), and HERO.

* **Simulation setting files**: Files containing the simulation environment configurations, including network, demand, vehicle types, detectors, and the primary simulation configuration file. All SUMO files are in XML format and located in the SUMO subfolder.

* **Ramp metering setting files**: Files comprising sensor information, including sensor IDs and periods for relevant ramps, stored in a series of JSON files. An independent JSON file (meterConfig.json) defines ramp metering signal information, encompassing phase IDs, green time, minimum red time, maximum red time, and more.

* **Traffic incident setting files (optional)**: Files containing traffic incident information, including start time, vehicle id, route id, vehicle type, lane, edge, position, duration, end time, bottleneck locations, upstream on-ramp location, edges from upstream on-ramp to the incident location, distance from on-ramp to the incident location, bottleneck length, stored in the trafficAccidentConfig.json file.


* **BSM generation functions (Noblis)**: Functions that generate BSM messages utilized by the enhanced control algorithm and queue estimation algorithm. The connected vehicles communication process is simulated through TCARandom.py and bsm_emulator.py files. Key parameters can be adjusted in the control.json file, while road side equipment (RSE) information can be set in the rse_locations.csv file.

* **Queue estimation functions (Noblis)**: Modules, including queue_estimator.py, queue_fx_new4.py, and control+queueEstimator.json, that perform queue estimation functions.


# Key Input Files Preparation
Before running the simualtion, please ensure that the following files/directories exist:

For benchmark scenario:

- a simulation project directory containing necessary files, using SUMO as example, the simulation files should including:     
  - main simulation configuration file (.sumocfg.xml) 
  - route file (.rou.xml, .flows.xml, .vehicleTypes.xml) 
  - network file (.net.xml) 
  - detectors file (.calibrator.xml, .detector.xml)
  - other additional files
- a list of json files of sensor description including senosr IDs and period; 
- a json file (meterConfig.json) of meter description including phase IDs, green time, minimum red time, maximum red time and so on;
- a json file (trafficAccidentConfig.json) specifying traffic incident setting including start time, duration, vehicle type, edge id, link id, position and so on;
- the python package freewayControl.py.

For enhanced ramp metering algorithm scenarios:

- a simulation project directory containing necessary files, using SUMO as example, the simulation files should including:     
  - main simulation configuration file (.sumocfg.xml) 
  - route file (.rou.xml, .flows.xml, .vehicleTypes.xml) 
  - network file (.net.xml) 
  - detectors file (.calibrator.xml, .detector.xml)
  - other additional files
- a list of json files of sensor description including senosr IDs and period; 
- a json file (meterConfig.json) of of meter description including phase IDs, green time, minimum red time, maximum red time and so on;
- a json file (control_queueEstimator.json) storing parameters of the queue estimator;
- a json file (control.json) specifying parameters of connected vehicles, such as communication error, equipment failure error, and so on;
- a csv file (rse_locations.csv) specifying locations of RSUs;
- the python package freewayControl.py;
- the python package connectedEnv.py;
- the python package TCARandom.py;
- the python package bsm_emulator.py;
- the python package queue_estimator.py.


# Usage
To start the simulation and apply ramp metering control algorithms using given cases, run the following script:
```
python run.py
```

To evaluate the results, run the following script:
```
python evaluate.py
```

To use the algorithm from scratch, please follow the steps below:

**Step 1: Setting up the environment**

- Install Python, simualtion software (e.g., SUMO), and the required dependencies.
- Clone or download the source code repository.
- Verify that all necessary files are present, including run.py, connectedEnv.py, freewayControl.py, simulation setting files, ramp metering setting files, BSM generation functions, and queue estimation functions.

**Step 2: Configuring the simulation settings**

Generate a simulation environment, including main road and ramps. Check the XML files for network, demand, vehicle types, detectors, and the main simulation configuration.

**Step 3: Configuring ramp metering settings**

- Generate the sensor description JSON files and update sensor IDs and periods for involved ramps based on the requirements and simulated environment.
- Generate meterConfig.json to maintain ramp metering signal information, including phase IDs, green time, minimum red time, maximum red time, and other relevant settings.

**Step 4: Configuring BSM generation settings**

Update the control.json file to modify key parameters for BSM generation.
Set the road side equipment (RSE) information in the rse_locations.csv file.

**Step 5 (optional): Configuring traffic incident settings**

Update the trafficAccidentConfig.json, if necessary, to modify parameters of traffic incidents that causes bottlenecks downstream of on-ramps.

**Step 6: Running the traffic simulation**

Execute run.py, which will run the traffic simulation under the chosen ramp metering control algorithm. connectedEnv.py will collect information from BSMs to estimate bottleneck-related information, including travel time, traffic flow, and density around bottlenecks. freewayControl.py will implement the selected ramp metering algorithms in SUMO.

**Step 7: Analyzing the results**

Once the simulation is complete, analyze the output data to evaluate the performance of the chosen ramp metering algorithm and its impact on traffic flow and bottleneck formation.

**Step 8: Modifying the control algorithm**

If necessary, make modifications to the control algorithm in freewayControl.py and rerun the simulation to test the impact of the changes.

**Step 9: Finalizing the case study**

After obtaining the results, document the case study, including the chosen simulation settings, ramp metering configurations, control algorithm settings, and the resulting traffic flow and bottleneck performance.

# Additional Notes
The following additional notes are provided to facilitate a better understanding of the program settings and code execution processes.

**ALINEA vs. HERO**

The primary distinction between running the ALINEA and HERO algorithms lies in the run.py file, which loads different control algorithms from freewayControl.py.

**No/Few Incidents vs. Many Incidents**

In many incidents scenario, two major incidents are simulated in SUMO. The incident involved vehicle halted at the middle of the highway, and caused changes in traffic conditions. The primary distinction between running the No/Few Incidents and Many Incidents code is the simulation settings, expecially in SUMO demand files (.rou.xml, .flows.xml).

**Benchmark Ramp Metering Scenario vs. Enhanced Ramp Metering Scenario**

In benchmark scenarios, the basic localized control algorithm (ALINEA) is employed without incorporating CAT data. This means that while running the benchmark scenario, the main program only loads the simulation environments, ramp metering settings, and basic ramp metering control algorithms from the freewayControl.py file.

In enhanced scenarios, the enhanced ramp metering control algorithms (i.e., queue-informed ALINEA, queue-informed HERO, incident-aware ALINEA, incident-aware HERO) are utilized. Therefore, in addition to the files or functions required in the benchmark scenario, the BSM generation module, the queue estimation module, and all other enhanced control algorithms need to be loaded and used.

The enhanced scenarios use different connected vehicle communication settings. Those settings adjustment can be found in the following three files:

- control.json: Defines most of the parameters, includes the equipment failure rate, communication failure rate, CAT infrastructure deployment levels.

- res_locations.csv: Defines the number and locations of road side equipments according to the scenarios settings;

- SUMO demand file (.rou.xml, .flow.xml): Defines the market penetration rate of connected vehicles.

# License
This project is licensed under the MIT License - see the [License.MD](https://github.com/NCHRP-BSM-Traffic-Measures/CAT-Data-For-Freeway-Operations/blob/main/LICENSE) for more details. 

# Contributions
This repository is not actively monitored or updated with pull requests, users looking to make contributions should fork the repository and create their own version in accordance with the License detailed above.

# Acknowledgements
## Citing this code
To cite this code in a publication or report, please cite our associated report/paper and/or our source code. Below is a sample citation for this code:

> Noblis. 2023. _BSM Emulator_(v1) [Source code]. Provided by ITS CodeHub through GitHub.com. Accessed YYYY-MM-DD from `url`.


When you copy or adapt from this code, please include the original URL you copied the source code from and date of retrieval as a comment in your code. Additional information on how to cite can be found in the [ITS CodeHub FAQ](https://its.dot.gov/code/#/faqs).

## Authors
Yu Tang, Fan Zuo, Di Sha, Jingqin Gao, Kaan Ozbay
C2SMART Center
New York University
6 MetroTech Center, Brooklyn, NY 11201
Email: tangyu@nyu.edu

James O'Hara, Matt Samach, Claire Silverstein, Haley Townsend, Meenakshy Vasudevan
Noblis

# Attribution
The National Cooperative Research Program (NCHRP) produces ready-to-implement solutions to the challenges facing transportation professionals. NCHRP is sponsored by the individual state departments of transportation of the American Association of State Highway and Transportation Officials. NCHRP is administered by the Transportation Research Board (TRB), part of the National Academies of Sciences, Engineering, and Medicine, under a Cooperative Agreement with the Federal Highway Administration (FHWA).  Any opinions and conclusions expressed or implied in resulting research products are those of the individuals and organizations who performed the research and are not necessarily those of TRB; the National Academies of Sciences, Engineering, and Medicine; FHWA; or NCHRP sponsors. The source code herein was developed under NCHRP Project 08-145 https://apps.trb.org/cmsfeed/TRBNetProjectDisplay.asp?ProjectID=4956
