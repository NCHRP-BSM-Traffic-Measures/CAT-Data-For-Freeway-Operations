# BSM Emulator

# README Outline:
* Project Description
* Prerequisites
* Usage
	* Building
	* Testing
	* Execution
* Version History and Retention
* License
* Contributions
* Acknowledgements

# Project Description

This is an offshoot of the Trajectory Coversion Algorithm (TCA) upgrading to Python 3.10, streamlining the code to focus only on the generation of Basic Safety Messages following the J2735 standard and improving the processes to decrease processing time. Pandas was removed and data storage was streamlined to use a python dictionary. Roadside equipment (RSE) range checks were changed to use a numpy array for faster processing and the equipped status of individual vehicles to meet market penetration is now determined by the vehicle type field from the microsimulation. Additional features added include modifying equipment failure rates to include roadside equipment and adding communications failure rate which impacts all generated messages.

# Prerequisites

Requires:
- Python 3.10
- numpy >= 1.24.2

# Usage

## Building
No building or compiling is required prior to running the program.

## Testing
Unit and integration test code are in the test folder. Unit test code tests each individual method while integration test code is meant to compare BSM output to the output of the TCA.

To run unit tests:
```
python -m unittest test/bsm_emulator_tests.py
```

To run integration tests you must have a valid vehicle trajectory file and control.json file. Examples are provided in the tests folder. Edit bsm_emulator_integration_test.py to read in the trajectory file then run:
```
python test/bsm_emulator_integration_test.py
```

## Execution
Expected execution is to import the BSMEmulator class into a python module that reads vehicle trajectories from a simulation tool. This module would then call the process_time_step method with the input parameter of a list of vehicle trajectory points containing the following data:  
```
id, time, angle, accel.fpss,speed.mph,link,lane,type
```
The BSM Emulator has the following assumptions about the input data:
- id is unique and persistent for the given vehicle across the time it is in the simulation
- time is in seconds
- angle is the heading of the vehicle in degrees
- accel.fpss is the acceleration of the vehicle in feet per second squared
- speed.mph is the speed of the vehicle in miles per hour
- link is a unique identifier of a single travel direction of a defined roadway stretch
- lane is the lane number the vehicle currently occupies on the link
- type is truck or car and includes "\_cv" to indicate a connected vehicle or "\_hv" to indicate a non-connected vehicle

The BSMEmulator can also be run directly with a csv input file as shown in bsm_emulator_integration_test.py above.


# Version History and Retention
**Status:** This project is in the release phase, no further development is expected.

**Release Frequency:** This is a one-time release with no expected future release versions.

**Retention:** This project will remain publicly accessible indefinitely. 

# License
This project is licensed under the MIT License - see the [License.MD](https://github.com/NCHRP-BSM-Traffic-Measures/CAT-Data-For-Freeway-Operations/blob/main/LICENSE) for more details. 

# Contributions
This repository is not actively monitored or updated with pull requests, users looking to make contributions should fork the repository and create their own version in accordance with the License detailed above.

# Acknowledgements
Example:

## Citing this code
To cite this code in a publication or report, please cite our associated report/paper and/or our source code. Below is a sample citation for this code:

> Noblis. 2023. _BSM Emulator_(v1) [Source code]. Provided by ITS CodeHub through GitHub.com. Accessed YYYY-MM-DD from `url`.


When you copy or adapt from this code, please include the original URL you copied the source code from and date of retrieval as a comment in your code. Additional information on how to cite can be found in the [ITS CodeHub FAQ](https://its.dot.gov/code/#/faqs).