import math
import numpy as np
import json

from TCARandom import Random_generator

"""
Create emulated Basic Safety Messages from SUMO vehicle trajectories by timestamp. Requires json control file
and csv of RSE locations to generate messages. Vehicles are randomly equipped based on market penetration rate and
generate a message when in range of an RSE. Comm Failure rate randomly determines dropped messages. Generated messages
that aren't dropped are returned as a list.
"""

class BSMEmulator(object):

    def __init__(self, control_file):
        """
        Define the bsm emulator with the given json control file, set random rates, load RSE locations and set
        statistics counters to zero.
        """
        with open(control_file) as in_f:
            self.control = json.load(in_f)

        self.random_generator = Random_generator(self.control['randomGen_seed'])
        self.random_generator.add_generator_bit('BSM_Tmp_ID', 32)
        self.random_generator.add_generator_int('CommFailure', 1, 100)
        self.random_generator.add_generator_int('EqiupFailure', 1, 100)

        #Keeps track of vehicle IDs that have already been seen across time steps
        self.equipped_veh = {}
        self.unequipped_veh = []
        self.equip_failure_veh = []
    
        try:
            self.rse_locations = np.genfromtxt(self.control['rse_locations_filename'], delimiter=',',skip_header=True)
            rse_failure = np.array([self.random_generator['EqiupFailure'] for i in range(self.rse_locations.shape[0])])
            self.rse_locations = self.rse_locations[rse_failure > self.control['equip_failure_rate']]
        except (TypeError, KeyError):
            self.rse_locations = None
            print('RSE locations file could not be properly loaded.')

        self.total_vehicles = 0
        self.equipped_vehicles = 0
        self.equipment_failure_vehicles = 0
        self.bsms_generated = 0
        self.trans_in_range = 0
        self.comm_failures = 0

        self.inactive_veh = []

    def process_time_step(self, trajectories):
        """
        Main method that processes a given time step of trajectories and returns a list of BSMs.
        ::trajectories:: a list of dictionaries each representing a vehicle trajectory point with keys of
            id, time, angle, accel.fpss,speed.mph,link,lane,type
        """
        bsm_output = []
        timestamp = None
        for vehicle in trajectories:
            veh_id = vehicle['id']
            timestamp = vehicle['time']
            equipped = '_cv' in vehicle['type'] 
            if equipped and not veh_id in self.equip_failure_veh:
                #Vehicle ID has not been seen before, add it to active vehicles
                if not veh_id in self.equipped_veh.keys():
                    if veh_id in self.inactive_veh:
                        print("Removed equipped vehicle {} has appeared again.".format(veh_id))
                        self.generate_new_record(veh_id)
                    else:
                        self.total_vehicles += 1
                        if self.random_generator['EqiupFailure'] > self.control['equip_failure_rate']:
                            self.equipped_vehicles += 1
                            self.generate_new_record(vehicle)
                        else:
                            self.equip_failure_veh.append(veh_id)
                            continue
                #If vehicle is equipped check if it generated a message received by an RSE
                self.update_veh_record(vehicle)
                self.bsms_generated += 1
                if not self.rse_locations is None:
                    transTo = self.inRange(veh_id)
                else:
                    transTo = 1
                if transTo:
                    self.trans_in_range += 1
                    new_bsm = self.generate_bsm(veh_id, transTo)
                    if self.random_generator['CommFailure'] > self.control['comm_failure_rate']:
                        bsm_output.append(new_bsm)
                    else:
                        self.comm_failures += 1
            elif veh_id not in self.unequipped_veh:
                self.unequipped_veh.append(veh_id)
                self.total_vehicles += 1
        if timestamp:
            #Remove vehicles from active table that haven't been seen for some time
            self.remove_inactive_veh(timestamp - self.control['inactive_veh_threshold'])
        return bsm_output

    def generate_new_record(self, vehicle):
        """ Create a new entry in the active equipped vehicle table for the given vehicle
        """
        veh_id = vehicle['id']
        new_veh_record = {  'BSM_tmp_ID': self.random_generator['BSM_Tmp_ID'],
                            'BSM_time_to_ID_chg': vehicle['time'] + self.control['tmp_id_timeout'],
                            'time':vehicle['time'],
                            'x':vehicle['x'],
                            'y':vehicle['y'],
                            'heading':vehicle['angle'],
                            'acceleration':vehicle['accel.fpss'] * 0.3048, #convert to mpss
                            'speed':vehicle['speed.mph'] * 0.44704, #convert to mps (meter per second)
                            'link':vehicle['link'],
                            'lane':vehicle['lane'],
                            'type':vehicle['type']}
        self.equipped_veh[veh_id] = new_veh_record
        self.checkBrakes(veh_id)

    def update_veh_record(self, vehicle):
        """ Update the active equipped vehicle table for the given vehicle
        """
        veh_id = vehicle['id']
        self.equipped_veh[veh_id]['time'] = vehicle['time']
        self.equipped_veh[veh_id]['x'] = vehicle['x']
        self.equipped_veh[veh_id]['y'] = vehicle['y']
        self.equipped_veh[veh_id]['speed'] = vehicle['speed.mph'] * 0.44704
        self.equipped_veh[veh_id]['heading'] = vehicle['angle']
        self.equipped_veh[veh_id]['acceleration'] = vehicle['accel.fpss'] * 0.3048
        self.equipped_veh[veh_id]['link'] = vehicle['link']
        self.equipped_veh[veh_id]['lane'] = vehicle['lane']
        self.checkBrakes(veh_id)
        self.tmp_ID_check(veh_id)

    def generate_bsm(self, veh_id, transTo):
        """ Create a BSM for the given vehicle transmitted to the given RSE.
        """
        new_bsm = { 'Vehicle_ID':veh_id,
                    'bsm_tmp_id' : self.equipped_veh[veh_id]['BSM_tmp_ID'],
                    'time' : self.equipped_veh[veh_id]['time'],
                    'speed' : self.equipped_veh[veh_id]['speed'],
                    'x' : self.equipped_veh[veh_id]['x'],
                    'y' : self.equipped_veh[veh_id]['y'],
                    'transtime' : self.equipped_veh[veh_id]['time'],
                    'transTo' : transTo,
                    'acceleration' : self.equipped_veh[veh_id]['acceleration'],
                    'brakeStatus' : self.equipped_veh[veh_id]["brake_status"],
                    'brakePressure' : self.equipped_veh[veh_id]['brake_pressure'],
                    'hardBraking' : self.equipped_veh[veh_id]['hard_braking'],
                    'transmission_received_time' : self.equipped_veh[veh_id]['time'],
                    'heading' : self.equipped_veh[veh_id]['heading'],
                    'link' : self.equipped_veh[veh_id]['link'],
                    'lane' : self.equipped_veh[veh_id]['lane'],
                    'type': self.equipped_veh[veh_id]['type']
                }
        return new_bsm

    def checkBrakes(self, veh_id):
        """
        Check brake status of given vehicle using instantaneous acceleration

        :param veh_id: Vehicle whose brakes to check
        """

        # Set brake_status as applied if decelerating more than the defined threshold
        self.equipped_veh[veh_id]['brake_status'] = '0000'
        if self.equipped_veh[veh_id]['acceleration'] <= self.control['brake_threshold']:
            self.equipped_veh[veh_id]['brake_status'] = '1111'

        if self.equipped_veh[veh_id]['acceleration'] <= 0:
            self.equipped_veh[veh_id]['brake_pressure'] =  self.equipped_veh[veh_id]['acceleration']
        else:
            self.equipped_veh[veh_id]['brake_pressure'] = 0.0

        # Set the hard braking (1: true, 0: false) if decelerating greater than 0.4g (J2735 standard) or approx. 3.92266 m/s^2
        self.equipped_veh[veh_id]['hard_braking'] = 0
        if self.equipped_veh[veh_id]['acceleration'] <= -3.92266:
            self.equipped_veh[veh_id]['hard_braking'] = 1

    def inRange(self, veh_id):
        """ Check if given vehicle is within broacast range of an RSE
        """
        veh_pos = (self.equipped_veh[veh_id]['x'],self.equipped_veh[veh_id]['y'])
        try:
            return self.rse_locations[np.sqrt(((self.rse_locations[:,1:] - veh_pos)**2).sum(1)) < self.control['rse_range']][0,0].astype(int)
        except IndexError:
            return None

    def tmp_ID_check(self, veh_id):
        """
        Check if the BSM ID needs to be updated

        :param veh_id:  The vehicle to check
        """

        if self.equipped_veh[veh_id]['time'] >= self.equipped_veh[veh_id]['BSM_time_to_ID_chg'] :
             self.equipped_veh[veh_id]['BSM_tmp_ID'] = self.random_generator['BSM_Tmp_ID']
             self.equipped_veh[veh_id]['BSM_time_to_ID_chg'] = self.equipped_veh[veh_id]['time'] + self.control['tmp_id_timeout']

    def remove_inactive_veh(self, threshold):
        """ Remove vehicles that haven't been seen for given threshold from active equipped vehicles table to save memory
        """
        inactive_veh = [veh_id for veh_id, veh_record in self.equipped_veh.items() if veh_record['time'] < threshold]
        self.inactive_veh += inactive_veh
        for veh_id in inactive_veh:
            del self.equipped_veh[veh_id]

