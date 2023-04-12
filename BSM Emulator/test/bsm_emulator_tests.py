import unittest
import sys

from bsm_emulator import BSMEmulator

class TestBSMEmulator(unittest.TestCase):
    def setUp(self):
        self.bsm_generator = BSMEmulator("control.json")
        self.vehicle = { 'time':10.0,
                    'id':'flow0.0',
                    'link':'M1_E',
                    'lane':1,
                    'x':139.96063439999998,
                    'y':12875.5581548,
                    'angle':56.76,
                    'speed.mph':66.95161420000001,
                    'accel.fpss':0.0,
                    'type':'car_cv'}
        self.bsm_generator.generate_new_record(self.vehicle)

    def testGenerateNewRecord(self):
        gen_rec = self.bsm_generator.equipped_veh[self.vehicle['id']]
        self.assertEqual(gen_rec['time'],self.vehicle['time'])
        self.assertEqual(gen_rec['x'],self.vehicle['x'])
        self.assertEqual(gen_rec['y'],self.vehicle['y'])
        self.assertEqual(gen_rec['speed'],self.vehicle['speed.mph'] * 0.44704)
        self.assertEqual(gen_rec['heading'],self.vehicle['angle'])
        self.assertEqual(gen_rec['acceleration'],self.vehicle['accel.fpss'] * 0.3048)
        self.assertEqual(gen_rec['link'],self.vehicle['link'])
        self.assertEqual(gen_rec['lane'],self.vehicle['lane'])

    def testUpdateVehRecord(self):
        vehicle_update = { 'time':10.1,
                    'id':'flow0.0',
                    'link':'M2_E',
                    'lane':2,
                    'x':143.96063439999998,
                    'y':12821.5581548,
                    'angle':58.76,
                    'speed.mph':60.95161420000001,
                    'accel.fpss':-0.2,
                    'type':'car_cv'}
        self.bsm_generator.update_veh_record(vehicle_update)
        gen_rec = self.bsm_generator.equipped_veh[vehicle_update['id']]
        self.assertEqual(gen_rec['time'],vehicle_update['time'])
        self.assertEqual(gen_rec['x'],vehicle_update['x'])
        self.assertEqual(gen_rec['y'],vehicle_update['y'])
        self.assertEqual(gen_rec['speed'],vehicle_update['speed.mph'] * 0.44704)
        self.assertEqual(gen_rec['heading'],vehicle_update['angle'])
        self.assertEqual(gen_rec['acceleration'],vehicle_update['accel.fpss'] * 0.3048)
        self.assertEqual(gen_rec['link'],vehicle_update['link'])
        self.assertEqual(gen_rec['lane'],vehicle_update['lane'])

    def testGenerateBSM(self):
        new_bsm = self.bsm_generator.generate_bsm(self.vehicle['id'], 2)
        self.assertEqual(new_bsm['time'],self.vehicle['time'])
        self.assertEqual(new_bsm['x'],self.vehicle['x'])
        self.assertEqual(new_bsm['y'],self.vehicle['y'])
        self.assertEqual(new_bsm['speed'],self.vehicle['speed.mph'] * 0.44704)
        self.assertEqual(new_bsm['heading'],self.vehicle['angle'])
        self.assertEqual(new_bsm['acceleration'],self.vehicle['accel.fpss'] * 0.3048)
        self.assertEqual(new_bsm['link'],self.vehicle['link'])
        self.assertEqual(new_bsm['lane'],self.vehicle['lane'])
        self.assertEqual(new_bsm['brakeStatus'],self.bsm_generator.equipped_veh[self.vehicle['id']]["brake_status"])
        self.assertEqual(new_bsm['brakePressure'],self.bsm_generator.equipped_veh[self.vehicle['id']]['brake_pressure'])
        self.assertEqual(new_bsm['hardBraking'],self.bsm_generator.equipped_veh[self.vehicle['id']]['hard_braking'])
        self.assertEqual(new_bsm['transTo'],2)
        self.assertEqual(new_bsm['transtime'],self.vehicle['time'])
        self.assertEqual(new_bsm['transmission_received_time'],self.vehicle['time'])

    def testCheckBrakes(self):
        gen_rec = self.bsm_generator.equipped_veh[self.vehicle['id']]
        self.assertEqual(gen_rec['brake_status'],'0000')
        self.assertEqual(gen_rec['brake_pressure'],0.0)
        self.assertEqual(gen_rec['hard_braking'],0)

        vehicle_update = { 'time':10.1,
                    'id':'flow0.0',
                    'link':'M2_E',
                    'lane':2,
                    'x':143.96063439999998,
                    'y':12821.5581548,
                    'angle':58.76,
                    'speed.mph':60.95161420000001,
                    'accel.fpss':-0.6562,
                    'type':'car_cv'}
        self.bsm_generator.update_veh_record(vehicle_update)
        gen_rec = self.bsm_generator.equipped_veh[vehicle_update['id']]
        self.assertEqual(gen_rec['brake_status'],'1111')
        self.assertEqual(gen_rec['brake_pressure'],-0.20000976)
        self.assertEqual(gen_rec['hard_braking'],0)

        vehicle_update = { 'time':10.1,
                    'id':'flow0.0',
                    'link':'M2_E',
                    'lane':2,
                    'x':143.96063439999998,
                    'y':12821.5581548,
                    'angle':58.76,
                    'speed.mph':60.95161420000001,
                    'accel.fpss':-12.87,
                    'type':'car_cv'}
        self.bsm_generator.update_veh_record(vehicle_update)
        gen_rec = self.bsm_generator.equipped_veh[vehicle_update['id']]
        self.assertEqual(gen_rec['brake_status'],'1111')
        self.assertEqual(gen_rec['brake_pressure'],-3.922776)
        self.assertEqual(gen_rec['hard_braking'],1)

    def testInRange(self):
        transTo = self.bsm_generator.inRange(self.vehicle['id'])
        self.assertEqual(transTo,1)

        self.bsm_generator.equipped_veh[self.vehicle['id']]['x'] = 386.1
        self.bsm_generator.equipped_veh[self.vehicle['id']]['y'] = 3950.4
        transTo = self.bsm_generator.inRange(self.vehicle['id'])
        self.assertEqual(transTo,2)

        self.bsm_generator.equipped_veh[self.vehicle['id']]['x'] = 5046.3
        self.bsm_generator.equipped_veh[self.vehicle['id']]['y'] = 250.8
        transTo = self.bsm_generator.inRange(self.vehicle['id'])
        self.assertEqual(transTo,3)

    def testTmpIDCheck(self):
        tmp_ID = self.bsm_generator.equipped_veh[self.vehicle['id']]['BSM_tmp_ID']
        self.assertIsInstance(tmp_ID, int)
        self.assertEqual(sys.getsizeof(tmp_ID),32)

        self.bsm_generator.tmp_ID_check(self.vehicle['id'])
        self.assertEqual(self.bsm_generator.equipped_veh[self.vehicle['id']]['BSM_tmp_ID'],tmp_ID)
        self.assertEqual(self.bsm_generator.equipped_veh[self.vehicle['id']]['BSM_time_to_ID_chg'],
            self.bsm_generator.equipped_veh[self.vehicle['id']]['time'] + 300)

        self.bsm_generator.equipped_veh[self.vehicle['id']]['time'] += 300
        self.bsm_generator.tmp_ID_check(self.vehicle['id'])
        self.assertNotEqual(self.bsm_generator.equipped_veh[self.vehicle['id']]['BSM_tmp_ID'],tmp_ID)
        self.assertIsInstance(self.bsm_generator.equipped_veh[self.vehicle['id']]['BSM_tmp_ID'], int)
        self.assertEqual(sys.getsizeof(self.bsm_generator.equipped_veh[self.vehicle['id']]['BSM_tmp_ID']),32)
        self.assertEqual(self.bsm_generator.equipped_veh[self.vehicle['id']]['BSM_time_to_ID_chg'],
            self.bsm_generator.equipped_veh[self.vehicle['id']]['time'] + 300)

    def testRemoveInactiveVeh(self):
        self.bsm_generator.remove_inactive_veh(self.bsm_generator.equipped_veh[self.vehicle['id']]['time'] - 30)
        self.assertEqual(len(self.bsm_generator.equipped_veh.keys()),1)

        self.bsm_generator.remove_inactive_veh(self.bsm_generator.equipped_veh[self.vehicle['id']]['time'] + 1)
        self.assertEqual(len(self.bsm_generator.equipped_veh.keys()),0)    

    def testProcessTimestamp(self):
        trajectories = []
        bsm_ids = ['flow0.0']
        not_in_range = ['flow3.0','flow4.0','flow5.0']
        with open('test_traj.txt') as in_f:
            for line in in_f:
                row = line.strip('\n').split(',')
                trajectories.append({ 'time':float(row[0]),
                    'id':row[1],
                    'link':row[3],
                    'lane':int(row[4]),
                    'x':float(row[5]),
                    'y':float(row[6]),
                    'angle':float(row[7]),
                    'speed.mph':float(row[8]),
                    'accel.fpss':float(row[10]),
                    'type':row[2]})
        bsm_output = self.bsm_generator.process_time_step(trajectories)
        for traj in trajectories:
            if traj['id'] in self.bsm_generator.equipped_veh:
                self.assertEqual(self.bsm_generator.equipped_veh[traj['id']]['time'],traj['time'])
                self.assertEqual(self.bsm_generator.equipped_veh[traj['id']]['x'],traj['x'])
                self.assertEqual(self.bsm_generator.equipped_veh[traj['id']]['y'],traj['y'])
                self.assertEqual(self.bsm_generator.equipped_veh[traj['id']]['speed'],traj['speed.mph'] * 0.44704)
                self.assertEqual(self.bsm_generator.equipped_veh[traj['id']]['heading'],traj['angle'])
                self.assertEqual(self.bsm_generator.equipped_veh[traj['id']]['acceleration'],traj['accel.fpss'] * 0.3048)
                self.assertEqual(self.bsm_generator.equipped_veh[traj['id']]['link'],traj['link'])
                self.assertEqual(self.bsm_generator.equipped_veh[traj['id']]['lane'],traj['lane'])
                if traj['id'] in not_in_range:
                    self.assertNotIn(traj['id'],bsm_ids)
                else:
                    self.assertIn(traj['id'],bsm_ids)
            else:
                self.assertIn(traj['id'],self.bsm_generator.unequipped_veh)               

if __name__ == '__main__':
    unittest.main()