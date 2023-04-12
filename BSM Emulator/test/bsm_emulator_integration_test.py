from bsm_emulator import BSMEmulator
import csv
import time
import statistics

bsm_emulator = BSMEmulator("control.json")

with open("BSMOutput.csv",'w',newline='') as out_f:
    out_f.write("bsm_tmp_id,time,speed,x,y,transtime,transTo,acceleration,brakeStatus,brakePressure,hardBraking,transmission_received_time,heading,link,lane\n")
    with open("TrajSample.csv") as in_f:
        tp = 0.0
        trajectories = []
        bsm_output = []
        times = []
        header = True
        for line in in_f:
            if header:
                header = False
                continue
            row = line.strip('\n').split(',')
            current_time = float(row[0])
            if current_time == tp:
                trajectories.append({'time':current_time,
                    'id':row[1],
                    'link':row[3],
                    'lane':int(row[4]),
                    'x':float(row[5]),
                    'y':float(row[6]),
                    'angle':float(row[7]),
                    'speed.mph':float(row[8]),
                    'accel.fpss':float(row[10]),
                    'type':row[2]})
            else:
                timer = time.time()
                bsm_output += bsm_emulator.process_time_step(trajectories)
                times.append(time.time() - timer)
                trajectories = []
                tp = current_time
                trajectories.append({'time':current_time,
                    'id':row[1],
                    'link':row[3],
                    'lane':int(row[4]),
                    'x':float(row[5]),
                    'y':float(row[6]),
                    'angle':float(row[7]),
                    'speed.mph':float(row[8]),
                    'accel.fpss':float(row[10]),
                    'type':row[2]})
            if len(bsm_output) > 30000:
                print(tp)
                keys = bsm_output[0].keys()
                dict_writer = csv.DictWriter(out_f,keys)
                dict_writer.writerows(bsm_output)
                bsm_output = []

if len(bsm_output) > 0:
    with open('BSMOutput.csv','a',newline='') as out_f:
        keys = bsm_output[0].keys()
        dict_writer = csv.DictWriter(out_f,keys)
        dict_writer.writerows(bsm_output)    

print('\n\nTotal Vehicles: {}\nEquipped Vehicles:{}\nBSMs Generated:{}\nBSMs Transmitted:{}\nComm Failures:{}\n'.format(
    bsm_emulator.total_vehicles,bsm_emulator.equipped_vehicles,bsm_emulator.bsms_generated,bsm_emulator.trans_in_range,bsm_emulator.comm_failures))
print('Average time to process time step {}'.format(statistics.fmean(times)))