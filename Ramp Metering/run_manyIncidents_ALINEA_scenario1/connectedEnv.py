import numpy as np
import pandas as pd
import copy
import bsm_emulator

class ConnectedVehicleTrajectory:
    def __init__(self, env):
        self.env = env
        self.connectedVehIDs = set()
        self.connectedVehStates = []
        self.bsmGenerator = bsm_emulator.BSMEmulator("control.json")
        self.bsmOutput = []

    def updateConnectedVehIDs(self):
        vehIDsIn = self.env.sumo.simulation.getDepartedIDList()
        connectedVehIDsIn = []
        for vehID in vehIDsIn:
            if "_cv" in self.env.sumo.vehicle.getTypeID(vehID):
                connectedVehIDsIn.append(vehID)

        self.connectedVehIDs |= set(connectedVehIDsIn)
        self.connectedVehIDs -= set(self.env.sumo.simulation.getArrivedIDList())

    def updateConnectedVehStates(self):
        self.connectedVehStates = []
        for vehID in self.connectedVehIDs:
            x, y = self.env.sumo.vehicle.getPosition(vehID)
            self.connectedVehStates.append({
                "time": self.env.sumo.simulation.getTime(),
                "id": vehID,
                "type": self.env.sumo.vehicle.getTypeID(vehID),
                "link": self.env.sumo.vehicle.getRoadID(vehID),
                "lane": self.env.sumo.vehicle.getLaneIndex(vehID),
                "x": x*3.28084, # convert meter to feet.
                "y": y*3.28084, # convert meter to feet.
                "angle": self.env.sumo.vehicle.getAngle(vehID),
                "speed.mph": self.env.sumo.vehicle.getSpeed(vehID)*2.23694, # convert meter/second to mile/hour.
                "pos": self.env.sumo.vehicle.getLanePosition(vehID)*3.28084, # convert meter to feet.
                "accel.fpss": self.env.sumo.vehicle.getAcceleration(vehID)*3.28084 # convert meter to feet.
            })

    def generateBSMOutput(self):
        self.bsmOutput = self.bsmGenerator.process_time_step(self.connectedVehStates)


class BSMKeeper:
    def __init__(self, env, obsPeriod):
        # period: time length of BSMs in seconds.
        self.env = env
        self.rawTrajectory = ConnectedVehicleTrajectory(env)

        # Convert unit from in seconds to in unit of simulation step size.
        self.obsPeriod = self.env.convertUnit(obsPeriod)
        
        self.BSMs = [[] for _ in range(self.obsPeriod)]

        # Trajectory from BSM.
        self.sampledTraj = pd.DataFrame([])

    def collectBSM(self):
        # Generate a BSM.
        self.rawTrajectory.updateConnectedVehIDs()
        self.rawTrajectory.updateConnectedVehStates()
        self.rawTrajectory.generateBSMOutput()

        # Keep the BSM at each time step.
        self.BSMs[(self.env.step-1)%self.obsPeriod] = self.rawTrajectory.bsmOutput

    def generateSampledTrajectories(self):
        df = []
        for message in self.BSMs:
            df.extend(message)
        self.sampledTraj = pd.DataFrame(df)[["Vehicle_ID", "time", "x", "y", "speed", "link"]]
        

#################### Estimator based on connected vehicles ####################

class FlowEstimator:
    """
    Use trajectories of connected vehicles to estimate flows through a certain road section.
    Note that the estimator is only appropriate for smooth sections.
    """
    def __init__(self, obsPeriod, boundary, links, cvRate, smoothStep):
        # obsPeriod is in unit of hours.

        self.unsmoothedFlow = 0 # vehicles per hour.
        self.smoothedFlow = 0 # vehicles per hour.

        # boundary: ((x1, y1), (x2, y2))
        # (x1, y1) and (x2, y2) should be set carefully such that 
        # isPassing(x, y) returns one (resp. zero) once (x, y) crosses (does not cross) the boundary.
        x1, y1 = boundary[0]
        x2, y2 = boundary[1]
        self.isPassing = lambda x, y: (((x2-x1) * (y-y1) - (y2-y1) * (x-x1)) > 0) * 1

        self.obsPeriod = obsPeriod
        self.links = links
        self.cvRate = cvRate
        self.smoothStep = smoothStep

        self.passingVehIDs = []

        self.warehouse = {
            "unsmoothedFlow": [self.unsmoothedFlow],
            "smoothedFlow": [self.smoothedFlow]
        }

    def countPassingVehs(self, traj):
        # Only work when the boundary and the trajectories are smooth.
        # Does not support rolling horizon count.

        # traj: a dataframe with columns: "Vehicle_ID", "time", "x", "y", "speed" and "link".

        count = 0

        # sampled trajectory is a dataframe.
        vehIDs = pd.unique(traj["Vehicle_ID"])
        for vehID in vehIDs:
            if vehID in self.passingVehIDs:
                # Skip if the vehicle has passed the boundary.
                continue

            vehTraj = traj[traj["Vehicle_ID"]==vehID].sort_values(by="time")
            if len(vehTraj) == 1:
                # Skip in case of single record. 
                continue

            # passingIndex is a sequence of 0 and 1, e.g. 0, 0, ..., 1, 1, ... 1, that indicates whether trajectory crosses the boundary.
            passingIndex = self.isPassing(np.array(vehTraj["x"]), np.array(vehTraj["y"]))
            if (np.sum(passingIndex)==0):
                # Skip since the vehicle is upstream of the boundary.
                pass

            elif (np.sum(passingIndex)==len(passingIndex)):
                # The vehicle trajectory is downstream of the boundary.
                self.passingVehIDs.append(vehID)
                if vehTraj["link"].iloc[0] in self.links:
                    count += 1
            else:
                # The vehicle trajectory crosses the boundary.
                self.passingVehIDs.append(vehID)

                # breakPoint is the index of trajectory point immediately upstream of the bounary.
                breakPoint = np.where(np.diff(passingIndex)==1)[0][0]
                if vehTraj["link"].iloc[breakPoint] in self.links:
                    # Check the passing point is on target links.
                    count += 1

        return count

    def estimateFlow(self, traj):
        return self.countPassingVehs(traj) / (self.cvRate * self.obsPeriod)

    def updateEstimation(self, traj):
        self.unsmoothedFlow = self.estimateFlow(traj)
        self.warehouse["unsmoothedFlow"].append(self.unsmoothedFlow)

        self.smoothedFlow = np.mean(self.warehouse["unsmoothedFlow"][-self.smoothStep:])
        self.warehouse["smoothedFlow"].append(self.smoothedFlow)


class TravelTimeEstimator:
    """
    Use trajectories of connected vehicles to estimate travel time (in hours) from upstream boundary to downstream boundary.
    """
    def __init__(self, upBoundary, downBoundary, links, distance, maxSpeed=120):

        self.distance = distance
        self.maxSpeed= maxSpeed

        self.speed = maxSpeed
        self.travelTime = distance / maxSpeed  # Assume 120 km/h. Travel time is in unit of hours.

        # upBoundary: ((upX1, upY1), (upX2, upY2))
        # Important: (upX1, upY1) and (upX2, upY2) should be set carefully such that 
        # isPassingUpBoundary(x, y) returns one (resp. zero) once (x, y) crosses (does not cross) the upstream boundary.
        upX1, upY1 = upBoundary[0]
        upX2, upY2 = upBoundary[1]
        self.isPassingUpBoundary = lambda x, y: ((upX2-upX1)*(y-upY1) - (upY2-upY1)*(x-upX1) > 0) * 1

        # downBoundary: ((downX1, downY1), (downX2, downY2))
        # Important: (downX1, downY1) and (downX2, downY2) should be set carefully such that 
        # isPassingDownBoundary(x, y) returns one (resp. zero) once (x, y) crosses (does not cross) the downstream boundary.
        downX1, downY1 = downBoundary[0]
        downX2, downY2 = downBoundary[1]
        self.isPassingDownBoundary = lambda x, y: ((downX2-downX1)*(y-downY1) - (downY2-downY1)*(x-downX1) > 0) * 1

        self.links = links

        self.passingVehIDs = []

        self.warehouse = {
            "speed": [self.speed], 
            "travelTime": [self.travelTime]
        }

    def estimateSpeed(self, traj):
        # traj: a dataframe with columns: "Vehicle_ID", "time", "x", "y", "speed" and "link".
        # Note that the speed is in unit of meter per second.

        targetTraj = traj[traj["link"].isin(self.links)]
        x = np.array(targetTraj["x"])
        y = np.array(targetTraj["y"])
        isWithinBoundaries = (self.isPassingUpBoundary(x, y) == 1) & (self.isPassingDownBoundary(x, y) == 0)

        speed = targetTraj[isWithinBoundaries]["speed"]

        if len(speed) == 0:
            return self.maxSpeed
        else:
            # Convert meter per second to kilometer per hour.
            return np.mean(speed) * 3.6 + 1e-4 

    def estimateTravelTime(self, traj):
        self.speed = self.estimateSpeed(traj)
        self.travelTime = self.distance / self.speed # travel time in unit of hours.

    def updateEstimation(self, traj):
        self.estimateTravelTime(traj)

        self.warehouse["speed"].append(self.speed) # speed in unit of kilometer per hour.
        self.warehouse["travelTime"].append(self.travelTime) # travel time in unit of hours.



class VehicleNumberEstimator:
    def __init__(self, upBoundary, downBoundary, links, cvRate, timeWindow, smoothStep, deltaT=1):

        # upBoundary: ((upX1, upY1), (upX2, upY2))
        # (upX1, upY1) and (upX2, upY2) should be set carefully such that 
        # isPassingUpBoundary(x, y) returns one (resp. zero) once (x, y) crosses (does not cross) the upstream boundary.
        upX1, upY1 = upBoundary[0]
        upX2, upY2 = upBoundary[1]
        self.isPassingUpBoundary = lambda x, y: ((upX2-upX1)*(y-upY1) - (upY2-upY1)*(x-upX1) > 0) * 1

        # downBoundary: ((downX1, downY1), (downX2, downY2))
        # (downX1, downY1) and (downX2, downY2) should be set carefully such that 
        # isPassingDownBoundary(x, y) returns one (resp. zero) once (x, y) crosses (does not cross) the downstream boundary.
        downX1, downY1 = downBoundary[0]
        downX2, downY2 = downBoundary[1]
        self.isPassingDownBoundary = lambda x, y: ((downX2-downX1)*(y-downY1) - (downY2-downY1)*(x-downX1) > 0) * 1

        self.links = links

        self.cvRate = cvRate
        self.timeWindow = timeWindow
        self.smoothStep = smoothStep
        self.deltaT = deltaT

        self.unsmoothedVehNum = 0

        self.smoothedVehNum = 0

        self.warehouse = {
            "unsmoothedVehNum": [self.unsmoothedVehNum],
            "smoothedVehNum": [self.smoothedVehNum],
        }

    def estimateVehicleNum(self, traj, currentTime):
        # traj: a dataframe with columns: "Vehicle_ID", "time", "x", "y", "speed" and "link".

        # Target trajectory is the collection of current trajectories on the target links.
        targetTraj = traj[(np.abs(traj["time"]-currentTime)<(self.timeWindow-1+1e-4)) & (traj["link"].isin(self.links))]

        x = np.array(targetTraj["x"])
        y = np.array(targetTraj["y"])
        isWithinBoundaries = (self.isPassingUpBoundary(x, y) == 1) & (self.isPassingDownBoundary(x, y) == 0)

        return len(targetTraj[isWithinBoundaries]) / (self.timeWindow / self.deltaT) / self.cvRate


    def updateEstimation(self, traj, currentTime):
        self.unsmoothedVehNum = self.estimateVehicleNum(traj, currentTime)
        self.warehouse["unsmoothedVehNum"].append(self.unsmoothedVehNum)
        

        self.smoothedVehNum = np.mean(self.warehouse["unsmoothedVehNum"][-self.smoothStep:])
        self.warehouse["smoothedVehNum"].append(self.smoothedVehNum)


class TrafficDensityEstimator(VehicleNumberEstimator):
    def __init__(self, upBoundary, downBoundary, links, distance, cvRate, timeWindow, smoothStep, deltaT=1):
        super().__init__(upBoundary, downBoundary, links, cvRate, timeWindow, smoothStep, deltaT)

        self.distance = distance

        self.unsmoothedDensity = 0
        self.smoothedDensity = 0

        self.warehouse["unsmoothedDensity"] = [self.unsmoothedDensity]
        self.warehouse["smoothedDensity"] = [self.smoothedDensity]

    def updateEstimation(self, traj, currentTime):
        self.unsmoothedVehNum = self.estimateVehicleNum(traj, currentTime)
        self.unsmoothedDensity = self.unsmoothedVehNum / self.distance

        self.warehouse["unsmoothedVehNum"].append(self.unsmoothedVehNum)
        self.warehouse["unsmoothedDensity"].append(self.unsmoothedDensity)

        self.smoothedVehNum = np.mean(self.warehouse["unsmoothedVehNum"][-self.smoothStep:])
        self.smoothedDensity = np.mean(self.warehouse["unsmoothedDensity"][-self.smoothStep:])

        self.warehouse["smoothedVehNum"].append(self.smoothedVehNum)
        self.warehouse["smoothedDensity"].append(self.smoothedDensity)