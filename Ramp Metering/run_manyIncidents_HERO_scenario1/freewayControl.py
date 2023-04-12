import numpy as np

class SimulationEnvironment:
    def __init__(self, sumo):
        self.sumo = sumo
        self.deltaT = sumo.simulation.getDeltaT()
        self.step = 0
        self.simTime = 0

    def update(self):
        self.sumo.simulationStep()

        self.step += 1
        self.simTime += self.deltaT
        
    def convertUnit(self, t):
        return round(t/self.deltaT)

#################################################
#################################################
#################### Sensors ####################
#################################################
#################################################

class InductionLoop:
    def __init__(self, env, inductionLoopIDs, period, occThrehsold=30):

        self.env = env

        self.inductionLoopIDs = inductionLoopIDs
        self.numInductionLoop = len(inductionLoopIDs)
        self.period = self.env.convertUnit(period)

        self.passingVehEachStep = [np.zeros(self.period) for _ in range(self.numInductionLoop)]
        self.laneFlow = [np.sum(numVeh)/(self.period / self.env.convertUnit(3600)) for numVeh in self.passingVehEachStep]
        self.edgeFlow = np.sum(self.laneFlow)

        self.laneOccEachStep = [np.zeros(self.period) for _ in range(self.numInductionLoop)]
        self.laneOcc = [np.mean(occ) for occ in self.laneOccEachStep]
        self.edgeOcc = np.mean(self.laneOcc)

        self.laneSpeedEachStep = [np.zeros(self.period) for _ in range(self.numInductionLoop)]
        self.laneSpeed = [np.mean(speed) for speed in self.laneSpeedEachStep]
        self.edgeSpeed = np.mean(self.laneSpeed)

        self.occThreshold = occThrehsold
        self.queueIndicator = False

        self.warehouse = {
            "laneFlow": [self.laneFlow],
            "edgeFlow": [self.edgeFlow],
            "laneOcc": [self.laneOcc],
            "edgeOcc": [self.edgeOcc],
            "laneSpeed": [self.laneSpeed],
            "edgeSpeed": [self.edgeSpeed],
            "queueIndicator": [self.queueIndicator]
        }

    def collectData(self):
        for i, inductionLoopID in enumerate(self.inductionLoopIDs):
            numVeh, occ = 0, 0

            vehData = self.env.sumo.inductionloop.getVehicleData(inductionLoopID)
            for data in vehData:
                # data is a tuple (veh_id, veh_length, entry_time, exit_time, vType)
                # Note that current time window is (simTime-deltaT, simTime].
                if data[3] < 0:
                    # Vehicle not passing the induction loop.
                    occ += (self.env.simTime - max(self.env.simTime-self.env.deltaT, data[2])) / self.env.deltaT * 100

                elif data[3] > (self.env.simTime - self.env.deltaT+1e-5):
                    numVeh += 1
                    occ += (data[3] - max(self.env.simTime-self.env.deltaT, data[2])) / self.env.deltaT * 100

            self.passingVehEachStep[i][(self.env.step - 1) % self.period] = numVeh

            self.laneOccEachStep[i][(self.env.step - 1) % self.period] = occ

            self.laneSpeedEachStep[i][(self.env.step - 1) % self.period] = self.env.sumo.inductionloop.getLastStepMeanSpeed(inductionLoopID)

    def updateFlowMeasurement(self):
        self.laneFlow = [np.sum(numVeh)/(self.period / self.env.convertUnit(3600)) for numVeh in self.passingVehEachStep]
        self.edgeFlow = np.sum(self.laneFlow)
        
        self.warehouse["laneFlow"].append(self.laneFlow)
        self.warehouse["edgeFlow"].append(self.edgeFlow)

    def updateOccMeasurement(self):
        self.laneOcc = [np.mean(occ) for occ in self.laneOccEachStep]
        self.edgeOcc = np.mean(self.laneOcc)
        self.queueIndicator = self.edgeOcc > self.occThreshold

        # Save data.
        self.warehouse["laneOcc"].append(self.laneOcc)
        self.warehouse["edgeOcc"].append(self.edgeOcc)
        self.warehouse["queueIndicator"].append(self.queueIndicator)

    def updateSpeedMeasurement(self):
        self.laneSpeed = [np.mean(speed) for speed in self.laneSpeedEachStep]
        self.edgeSpeed = np.mean(self.laneSpeed)

        self.warehouse["laneSpeed"].append(self.laneSpeed)
        self.warehouse["edgeSpeed"].append(self.edgeSpeed)


    def run(self):
        self.collectData()

        if self.env.step % self.period == 0:
            self.updateFlowMeasurement()
            self.updateOccMeasurement()
            self.updateSpeedMeasurement()



class LaneAreaDetector:
    def __init__(self, env, detectorIDs, period):
        self.env = env
        self.detectorIDs = detectorIDs
        self.period = self.env.convertUnit(period)

        self.numVehEachLane = [0 for _ in self.detectorIDs]
        self.numQueueingVehEachLane = [0 for _ in self.detectorIDs]
        self.vehSpeedEachLane = [0 for _ in self.detectorIDs]

        self.numVeh = np.sum(self.numVehEachLane)
        self.numQueueingVeh = np.sum(self.numQueueingVehEachLane)
        self.vehSpeed = np.mean(self.vehSpeedEachLane)

        self.warehouse = {
            "numVehEachLane": [self.numVehEachLane],
            "numVeh": [self.numVeh],
            "numQueueingVehEachLane": [self.numQueueingVehEachLane],
            "numQueueingVeh": [self.numQueueingVeh],
            "vehSpeedEachLane": [self.vehSpeedEachLane],
            "vehSpeed": [self.vehSpeed]
        }

    def updateVehicleNumberMeasurement(self):
        self.numVehEachLane = []
        for detectorID in self.detectorIDs:
            self.numVehEachLane.append(self.env.sumo.lanearea.getLastStepVehicleNumber(detectorID))
        self.numVeh = np.sum(self.numVehEachLane)

        # Save vehicle number on each lane.
        self.warehouse["numVehEachLane"].append(self.numVehEachLane)
        # Save total vehicle number over all lanes.
        self.warehouse["numVeh"].append(self.numVeh)

    def updateQueueingVehicleNumberMeasurement(self):
        self.numQueueingVehEachLane = []
        for detectorID in self.detectorIDs:
            self.numQueueingVehEachLane.append(self.env.sumo.lanearea.getJamLengthVehicle(detectorID))
        self.numQueueingVeh = np.sum(self.numQueueingVehEachLane)

        # Save queueing (halting) vehicle number on each lane.
        self.warehouse["numQueueingVehEachLane"].append(self.numQueueingVehEachLane)
        # Save total queueing (halting) vehicle number over all lanes.
        self.warehouse["numQueueingVeh"].append(self.numQueueingVeh)

    def updateVehicleSpeedMeasurement(self):
        self.vehSpeedEachLane = []
        for detectorID in self.detectorIDs:
            self.vehSpeedEachLane.append([self.env.sumo.lanearea.getLastStepMeanSpeed(detectorID)])
        self.vehSpeed = np.mean(self.vehSpeedEachLane)

        self.warehouse["vehSpeedEachLane"].append(self.vehSpeedEachLane)
        self.warehouse["vehSpeed"].append(self.vehSpeed)
        
    
    def run(self):
        # Update every second.
        self.updateVehicleNumberMeasurement()
        self.updateQueueingVehicleNumberMeasurement()
        self.updateVehicleSpeedMeasurement()


class MultiEntryExitDetector:
    def __init__(self, env, detectorID, period):
        self.env = env
        self.detectorID = detectorID
        self.period = self.env.convertUnit(period)

        self.numVeh = 0
        self.meanTravelTime = 0
        self.numStop = 0
        self.vehSpeed = 0
        self.vmtvht = 0

        self.warehouse = {
            "numVeh": [self.numVeh],
            "meanTravelTime": [self.meanTravelTime],
            "numStop": [self.numStop],
            "vehSpeed": [self.vehSpeed],
            "vmtvht": [self.vmtvht]
        }

    def updateVehicleNumberMeasurement(self):
        self.numVeh = self.env.sumo.multientryexit.getLastStepVehicleNumber(self.detectorID)
        # Save total vehicle number within the detector.
        self.warehouse["numVeh"].append(self.numVeh)


    def updateVehSpeedMeasurement(self):
        self.vehSpeed = self.env.sumo.multientryexit.getLastStepMeanSpeed(self.detectorID)
        self.warehouse["vehSpeed"].append(self.vehSpeed)

    def updateTravelTimeMeasurement(self):
        self.meanTravelTime = self.env.sumo.multientryexit.getLastIntervalMeanTravelTime(self.detectorID)
        # Save mean travel time through the detector within the observation period.
        self.warehouse["meanTravelTime"].append(self.meanTravelTime)

    
    def updateNumStopMeasurement(self):
        self.numStop = self.env.sumo.multientryexit.getLastIntervalMeanHaltsPerVehicle(self.detectorID)
        self.warehouse["numStop"].append(self.numStop)
    
    def updateVMTVHTMeasurement(self):
        self.vmtvht = self.env.sumo.multientryexit.getLastStepMeanSpeed(self.detectorID)
        self.warehouse["vmtvht"].append(self.vmtvht)
    
    def run(self):
        # Update every second.
        self.updateVehicleNumberMeasurement()
        self.updateVehSpeedMeasurement()
        self.updateVMTVHTMeasurement()
        
        # Update every interval.
        if self.env.step % self.period == 0:
            self.updateTravelTimeMeasurement()
            self.updateNumStopMeasurement()



################################################################
################################################################
#################### Queue Length Estimator ####################
################################################################
################################################################

class KalmanFilterBasedVehicleNumberObserver:
    """
    N(k+1) = N(k) + T*(inflow - outflow) + gain*(kappa*measurement - N(k))
    """
    def __init__(self, gain, maxNumVeh, kappa=1, initialNumVeh=0):
        self.gain = gain
        self.maxNumVeh = maxNumVeh
        self.kappa = kappa

        self.numVeh = initialNumVeh
        self.warehouse = {"numVeh": [self.numVeh]}

    def updateEstimation(self, inflow, outflow, T, numVehMeasurement):
        # Note that measurement is vehicle number.
        # T is in unit of hours.
        self.numVeh = self.numVeh + T*(inflow-outflow) + self.gain*(self.kappa*numVehMeasurement-self.numVeh)
        self.numVeh = np.clip(self.numVeh, 0, self.maxNumVeh)

        self.warehouse["numVeh"].append(self.numVeh)
    

class KalmanFilterBasedTrafficDensityObserver(KalmanFilterBasedVehicleNumberObserver):
    def __init__(self, gain, maxNumVeh, distance, kappa=1, initialNumVeh=0):
        # Initial value is vehicle number.
        super().__init__(gain, maxNumVeh, kappa, initialNumVeh)

        self.distance = distance
        self.density = self.numVeh / self.distance

        self.warehouse["density"] = [self.density]

    def updateEstimation(self, inflow, outlfow, T, densityMeasurement):
        # Note that measurement is density.
        self.numVeh = self.numVeh + T*(inflow - outlfow) + self.gain*(self.kappa*densityMeasurement*self.distance-self.numVeh)
        self.numVeh = np.clip(self.numVeh, 0, self.maxNumVeh)

        self.density = self.numVeh / self.distance

        self.warehouse["numVeh"].append(self.numVeh)
        self.warehouse["density"].append(self.density)
        

##########################################################################
##########################################################################
#################### Ramp Meters & Control Algorithms ####################
##########################################################################
##########################################################################

class RampMeter:
    """
    Fix green time, adjust red time.
    """
    def __init__(self, env, meterID, phaseIDs, numRampLane, numVehPerGreenPerLane,
                 greenPhaseLen, redPhaseLen, minRedPhaseLen, maxRedPhaseLen):
        self.env = env
        self.meterID = meterID
        self.phaseIDs = phaseIDs

        self.numRampLane = numRampLane
        self.numVehPerGreenPerLane = numVehPerGreenPerLane

        self.greenPhaseLen = self.env.convertUnit(greenPhaseLen)
        self.redPhaseLen = self.env.convertUnit(redPhaseLen)
        self.minRedPhaseLen = self.env.convertUnit(minRedPhaseLen)
        self.maxRedPhaseLen = self.env.convertUnit(maxRedPhaseLen)

        self.timer = 0

    def generateMeteringPlan(self, phaseLens, period):
        # Given a list of phase lengths, we generate a metering plan, i.e. a phase sequence at each time step, over a certain period.

        self.meteringPlan = []

        # Repeat phase sequence of cycles.
        for _ in range(period // np.sum(phaseLens)): 
            for phaseLen, phaseID in zip(phaseLens, self.phaseIDs):
                self.meteringPlan.extend(phaseLen*[phaseID])
                
        # Fill incomplete metering plan by extending the last phase.
        self.meteringPlan.extend([self.phaseIDs[-1]]*(period % np.sum(phaseLens)))

    def setPhase(self, phaseID):
        self.env.sumo.trafficlight.setPhase(self.meterID, phaseID)

    def run(self):
        self.setPhase(self.meteringPlan[self.timer])
        self.timer = 0 if (self.timer+1) == len(self.meteringPlan) else (self.timer+1)

    def convertRedPhaseLenToMeteredFlow(self, redPhaseLen, meteringUpdateFreq):
        # Given redPhaseLen in unit of simulation step size, compute metered flow (veh/h).
        numCycle = (self.env.convertUnit(3600) // meteringUpdateFreq) * (meteringUpdateFreq // (self.greenPhaseLen + redPhaseLen))
        return numCycle * self.numVehPerGreenPerLane * self.numRampLane
    
    def convertMeteredFlowToRedPhaseLen(self, meteredFlow, meteringUpdateFreq):
        # Given metered flow (veh/h), compute red phase length in unit of simualtion step.
        numCycle = round(meteredFlow / ((self.env.convertUnit(3600) // meteringUpdateFreq)*self.numVehPerGreenPerLane*self.numRampLane))
        return (meteringUpdateFreq // numCycle) - self.greenPhaseLen


class FixedRateController:
    def __init__(self, env, meter, hasRampQueueOverride=True, meteringUpdateFreq=30):
        self.env = env
        self.meter = meter

        # If hasQueueOverride is true, meteringUpdateFreq should be specified in advance.
        self.hasRampQueueOverride = hasRampQueueOverride
        if hasRampQueueOverride:
            self.meteringUpdateFreq = self.env.convertUnit(meteringUpdateFreq)
        else:
            self.meteringUpdateFreq = self.meter.greenPhaseLen + self.meter.redPhaseLen

        # Initialize metering plan.
        self.meter.generateMeteringPlan([self.meter.greenPhaseLen, self.meter.redPhaseLen], self.meteringUpdateFreq)

        # Save initial length of red time.
        self.warehouse = {'redPhaseLen': [self.meter.redPhaseLen]}
            
    def updateMeteringPlan(self, **measurement):
        # If hasQueueOverride is true, check whether queue override is required.
        if self.hasRampQueueOverride:

            # Check whether timer is reset correctly.
            if self.meter.timer != 0:
                print("Error in meter timer.")
            
            # If queueIndicator is true, select minimum red length; otherwise, select the nominal value.
            redPhaseLen = self.meter.minRedPhaseLen if measurement["rampQueueIndicator"] else self.meter.redPhaseLen
            self.meter.generateMeteringPlan([self.meter.greenPhaseLen, redPhaseLen], self.meteringUpdateFreq)
            self.warehouse['redPhaseLen'].append(redPhaseLen)
        else:
            # If queue is not considered, metering plan is not updated even when spillback happens.
            pass


class ALINEA:
    """
    ALINEA is an integral control that controls on-ramp flow to keep mainline occupancy around the critical value.
    This implementation also adopts queue override in case of long ramp queue.
    """
    def __init__(self, env, meter, meteringUpdateFreq, gain, mainlineSetpoint, hasRampQueueOverride=True):
        self.env = env
        self.meter = meter

        self.meteringUpdateFreq = self.env.convertUnit(meteringUpdateFreq)

        # Set control parameters.
        # setpoint is either occupancy (%) or density (veh/(km*lane)). 
        self.gain = gain
        self.mainlineSetpoint = mainlineSetpoint

        self.hasRampQueueOverride = hasRampQueueOverride

        self.minMeteredFlow = self.meter.convertRedPhaseLenToMeteredFlow(self.meter.maxRedPhaseLen, self.meteringUpdateFreq)
        self.maxMeteredFlow = self.meter.convertRedPhaseLenToMeteredFlow(self.meter.minRedPhaseLen, self.meteringUpdateFreq)

        # metered flow by mainline.
        self.meteredFlowByALINEA = self.maxMeteredFlow

        # metered flow by ramp.
        self.meteredFlowByRampQueueOverride = self.minMeteredFlow

        # final metered flow and red phase length.
        self.meteredFlowImplemented = self.maxMeteredFlow
        self.meter.redPhaseLen = self.meter.minRedPhaseLen
        self.meter.generateMeteringPlan([self.meter.greenPhaseLen, self.meter.redPhaseLen], self.meteringUpdateFreq)

        # mainlineIndex is ratio of nearby downstream mainline density or occupancy to the setpoint.
        self.mainlineIndex = 0

        # Initialize saver.
        self.warehouse = {
            "meteredFlowByALINEA": [self.meteredFlowByALINEA],
            "meteredFlowByRampQueueOverride": [self.meteredFlowByRampQueueOverride],
            "meteredFlowImplemented": [self.meteredFlowImplemented], 
            "redPhaseLen": [self.meter.redPhaseLen],
            "mainlineIndex": [self.mainlineIndex]
        }

    def computeMainlineIndex(self, **measurement):
        return measurement["nearbyDownstreamMainlineOcc"] / self.mainlineSetpoint

    def computeMeteredFlowByMainline(self, **measurement):
        # Larger measurement value leads to fewer metered flows.
        meteredFlowByALINEA = self.meteredFlowByALINEA - self.gain*(measurement["nearbyDownstreamMainlineOcc"] - self.mainlineSetpoint)
        # Clip so that ALINEAMeteredFlow is bounded.
        meteredFlowByALINEA = np.clip(meteredFlowByALINEA, self.minMeteredFlow, self.maxMeteredFlow)
        
        self.meteredFlowByALINEA = meteredFlowByALINEA
        self.warehouse["meteredFlowByALINEA"].append(self.meteredFlowByALINEA)

        return meteredFlowByALINEA

    def computeMeteredFlowByRamp(self, **measurement):
        if self.hasRampQueueOverride:
            meteredFlowByRampQueueOverride = self.maxMeteredFlow if measurement["rampQueueIndicator"] else self.minMeteredFlow
        else:
            meteredFlowByRampQueueOverride = self.minMeteredFlow

        self.meteredFlowByRampQueueOverride = meteredFlowByRampQueueOverride
        self.warehouse["meteredFlowByRampQueueOverride"].append(self.meteredFlowByRampQueueOverride)

        return meteredFlowByRampQueueOverride

    def updateMeteringPlan(self, **measurement):
        # Before updating metering plan, we first compute mainline index to evaluate the previous metering.
        self.mainlineIndex = self.computeMainlineIndex(**measurement) 
        self.warehouse["mainlineIndex"].append(self.mainlineIndex)

        # Check whether timer is reset correctly.
        if self.meter.timer != 0:
            print("Error in meter timer.")

        # First, compute metered flow by considering mainline.
        meteredFlowByMainline = self.computeMeteredFlowByMainline(**measurement)
        
        # Second, compute metered flow by considering ramps.
        meteredFlowByRamp = self.computeMeteredFlowByRamp(**measurement)

        # Third, determine final metered flow.
        self.meteredFlowImplemented = max(meteredFlowByMainline, meteredFlowByRamp)
        self.warehouse["meteredFlowImplemented"].append(self.meteredFlowImplemented)

        # Finally, convert final metered flow to red phase length.
        self.meter.redPhaseLen = self.meter.convertMeteredFlowToRedPhaseLen(self.meteredFlowImplemented, self.meteringUpdateFreq)
        self.meter.generateMeteringPlan([self.meter.greenPhaseLen, self.meter.redPhaseLen], self.meteringUpdateFreq)
        self.warehouse["redPhaseLen"].append(self.meter.redPhaseLen)


class QueueInformedALINEA(ALINEA):
    """
    QueueInformedALINEA assumes the knowledge of ramp queue, i.e. vehicle number at on-ramps.
    It tries to avoid aggressive queue override strategy.
    """
    def __init__(self, env, meter, meteringUpdateFreq, gain, mainlineSetpoint, allowedMaxRampQueueLen, hasRampQueueOverride=True):
        super().__init__(env, meter, meteringUpdateFreq, gain, mainlineSetpoint, hasRampQueueOverride)

        # In unit of vehicles.
        self.allowedMaxRampQueueLen = allowedMaxRampQueueLen

        # metered flow by ramp queue control.
        self.meteredFlowByRampQueueControl = self.minMeteredFlow

        # rampIndex is the ratio of vehicle number at on-ramp to the allowed maximum vehicle number.
        self.rampIndex = 0

        # Initialize saver.
        self.warehouse["meteredFlowByRampQueueControl"] = [self.meteredFlowByRampQueueControl]
        self.warehouse["rampIndex"] = [self.rampIndex]

    def computeRampIndex(self, **measurement):
        return measurement["rampQueueLen"] / self.allowedMaxRampQueueLen

    def computeMeteredFlowByRamp(self, **measurement):
        # By queue override.
        if self.hasRampQueueOverride:
            meteredFlowByRampQueueOverride = self.maxMeteredFlow if measurement["rampQueueIndicator"] else self.minMeteredFlow
        else:
            meteredFlowByRampQueueOverride = self.minMeteredFlow

        self.meteredFlowByRampQueueOverride = meteredFlowByRampQueueOverride
        self.warehouse["meteredFlowByRampQueueOverride"].append(self.meteredFlowByRampQueueOverride)

        # By queue control.
        meteredFlowByRampQueueControl = (measurement["rampQueueLen"] - self.allowedMaxRampQueueLen) / (self.meteringUpdateFreq/self.env.convertUnit(3600)) + measurement["rampDemand"]
        meteredFlowByRampQueueControl = np.clip(meteredFlowByRampQueueControl, self.minMeteredFlow, self.maxMeteredFlow)

        self.meteredFlowByRampQueueControl = meteredFlowByRampQueueControl
        self.warehouse["meteredFlowByRampQueueControl"].append(self.meteredFlowByRampQueueControl)

        return max(meteredFlowByRampQueueOverride, meteredFlowByRampQueueControl)
        
    def updateMeteringPlan(self, **measurement):
        # Before updating metering plan, we compute mainline index to evaluate the previous metering.
        self.mainlineIndex = self.computeMainlineIndex(**measurement) 
        self.warehouse["mainlineIndex"].append(self.mainlineIndex)

        # Before updating metering plan, we compute ramp index to evaluate the previous metering.
        self.rampIndex = self.computeRampIndex(**measurement)
        self.warehouse["rampIndex"].append(self.rampIndex)

        # Check whether timer is reset correctly.
        if self.meter.timer != 0:
            print("Error in meter timer.")

        # First, compute metered flow based on mainline occupancy.
        meteredFlowByMainline = self.computeMeteredFlowByMainline(**measurement)

        # Second, compute metered flow based on ramp queue.
        meteredFlowByRamp = self.computeMeteredFlowByRamp(**measurement)

        # Third, determine final metered flow.
        self.meteredFlowImplemented = max(meteredFlowByMainline, meteredFlowByRamp)
        self.warehouse["meteredFlowImplemented"].append(self.meteredFlowImplemented)

        # Convert final metered flow to red phase.
        self.meter.redPhaseLen = self.meter.convertMeteredFlowToRedPhaseLen(self.meteredFlowImplemented, self.meteringUpdateFreq)
        self.meter.generateMeteringPlan([self.meter.greenPhaseLen, self.meter.redPhaseLen], self.meteringUpdateFreq)
        self.warehouse["redPhaseLen"].append(self.meter.redPhaseLen)


class FeedforwardALINEA:
    def __init__(self, env, meter, meteringUpdateFreq, distGain, distMainlineSetpoint, hasRampQueueOverride=True):
        self.env = env
        self.meter = meter
        self.meteringUpdateFreq = self.env.convertUnit(meteringUpdateFreq)

        self.distGain = distGain
        self.distMainlineSetpoint = distMainlineSetpoint

        self.hasRampQueueOverride = hasRampQueueOverride

        self.minMeteredFlow = self.meter.convertRedPhaseLenToMeteredFlow(self.meter.maxRedPhaseLen, self.meteringUpdateFreq)
        self.maxMeteredFlow = self.meter.convertRedPhaseLenToMeteredFlow(self.meter.minRedPhaseLen, self.meteringUpdateFreq)


        self.mainlineIndex = 0

        # metered flow by mainline.
        self.meteredFlowByFeedforwardALINEA = self.maxMeteredFlow

        # metered flow by ramp.
        self.meteredFlowByRampQueueOverride = self.minMeteredFlow

        # final metered flow and red phase length.
        self.meteredFlowImplemented = self.maxMeteredFlow
        self.meter.redPhaseLen = self.meter.minRedPhaseLen
        self.meter.generateMeteringPlan([self.meter.greenPhaseLen, self.meter.redPhaseLen], self.meteringUpdateFreq)
    
        # Initialize saver.
        self.warehouse = {
            "meteredFlowByFeedforwardALINEA": [self.meteredFlowByFeedforwardALINEA],
            "meteredFlowByRampQueueOverride": [self.meteredFlowByRampQueueOverride],
            "meteredFlowImplemented": [self.meteredFlowImplemented], 
            "redPhaseLen": [self.meter.redPhaseLen],
            "mainlineIndex": [self.mainlineIndex]
        }
        
    def computeMainlineIndex(self, **measurement):
        return measurement["distDensity"] / self.distMainlineSetpoint

    def computeMeteredFlowByMainline(self, **measurement):
        travelTime = measurement["travelTime"]
        inflow = measurement["inflow"]
        outflow = measurement["outflow"]
        distDensity = measurement["distDensity"]
        bottleneckLen = measurement["bottleneckLen"]

        meteredFlowByFeedforwardALINEA = self.meteredFlowByFeedforwardALINEA + self.distGain * (self.distMainlineSetpoint - max(0, travelTime*(inflow-outflow)/bottleneckLen) - distDensity)
        meteredFlowByFeedforwardALINEA = np.clip(meteredFlowByFeedforwardALINEA, self.minMeteredFlow, self.maxMeteredFlow)

        self.meteredFlowByFeedforwardALINEA = meteredFlowByFeedforwardALINEA
        self.warehouse["meteredFlowByFeedforwardALINEA"].append(self.meteredFlowByFeedforwardALINEA)

        return meteredFlowByFeedforwardALINEA

    def computeMeteredFlowByRamp(self, **measurement):
        if self.hasRampQueueOverride:
            meteredFlowByRampQueueOverride = self.maxMeteredFlow if measurement["rampQueueIndicator"] else self.minMeteredFlow
        else:
            meteredFlowByRampQueueOverride = self.minMeteredFlow

        self.meteredFlowByRampQueueOverride = meteredFlowByRampQueueOverride
        self.warehouse["meteredFlowByRampQueueOverride"].append(self.meteredFlowByRampQueueOverride)

        return meteredFlowByRampQueueOverride

    def updateMeteringPlan(self, **measurement):
        # Before updating metering plan, we first compute mainline index to evaluate the previous metering.
        self.mainlineIndex = self.computeMainlineIndex(**measurement) 
        self.warehouse["mainlineIndex"].append(self.mainlineIndex)

        # Check whether timer is reset correctly.
        if self.meter.timer != 0:
            print("Error in meter timer.")

        # First, compute metered flow by considering mainline.
        meteredFlowByMainline = self.computeMeteredFlowByMainline(**measurement)
        
        # Second, compute metered flow by considering ramps.
        meteredFlowByRamp = self.computeMeteredFlowByRamp(**measurement)

        # Third, determine final metered flow.
        self.meteredFlowImplemented = max(meteredFlowByMainline, meteredFlowByRamp)
        self.warehouse["meteredFlowImplemented"].append(self.meteredFlowImplemented)

        # Finally, convert final metered flow to red phase length.
        self.meter.redPhaseLen = self.meter.convertMeteredFlowToRedPhaseLen(self.meteredFlowImplemented, self.meteringUpdateFreq)
        self.meter.generateMeteringPlan([self.meter.greenPhaseLen, self.meter.redPhaseLen], self.meteringUpdateFreq)
        self.warehouse["redPhaseLen"].append(self.meter.redPhaseLen)

 
class FeedforwardFeedbackALINEA(ALINEA):
    """
    FeedforwardFeedbackALINEA supports
    1) feedback ALINEA always with feedforward ALINEA for both nearby and distant bottlenecks;
    2) feedback ALINEA with temporally activated feedforward ALINEA for abrupt distant bottlenecks, e.g. induced by traffic accidents.
    """
    def __init__(self, env, meter, meteringUpdateFreq, gain, mainlineSetpoint, distGain, distMainlineSetpoint, hasRampQueueOverride=True):
        super().__init__(env, meter, meteringUpdateFreq, gain, mainlineSetpoint, hasRampQueueOverride)

        self.distGain = distGain
        self.distMainlineSetpoint = distMainlineSetpoint

        self.meteredFlowByFeedforwardALINEA = self.maxMeteredFlow
        self.warehouse["meteredFlowByFeedforwardALINEA"] = [self.meteredFlowByFeedforwardALINEA]

    def computeMainlineIndex(self, **measurement):
        if measurement["isBottleneckActivated"]:
            return max(measurement["nearbyDownstreamOcc"]/self.mainlineSetpoint, measurement["distDensity"]/self.distMainlineSetpoint)
        else:
            return measurement["nearbyDownstreamOcc"]/self.mainlineSetpoint
    
    def computeMeteredFlowByMainline(self, **measurement):
        # Compute metered flow based on nearby bottleneck.
        meteredFlowByALINEA = self.meteredFlowByALINEA - self.gain * (measurement["nearbyDownstreamOcc"] - self.mainlineSetpoint)
        meteredFlowByALINEA = np.clip(meteredFlowByALINEA, self.minMeteredFlow, self.maxMeteredFlow)

        self.meteredFlowByALINEA = meteredFlowByALINEA
        self.warehouse["meteredFlowByALINEA"].append(self.meteredFlowByALINEA)

        # Compute metered flow based on distant bottleneck.
        if measurement["isBottleneckActivated"]:
            # If bottleneck is active, estimated travel time, inflow, outflow, density and distance should be provided.
            travelTime = measurement["travelTime"]
            inflow = measurement["inflow"]
            outflow = measurement["outflow"]
            distDensity = measurement["distDensity"]
            bottleneckLen = measurement["bottleneckLen"]

            meteredFlowByFeedforwardALINEA = self.meteredFlowByFeedforwardALINEA + self.distGain * (self.distMainlineSetpoint - max(0, travelTime*(inflow-outflow)/bottleneckLen) - distDensity)
            meteredFlowByFeedforwardALINEA = np.clip(meteredFlowByFeedforwardALINEA, self.minMeteredFlow, self.maxMeteredFlow)
        else:
            # Bottleneck is inactive.
            meteredFlowByFeedforwardALINEA = self.maxMeteredFlow
        
        self.meteredFlowByFeedforwardALINEA = meteredFlowByFeedforwardALINEA
        self.warehouse["meteredFlowByFeedforwardALINEA"].append(self.meteredFlowByFeedforwardALINEA)

        return min(meteredFlowByALINEA, meteredFlowByFeedforwardALINEA)
 

    def updateMeteringPlan(self, **measurement):
        # Before updating metering plan, we first compute mainline index to evaluate the previous metering.
        self.mainlineIndex = self.computeMainlineIndex(**measurement) 
        self.warehouse["mainlineIndex"].append(self.mainlineIndex)

        # Check whether timer is reset correctly.
        if self.meter.timer != 0:
            print("Error in meter timer.")

        # First, compute metered flow by mainline.
        meteredFlowByMainline = self.computeMeteredFlowByMainline(**measurement)

        # Second, compute metered flow by ramp.
        meteredFlowByRamp = self.computeMeteredFlowByRamp(**measurement)

       # Third, determine final metered flow.
        self.meteredFlowImplemented = max(meteredFlowByMainline, meteredFlowByRamp)
        self.warehouse["meteredFlowImplemented"].append(self.meteredFlowImplemented)

        # Convert final metered flow into red phase length.
        self.meter.redPhaseLen = self.meter.convertMeteredFlowToRedPhaseLen(self.meteredFlowImplemented, self.meteringUpdateFreq)
        self.meter.generateMeteringPlan([self.meter.greenPhaseLen, self.meter.redPhaseLen], self.meteringUpdateFreq)
        self.warehouse["redPhaseLen"].append(self.meter.redPhaseLen)
    

class QueueInformedFeedforwardFeedbackALINEA(FeedforwardFeedbackALINEA):
    def __init__(self, env, meter, meteringUpdateFreq, gain, mainlineSetpoint, distGain, distMainlineSetpoint, allowedMaxRampQueueLen, hasRampQueueOverride=True):
        super().__init__(env, meter, meteringUpdateFreq, gain, mainlineSetpoint, distGain, distMainlineSetpoint, hasRampQueueOverride)

        self.allowedMaxRampQueueLen = allowedMaxRampQueueLen

        # metered flow by ramp.
        self.meteredFlowByRampQueueControl = self.minMeteredFlow

        self.warehouse["meteredFlowByRampQueueControl"] = [self.meteredFlowByRampQueueControl]

        self.rampIndex = 0
        self.warehouse["rampIndex"] = [self.rampIndex]

    def computeRampIndex(self, **measurement):
        return measurement["rampQueueLen"] / self.allowedMaxRampQueueLen

    def computeMeteredFlowByRamp(self, **measurement):
        # By queue override.
        if self.hasRampQueueOverride:
            meteredFlowByRampQueueOverride = self.maxMeteredFlow if measurement["rampQueueIndicator"] else self.minMeteredFlow
        else:
            meteredFlowByRampQueueOverride = self.minMeteredFlow

        self.meteredFlowByRampQueueOverride = meteredFlowByRampQueueOverride
        self.warehouse["meteredFlowByRampQueueOverride"].append(self.meteredFlowByRampQueueOverride)

        # By queue control.
        meteredFlowByRampQueueControl = (measurement["rampQueueLen"] - self.allowedMaxRampQueueLen) / (self.meteringUpdateFreq/self.env.convertUnit(3600)) + measurement["rampDemand"]
        meteredFlowByRampQueueControl = np.clip(meteredFlowByRampQueueControl, self.minMeteredFlow, self.maxMeteredFlow)

        self.meteredFlowByRampQueueControl = meteredFlowByRampQueueControl
        self.warehouse["meteredFlowByRampQueueControl"].append(self.meteredFlowByRampQueueControl)

        return max(meteredFlowByRampQueueOverride, meteredFlowByRampQueueControl)
        
    def updateMeteringPlan(self, **measurement):
        # Before updating metering plan, we first compute mainline index to evaluate the previous metering.
        self.mainlineIndex = self.computeMainlineIndex(**measurement) 
        self.warehouse["mainlineIndex"].append(self.mainlineIndex)

        # Before updating metering plan, we compute ramp index to evaluate the previous metering.
        self.rampIndex = self.computeRampIndex(**measurement)
        self.warehouse["rampIndex"].append(self.rampIndex)

        # Check whether timer is reset correctly.
        if self.meter.timer != 0:
            print("Error in meter timer.")

        # First, compute metered flow by mainline.
        meteredFlowByMainline = self.computeMeteredFlowByMainline(**measurement)

        # Second, compute metered flow by queue control.
        meteredFlowByRamp = self.computeMeteredFlowByRamp(**measurement)

        # Third, determine final metered flow.
        self.meteredFlowImplemented = max(meteredFlowByMainline, meteredFlowByRamp)
        self.warehouse["meteredFlowImplemented"].append(self.meteredFlowImplemented)

        # Convert final metered flow into red phase length.
        self.meter.redPhaseLen = self.meter.convertMeteredFlowToRedPhaseLen(self.meteredFlowImplemented, self.meteringUpdateFreq)
        self.meter.generateMeteringPlan([self.meter.greenPhaseLen, self.meter.redPhaseLen], self.meteringUpdateFreq)
        self.warehouse["redPhaseLen"].append(self.meter.redPhaseLen)
            

class HeuristicRampMeteringCoordinator:
    def __init__(self, controllers,
                 activationMainlineThreshold=0.9, deactivationMainlineThreshold=0.8,
                 activationRampThreshold=0.3, deactivationRampThreshold=0.15):

        self.controllers = controllers
        self.numController = len(self.controllers)

        self.activationMainlineThreshold = activationMainlineThreshold
        self.deactivationMainlineThreshold = deactivationMainlineThreshold
        
        self.activationRampThreshold = activationRampThreshold
        self.deactivationRampThreshold = deactivationRampThreshold

        for controller in self.controllers:
            controller.isMaster = False
            controller.isSlave = False
            controller.rampQueueLenByCoordination = None
            controller.meteredFlowByCoordination = None

            controller.warehouse["isMaster"] = [controller.isMaster]
            controller.warehouse["isSlave"] = [controller.isSlave]
            controller.warehouse["rampQueueLenByCoordination"] = [controller.rampQueueLenByCoordination]
            controller.warehouse["meteredFlowByCoordination"] = [controller.meteredFlowByCoordination]

    def determineCoordination(self, measurements):
        for i in range(self.numController-1):
            if self.controllers[i+1].isMaster:
                if  (self.controllers[i+1].rampIndex < self.deactivationRampThreshold) or (self.controllers[i+1].mainlineIndex < self.deactivationMainlineThreshold):
                    self.controllers[i+1].isMaster = False
                    self.controllers[i].isSlave = False 
            else:
                if (self.controllers[i+1].rampIndex > self.activationRampThreshold) and (self.controllers[i+1].mainlineIndex > self.activationMainlineThreshold):
                    self.controllers[i+1].isMaster = True
                    self.controllers[i].isSlave = True
        
            if self.controllers[i+1].isMaster:
                upMax = self.controllers[i].allowedMaxRampQueueLen
                downMax = self.controllers[i+1].allowedMaxRampQueueLen

                self.controllers[i].rampQueueLenByCoordination = upMax * (measurements[i]["rampQueueLen"]+measurements[i+1]["rampQueueLen"]) / (upMax + downMax)
            else:
                self.controllers[i].rampQueueLenByCoordination = None

        for controller in self.controllers:
            controller.warehouse["isMaster"].append(controller.isMaster)
            controller.warehouse["isSlave"].append(controller.isSlave)
            controller.warehouse["rampQueueLenByCoordination"].append(controller.rampQueueLenByCoordination)
    
    def updateMeteringPlan(self, controller, **measurement):
        # Check whether timer is reset correctly.
        if controller.meter.timer != 0:
            print("Error in meter timer.")

        # By mainline.
        meteredFlowByMainline = controller.computeMeteredFlowByMainline(**measurement)

        # By ramp.
        meteredFlowByRamp = controller.computeMeteredFlowByRamp(**measurement)

        # By coordinated control.
        if controller.rampQueueLenByCoordination is not None:
            meteredFlowByCoordination =  (measurement["rampQueueLen"] - controller.rampQueueLenByCoordination) / (controller.meteringUpdateFreq/controller.env.convertUnit(3600)) + measurement["rampDemand"]
            meteredFlowByCoordination = np.clip(meteredFlowByCoordination, controller.minMeteredFlow, controller.maxMeteredFlow)            
            controller.meteredFlowImplemented = max(min(meteredFlowByMainline, meteredFlowByCoordination), meteredFlowByRamp)
        else:
            meteredFlowByCoordination = None
            controller.meteredFlowImplemented = max(meteredFlowByMainline, meteredFlowByRamp)

        controller.meteredFlowByCoordination = meteredFlowByCoordination
        controller.warehouse["meteredFlowByCoordination"].append(controller.meteredFlowByCoordination)
        controller.warehouse["meteredFlowImplemented"].append(controller.meteredFlowImplemented)

        # Convert final metered flow into red phase length.
        controller.meter.redPhaseLen = controller.meter.convertMeteredFlowToRedPhaseLen(controller.meteredFlowImplemented, controller.meteringUpdateFreq)
        controller.meter.generateMeteringPlan([controller.meter.greenPhaseLen, controller.meter.redPhaseLen], controller.meteringUpdateFreq)
        controller.warehouse["redPhaseLen"].append(controller.meter.redPhaseLen)
            

    def updateMeteringPlans(self, measurements):
        for i, controller in enumerate(self.controllers):
            controller.mainlineIndex = controller.computeMainlineIndex(**measurements[i])
            controller.warehouse["mainlineIndex"].append(controller.mainlineIndex)

            controller.rampIndex = controller.computeRampIndex(**measurements[i])
            controller.warehouse["rampIndex"].append(controller.rampIndex)

        self.determineCoordination(measurements)

        for i, controller in enumerate(self.controllers):
            self.updateMeteringPlan(controller, **measurements[i])


