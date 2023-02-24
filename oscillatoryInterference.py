"""
oscillatoryInterference
By Emma Alexandrov

Provides a GridCell class which implements the grid cell model described in 
Burgess, Barry, and O'Keefe's 2007 article in Hippocampus. 
"""

import math
import numpy as np

class Soma:
    """    
    Implements sub-threshold membrane potential oscillations. 

    ARGUMENTS: 
    - samples_per_second: float; the number of samples to be "recorded" from the 
        simulated somatic oscillator or the grid cell of which it is a part
    - theta_frequency: float (Hertz); the number of cycles per second 
        characteristic of the sub-threshold "theta" oscillation. According to 
        the paper, MEC Layer II cells show such oscillations at 8-9 Hz. 
    - phase_offset: float (radians); the point in the oscillator's cycle from 
        which we wish the oscillation to begin at initialization

    ATTRIBUTES: 
    - .phase_offset: float (radians); the point in the oscillator's cycle from 
        which we wish the oscillation to begin at initialization
    - .angular_frequency: float (radians per second); the number of radians 
        traveled through the oscillator's cycle per second
    - .phase_step: float; the proportion of the oscillator's full cycle 
        accomplished per sampling interval
    - .phase: float (radians, [0, 2 * pi)); the state of the oscillator as 
        position in a cycle of 2-pi radians
    - .phase_history: list; all previous phase values 
    - .activity: float ([0, 1]): the value of the membrane potential as due to 
        the oscillation
    - .activity_history: list; all previous activity values

    METHODS: 
    - .step(): advance the state of the oscillator by one time step of length 
        determined by the samples_per_second parameter
    - .reset(): return the state of the oscillator to its initial value
    """
    def __init__(self, 
                 samples_per_second,
                 theta_frequency, 
                 phase_offset):
        self.phase_offset = phase_offset
        self.angular_frequency = theta_frequency * 2 * math.pi
        self.phase_step = self.angular_frequency / samples_per_second 
        self.phase = self.phase_offset
        self.phase_history = [] 
        self.activity = math.cos(self.phase)
        self.activity_history = []

    def step(self):
        self.phase = (self.phase + self.phase_step) % (2 * math.pi)
        self.activity = math.cos(self.phase)
        self.phase_history.append(self.phase)
        self.activity_history.append(self.activity)
    
    def reset(self):
        self.phase = self.phase_offset
        self.activity = math.cos(self.phase)



class Dendrite(Soma):
    """
    Implements the oscillatory dendritic inputs whose frequencies vary as 
        a function of running speed and direction

    ARGUMENTS: 
    - samples_per_second: float; the number of samples to be "recorded" from the 
        simulated somatic oscillator or the grid cell of which it is a part
    - theta_frequency: float (Hertz); the number of cycles per second 
        characteristic of the sub-threshold "theta" oscillation. According to 
        the paper, MEC Layer II cells show such oscillations at 8-9 Hz. 
    - preferred_heading: float (radians); the running direction we wish to 
        provoke the biggest increase in the dendritic input's oscillatory 
        frequency
    - phase_offset: float (radians); the point in the oscillator's cycle from 
        which we wish the oscillation to begin at initialization
    - scaling_parameter: float (radians per centimeter) the number of radians 
        through the cycle of the interference pattern (made by the summation 
        of the dendritic oscillator's activity with that of the somatic 
        oscillator) traveled per each centimeter of space that the simulated 
        animal traverses

    ATTRIBUTES: 
    - .phase_offset: float (radians); the point in the oscillator's cycle from 
        which we wish the oscillation to begin at initialization
    - .angular_frequency: float (radians); the number of radians traveled 
        through the oscillator's cycle per second
    - .phase_step: float; the proportion of the oscillator's full cycle 
        accomplished per sampling interval
    - .phase: float (radians, [0, 2 * pi)); the state of the oscillator as 
        position in a cycle of 2-pi radians
    - .phase_history: list; all previous phase values 
    - .activity: float ([0, 1]): the value of the membrane potential as due to 
        the oscillation
    - .activity_history: list; all previous activity values
    - .theta_angular_frequency: float (radians per second) the number of radians 
        traveled through the corresponding somatic oscillator's cycle per second
    - .preferred_heading: float (radians); the running direction we wish to 
        provoke the biggest increase in the dendritic input's oscillatory 
        frequency
    - .scaling_parameter: float (radians per centimeter) the number of radians 
        through the cycle of the interference pattern (made by the summation 
        of the dendritic oscillator's activity with that of the somatic 
        oscillator) traveled per each centimeter of space that the simulated 
        animal traverses
    - .samples_per_second: float; the number of samples to be "recorded" from 
        the simulated somatic oscillator or the grid cell of which it is a part

    METHODS: 
    - .step(speed, heading): advance the state of the oscillator by one time 
        step of length determined by the samples_per_second parameter. 
        ARGUMENTS:
        - speed: float (centimeters per second); the simulated animal's running 
            speed at time of sampling
        - heading: float (radians); the simulated animal's heading at time 
            of sampling
    - .reset(): return the state of the oscillator to its initial value
    """
    def __init__(self,
                 samples_per_second, 
                 theta_frequency, 
                 preferred_heading,
                 phase_offset,
                 scaling_parameter):
        Soma.__init__(self, 
                      samples_per_second, 
                      theta_frequency,  
                      phase_offset)
        self.theta_angular_frequency = self.angular_frequency
        self.preferred_heading = preferred_heading
        self.scaling_parameter = scaling_parameter
        self.samples_per_second = samples_per_second

    def step(self,
             speed,
             heading):
        self.angular_frequency = self.theta_angular_frequency +\
            (speed * self.scaling_parameter *\
                 math.cos(heading - self.preferred_heading))
        self.phase_step = self.angular_frequency / self.samples_per_second 
        self.phase = (self.phase + self.phase_step) % (2 * math.pi)
        self.activity = math.cos(self.phase)
        self.phase_history.append(self.phase)
        self.activity_history.append(self.activity)
    
    def reset(self):
        self.phase = self.phase_offset
        self.activity = math.cos(self.phase)



class GridCell:
    """
    Implements the oscillatory interference model of grid cell firing.

    ARGUMENTS: 
    - samples_per_second: float; the number of samples to be "recorded" from the 
        simulated somatic oscillator or the grid cell of which it is a part
    - theta_frequency: float (Hertz); the number of cycles per second 
        characteristic of the sub-threshold "theta" oscillation. According to 
        the paper, MEC Layer II cells show such oscillations at 8-9 Hz. 
    - somatic_phase_offset: float (radians); the point in the somatic 
        oscillator's cycle from which we wish its oscillation to begin at 
        initialization
    - cm_per_cycle: float (centimeters); the number of centimeters desired 
        between firing fields
    - n_dendritic: int bewteen 1 and 6, inclusive; the number of dendritic 
        oscillators desired
    - offset_proportion: float between 0 and 1; how far through one cycle of 
        the grid pattern we want the cell's firing field shifted, with 
        respect to baseline
    - orientation: float (radians) between 0 and Pi / 3; how much we want the 
        grid pattern rotated, with respect to baseline

    ATTRIBUTES:
    - .scaling_parameter: float (radians per centimeter) the number of radians 
        through the cycle of the interference pattern (made by the summation 
        of the dendritic oscillator's activity with that of the somatic 
        oscillator) traveled per each centimeter of space that the simulated 
        animal traverses
    - .firing_history: list; a record of where the simulated animal has been 
        and what the simulated cell's firing rate was there. The entries are 
        lists of the form [sample number, [position x-coordinate, 
        position y-coordinate], firing rate]
    - .soma: instance of class Soma; the simulated grid cell's somatic 
        oscillator
    - .dendrites: list containing instances of class Dendrite; the simulated 
        grid cell's dendritic oscillators 

    METHODS:
    - .record(path): simulate the activity of the grid cell as the simulated 
        animal runs a specific, unbroken, course
        ARGUMENTS:
        - path: list of lists of the form [[position x-coordinate, position 
            y-coordinate], heading, speed] representing the simulated animal's 
            trajectory through the environment; the path must be continuous
    - .clear(): return all oscillators to their initial states and empty 
        all histories
    - .test(path): simulate the activity of the grid cell as the simulated 
        animal runs a specific, potentially broken, course
        ARGUMENTS:
        - path: list of list of lists of the form [[position x-coordinate, 
            position y-coordinate], heading, speed] representing the simulated 
            animal's trajectory through the environment; the path need not 
            be continuous, as each list of lists will be treated as its own 
            path separate from the others 

    """
    def __init__(self, 
                 samples_per_second,
                 theta_frequency, 
                 somatic_phase_offset,
                 cm_per_cycle, 
                 n_dendritic, # Natural number in [1, 6]
                 offset_proportion, # Real number in [0, 1)
                 orientation # Real number in [0, 1) correponding to [0, math.pi / 3)
                 ):
        self.scaling_parameter = 2 * math.pi * (1 / cm_per_cycle)
        self.firing_history = []
        self.soma = Soma(samples_per_second,
                         theta_frequency, 
                         somatic_phase_offset)
        self.dendrites = []
        dendritic_phase_offsets = [1, 1, 0, -1, -1, 0]
        dendritic_phase_offsets = dendritic_phase_offsets[0:n_dendritic]
        dendritic_phase_offsets = np.multiply(dendritic_phase_offsets, 
                                              offset_proportion * 2 * math.pi)
        preferred_heading = 0
        for n in range(n_dendritic): 
            # Default offset between preferred directions corresponds to even spacing
            dendrite = Dendrite(samples_per_second, 
                                theta_frequency, 
                                preferred_heading + orientation * math.pi / 3,
                                dendritic_phase_offsets[n],
                                self.scaling_parameter)
            self.dendrites.append(dendrite)
            preferred_heading += math.pi / 3
 
    def record(self, path):
        for sample in range(len(path)): 
            position, heading, speed = path[sample]
            self.soma.step()
            somatic_activity = self.soma.activity
            dendritic_activity = []
            for n in range(len(self.dendrites)): 
                self.dendrites[n].step(speed, heading)
                activity = self.dendrites[n].activity
                dendritic_activity.append(activity)
            # membrane_potential = np.sum(np.multiply(somatic_activity, dendritic_activity))
            membrane_potential = np.prod(np.add(somatic_activity, dendritic_activity))
            firing_rate = membrane_potential
            # Heaviside function
            if firing_rate < 0:
                firing_rate = 0
            self.firing_history.append([sample, position, firing_rate])

    def clear(self):
        self.firing_history = []
        self.soma.phase_history = []
        self.soma.activity_history = []
        for dendrite in range(len(self.dendrites)):
            self.dendrites[dendrite].phase_history = []
            self.dendrites[dendrite].activity_history = []

    def test(self, path):
        # For performing tests requiring many separate runs, like the "Q1" test
        for segment in path: 
            self.record(segment)
            self.soma.reset()
            for dendrite in range(len(self.dendrites)):
                self.dendrites[dendrite].reset()