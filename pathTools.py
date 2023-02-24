import math
import random
import numpy as np
import csv

def generateNextStep(prev_position, prev_speed, prev_heading, max_speed, min_speed):
        position = (prev_position[0] + prev_speed * math.cos(prev_heading), 
                    prev_position[1] + prev_speed * math.sin(prev_heading))
        heading = (prev_heading + random.gauss(0, (math.pi / 3))) % (2 * math.pi)
        speed = prev_speed + random.gauss(0, (max_speed - min_speed) / 8) 
        return (position, heading, speed)

def generatePath(duration, arena_size):
    min_speed = arena_size * 0.05
    max_speed = arena_size * 0.2
    path = [[(0, 0), 0, 0]]
    for iteration in range(duration):
        t = iteration + 1
        prev_position, prev_heading, prev_speed = path[iteration] 
        position, heading, speed = generateNextStep(prev_position, 
                                                    prev_speed, 
                                                    prev_heading, 
                                                    max_speed, 
                                                    min_speed)
        while (abs(position[0]) > arena_size or abs(position[1]) > arena_size):
            position, heading, speed = generateNextStep(position, 
                                                        speed, 
                                                        heading, 
                                                        max_speed, 
                                                        min_speed)
        path.append([position, heading, speed])
    path = path[1:]
    return path

def generateConstantPath(start_position, 
                         duration, 
                         heading,
                         speed, # in cm/sec
                         samples_per_second):
    position_change_per_sample = speed / samples_per_second
    x_change_per_sample = position_change_per_sample * math.cos(heading)
    y_change_per_sample = position_change_per_sample * math.sin(heading)
    path = [[(start_position), 0, 0]]
    for sample_number in range(math.floor(duration)):
        x = start_position[0] + x_change_per_sample * sample_number
        y = start_position[1] + y_change_per_sample * sample_number
        position = (x, y)
        path.append([position, heading, speed])
    path = path[1:]
    return path

# This test needs to be fed to the model in a "batched" way
# Or have runs in a circular arena
def generateQ1Test(speed, # in cm/sec
                   samples_per_second,
                   arena_size):
    samples_per_wall = (arena_size / speed * samples_per_second)
    spacing_factor = arena_size / samples_per_wall
    path = []
    destination = [0, arena_size]
    while destination[0] < arena_size:
        distance = math.dist((0,0), destination)
        duration = (distance / speed) * samples_per_second
        heading = math.pi / 2
        if destination[0] != 0:
            heading = math.atan(destination[1] / destination[0]) 
            # (No need to adjust because we live in Q1)
        segment = generateConstantPath((0, 0),
                                       duration, 
                                       heading,
                                       speed, 
                                       samples_per_second)
        path.append(segment)
        destination[0] += spacing_factor
    while destination[1] > 0: 
        distance = math.dist((0,0), destination)
        duration = (distance / speed) * samples_per_second
        heading = math.atan(destination[1] / destination[0]) 
        segment = generateConstantPath((0, 0), # Account for heading
                                       duration, 
                                       heading,
                                       speed, 
                                       samples_per_second)
        path.append(segment)
        destination[1] += (-1 *  spacing_factor)
    return path
    

def getPath(file_name, sampling_rate): 
    sample_interval = 1 / sampling_rate
    path = [] # (Coordinates), heading, speed
    with open(file_name, newline = '') as file: 
        trajectory = csv.reader(file, quoting=csv.QUOTE_NONNUMERIC)
        for row in trajectory: 
            position = (row)
            if len(path) == 0: 
                path.append([position, 0, 0])
            else: 
                prev_position = path[len(path) - 1][0]
                displacement_vector = tuple(map(lambda after, before: after - before, 
                                        position, 
                                        prev_position))
                if displacement_vector[0] == 0:
                    if displacement_vector[1] > 0: 
                        heading = math.pi / 2
                    else: 
                        heading = path[len(path) - 1][1]
                else: 
                    heading = math.atan(displacement_vector[1] / displacement_vector[0])
                    if displacement_vector[0] < 0: 
                        heading += math.pi
                    elif displacement_vector[1] < 0: 
                        heading = 2 * math.pi + heading
                speed = math.hypot(displacement_vector[0], 
                                   displacement_vector[1]) / sample_interval
                path.append([position, heading, speed])
    path = path[1:] # Remove the first observation because we don't know the heading
    return path

def test_getPath(file_name, path):
    with open(file_name, 'w') as file: 
        writer = csv.writer(file)
        for datum in path: 
            writer.writerow(datum) 