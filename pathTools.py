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

def generateConstantPath(duration):
    path = [[(0, 0), 0, 0]]
    for iteration in range(duration):
        position = (100 * iteration, 0)
        heading = 0
        speed = 100
        path.append([position, heading, speed])
    path = path[1:]
    return path

def ratPath(file_name, sampling_rate): 
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

def test_ratPath(file_name, path):
    with open(file_name, 'w') as file: 
        writer = csv.writer(file)
        for datum in path: 
            writer.writerow(datum) 