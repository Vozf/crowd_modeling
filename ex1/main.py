import itertools
import json
import math
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from math import floor, ceil
from types import SimpleNamespace
import heapq as heap

import numpy as np


@dataclass
class State:
    field: np.ndarray
    pedestrians: np.ndarray
    obstacles: np.ndarray
    target: np.ndarray


def read_scenario(filepath):
    with open(filepath) as f:
        scenario = json.load(f, object_hook=lambda d: SimpleNamespace(**d))
    # scenario.pedestrians = tuple(tuple(coordinate) for coordinate in scenario.pedestrians)
    # scenario.obstacles = tuple(tuple(coordinate) for coordinate in scenario.obstacles)
    # scenario.target = tuple(tuple(coordinate) for coordinate in scenario.target)
    scenario.field = np.asarray(scenario.field)
    scenario.pedestrians = np.asarray(scenario.pedestrians)
    scenario.obstacles = np.asarray(scenario.obstacles) if scenario.obstacles else np.array([]).reshape(0, 2)
    scenario.target = np.asarray(scenario.target)
    return State(scenario.field, scenario.pedestrians, scenario.obstacles, scenario.target)


class Simulation:
    def __init__(self, scenario):
        self.scenario = scenario
        self.utility = self.generate_utility(scenario)
        self.states = self.generate_states(scenario, self.utility)

    @classmethod
    def generate_utility(cls, scenario: State, dijkstra: bool = True):
        if dijkstra:
            visited = set()
            pq = []
            node_costs = defaultdict(lambda: math.inf)
            for endnonde in scenario.target:
                node_costs[tuple(endnonde)] = 0
                heap.heappush(pq, (0, endnonde))

            while pq:
                # go greedily by always extending the shorter cost nodes first
                _, node = heap.heappop(pq)
                visited.add(tuple(node))

                neighbours = cls.get_neighbours(node, scenario)
                unvisited_neighbours = list(filter(lambda x: x not in visited, neighbours))

                for neighbor in unvisited_neighbours:
                    newCost = node_costs[tuple(node)] + cls.get_neighbour_distance(node, neighbor)
                    print(newCost)
                    if node_costs[neighbor] > newCost:
                        # parentsMap[neighbor] = node
                        node_costs[neighbor] = newCost
                        heap.heappush(pq, (newCost, neighbor))

            utility_map = np.zeros((scenario.field[0], scenario.field[1]))
            keys = np.asarray(list(node_costs.keys()))
            utility_map[keys[:, 0], keys[:, 1]] = list(node_costs.values())
            return utility_map

        else:
            grid = np.asarray([(a, b) for a in range(scenario.field[0]) for b in range(scenario.field[1])])
            # .reshape((scenario.field.width, scenario.field.height, 2))
            distances = np.linalg.norm(grid[:, np.newaxis, :] - scenario.target, axis=-1)
            utility_map = distances.min(axis=-1).reshape((scenario.field[0], scenario.field[1]))
            return utility_map

    @classmethod
    def generate_states(cls, scenario: State, utility):
        distances = np.linalg.norm(scenario.pedestrians[:, np.newaxis, :] - scenario.target, axis=-1)
        print(f"source distances {distances}")

        states = [(0, deepcopy(scenario))]
        actions = [*((0, idx) for idx in range(len(scenario.pedestrians)))]
        while actions:
            current_timestamp, idx = heap.heappop(actions)
            old_timestamp, current_state = deepcopy(states[-1])
            ped = current_state.pedestrians[idx]

            neighbours = cls.get_neighbours(ped, current_state)
            neighbours = np.array(list(neighbours))
            best_neighbour = neighbours[utility[neighbours[:, 0], neighbours[:, 1]].argmin()]
            if (best_neighbour == ped).all():
                print(f"{idx} reached target in {current_timestamp} time")
                continue

            step_time = cls.get_neighbour_distance(best_neighbour, ped)
            current_state.pedestrians[idx] = best_neighbour
            states.append((current_timestamp, current_state))
            heap.heappush(actions, (round(current_timestamp + step_time, 2), idx))

        # to improve performance keep only last state in every second timestep
        filtered_states = []
        for i in range(len(states) - 1):
            if ceil(states[i][0]) != ceil(states[i + 1][0]):
                filtered_states.append((ceil(states[i][0]), states[i][1]))
        filtered_states.append((ceil(states[-1][0]), states[-1][1]))

        return filtered_states

    @classmethod
    def get_neighbour_distance(cls, cell_1, cell_2):
        return 1.4 if np.abs(np.subtract(cell_1, cell_2)).sum() == 2 else 1.0

    @staticmethod
    def get_neighbours(ped, current_state):
        neighbours = itertools.product((ped[0] - 1, ped[0], ped[0] + 1), (ped[1] - 1, ped[1], ped[1] + 1))
        neighbours = filter(
            lambda coord: 0 <= coord[0] < current_state.field[0] and 0 <= coord[1] < current_state.field[1], neighbours)
        return neighbours


    def get_states(self):
        return self.states


def main():
    scenario = read_scenario('scenario_task_3.json')
    sim = Simulation(scenario)
    # print(sim.get_states())


if __name__ == '__main__':
    main()
