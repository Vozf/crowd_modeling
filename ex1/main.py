import itertools
import json
from copy import deepcopy
from dataclasses import dataclass
from math import floor
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

    @staticmethod
    def generate_utility(scenario):
        visited = set()
        pq = []
        nodeCosts = {}
        grid = np.asarray([(a, b) for a in range(scenario.field[0]) for b in range(scenario.field[1])])
        # .reshape((scenario.field.width, scenario.field.height, 2))
        distances = np.linalg.norm(grid[:, np.newaxis, :] - scenario.target, axis=-1)
        utility_map = distances.min(axis=-1).reshape((scenario.field[0], scenario.field[1]))
        return utility_map
        for start_node in scenario.target:
            nodeCosts[start_node] = 0
            heap.heappush(pq, (0, start_node))

        while pq:
            # go greedily by always extending the shorter cost nodes first
            _, node = heap.heappop(pq)
            visited.add(node)

            neighbours = itertools.product((node[0] - 1, node[0], node[0] + 1), (node[1] - 1, node[1], node[1] + 1))
            unvisited_neighbours = list(filter(lambda x: x in visited, neighbours))

            for neighbor in unvisited_neighbours:
                newCost = nodeCosts[node] + np.linalg.norm(node - neighbor)
                print(newCost)
                if nodeCosts[neighbor] > newCost:
                    parentsMap[neighbor] = node
                    nodeCosts[neighbor] = newCost
                    heap.heappush(pq, (newCost, neighbor))

        return parentsMap, nodeCosts

    @staticmethod
    def generate_states(scenario: State, utility):
        states = [(0, deepcopy(scenario))]
        actions = [*((0, idx) for idx in range(len(scenario.pedestrians)))]
        while actions:
            current_timestamp, idx = heap.heappop(actions)
            old_timestamp, current_state = deepcopy(states[-1])
            ped = current_state.pedestrians[idx]

            neighbours = itertools.product((ped[0] - 1, ped[0], ped[0] + 1), (ped[1] - 1, ped[1], ped[1] + 1))
            neighbours = filter(
                lambda coord: 0 <= coord[0] < scenario.field[0] and 0 <= coord[1] < scenario.field[1], neighbours)
            neighbours = np.array(list(neighbours))
            best_neighbour = neighbours[utility[neighbours[:, 0], neighbours[:, 1]].argmin()]
            if (best_neighbour == ped).all():
                continue

            step_time = 1.4 if np.abs(best_neighbour - ped).sum() == 2 else 1.0
            current_state.pedestrians[idx] = best_neighbour
            states.append((current_timestamp, current_state))
            heap.heappush(actions, (round(current_timestamp + step_time, 2), idx))

        filtered_states = []
        for i in range(len(states) - 1):
            if floor(states[i][0]) != floor(states[i + 1][0]):
                filtered_states.append(states[i])

        return filtered_states

    def get_states(self):
        return self.states


def main():
    scenario = read_scenario('scenario_task_3.json')
    sim = Simulation(scenario)
    print(sim.get_states())


if __name__ == '__main__':
    main()
