import json
from types import SimpleNamespace

import numpy as np


def read_scenario(filepath):
    with open(filepath) as f:
        scenario = json.load(f, object_hook=lambda d: SimpleNamespace(**d))
    scenario.pedestrians = np.asarray(scenario.pedestrians)
    scenario.obstacles = np.asarray(scenario.obstacles)
    scenario.target = np.asarray(scenario.target)
    return scenario


class Simulation:
    def __init__(self, scenario):
        self.scenario = scenario
        self.utility = self.generate_utility(scenario)
        self.states = self.generate_states(scenario, self.utility)

    @staticmethod
    def generate_utility(scenario):
        pass

    @staticmethod
    def generate_states(scenario, utility):
        pass

    def get_states(self):
        return self.states


def main():
    scenario = read_scenario('scenario.json')
    sim = Simulation(scenario)
    print(sim)


if __name__ == '__main__':
    main()
