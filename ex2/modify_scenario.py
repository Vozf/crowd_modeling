import json
from pathlib import Path


def main():
    scenario_path = Path(
        '/home/a_yaroshevich/projects/tum/vadere/Scenarios/Demos/bus_station/scenarios/bus_station.scenario')
    save_path = scenario_path.with_name('modified.scenario')
    with open(scenario_path) as f:
        scenario = json.load(f)

    scenario["name"] = scenario["name"] + "_modified"
    additional_pedestrian = {
        "id": 7,
        "shape": {
            "x": 11.0,
            "y": 17.0,
            "width": 1.0,
            "height": 1.0,
            "type": "RECTANGLE"
        },
        "visible": True,
        "targetIds": [2, 1],
        "spawner": {
            "type": "org.vadere.state.attributes.spawner.AttributesRegularSpawner",
            "constraintsElementsMax": -1,
            "constraintsTimeStart": 0.0,
            "constraintsTimeEnd": 0.0,
            "eventPositionRandom": False,
            "eventPositionGridCA": False,
            "eventPositionFreeSpace": True,
            "eventElementCount": 1,
            "eventElement": None,
            "distribution": {
                "type": "org.vadere.state.attributes.distributions.AttributesConstantDistribution",
                "updateFrequency": 1.0
            }
        },
        "groupSizeDistribution": [1.0]
    }
    scenario['scenario']['topography']['sources'].append(additional_pedestrian)
    with open(save_path, 'w') as f:
        json.dump(scenario, f, indent=2)


if __name__ == '__main__':
    main()
