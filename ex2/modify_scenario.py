import argparse
import json
from pathlib import Path


def create_additional_pedestrian():
    """Creates a dictionary representing an additional pedestrian to be added to the scenario.

    Returns:
        A dictionary with keys representing the attributes of the pedestrian.
    """
    return {
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


def main(args):
    """Modifies a Vadere scenario by adding an additional pedestrian.

    Args:
        args: An argparse.Namespace object containing parsed command-line arguments.

    Returns:
        None.
    """
    scenario_path = Path(args.scenario)
    save_path = Path(args.output) if args.output else scenario_path.with_name('modified.scenario')

    with scenario_path.open() as f:
        scenario = json.load(f)

    scenario["name"] = f"{scenario['name']}_modified"
    additional_pedestrian = create_additional_pedestrian()
    scenario['scenario']['topography']['sources'].append(additional_pedestrian)

    with save_path.open('w') as f:
        json.dump(scenario, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', help='path to the scenario JSON file',
                        default='/home/a_yaroshevich/projects/tum/vadere/Scenarios/Demos/bus_station/scenarios/bus_station.scenario')
    parser.add_argument('--output', help='path to the output file (optional)')
    args = parser.parse_args()
    main(args)
