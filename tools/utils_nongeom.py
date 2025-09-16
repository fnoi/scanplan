import os.path
import pathlib
import json


def gather_config(config_list: list) -> dict:
    """
    load n configs and join to single dictionary

    Parameters
    ----------
    config list

    Returns
    -------
    config dict
    """
    base_path = pathlib.Path().resolve()
    base_flag = False
    #TODO: get folder name (oder Stefan benennt sein parent directory zu scanplan um^^)
    if str(base_path).endswith('scanplan_new') or str(base_path).endswith('scanplan'):
        base_flag = True
    else:
        while not base_flag:
            base_path = os.path.dirname(base_path)
            #TODO: get folder name
            if str(base_path).endswith('scanplan_new') or str(base_path).endswith('scanplan'):
                base_flag = True

    config = {"project path": base_path}
    for config_file in config_list:
        with open(f'{config["project path"]}/config/{config_file}', 'r') as f:
            g = json.load(f)
        config = {**config, **g}

    for key in ["model file", "plane file"]:
        config[key] = f"{config['project path']}/data/{config[key]}"

    return config


if __name__ == "__main__":
    gather_config(config_list=["candidates_create.json"])


