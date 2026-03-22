import os
import argparse
import yaml

from .utils import map_glblidx_to_localidx, get_default_filename
from .rollouts import gen_scenario_rollouts
from src.utils.read_data import read_data

def gen_rollouts(global_idx: int, config: dict):
    dataset_type = config["wosac"]["dataset_type"]
    taskname = config["wosac"]["taskname"]
    
    file_number, local_idx = map_glblidx_to_localidx(dataset_type, global_idx)

    # read data
    tfrecord_path = os.path.join(config["wosac"]["dataset_dir"], get_default_filename(dataset_type, file_number))
    scenario = read_data(tfrecord_path)[local_idx]

    scenario_id = scenario.scenario_id
    print(f" global-index {global_idx} | file-number {file_number} | scenario-index {local_idx} | scenario-id {scenario_id} ".center(100, '#'))

    baseline_config = {
        "sumo_cfg_path": os.path.join( config["wosac"]["sumo_dir"], dataset_type, f"{scenario_id}/{scenario_id}.sumocfg"),
        "sumo_net_path": os.path.join( config["wosac"]["sumo_dir"], dataset_type, f"{scenario_id}/{scenario_id}.net.xml")
    }

    # run simulations
    try:
        rollouts = gen_scenario_rollouts(scenario, baseline_cfg=baseline_config, sim_cfg=config["simulation"])
    except Exception as e:
        print(f"ERROR in running scenario {global_idx}-{scenario_id}:")
        print(e)
        return

    # rollouts output
    output_path = os.path.join(config["wosac"]["output_dir"], taskname, f"{file_number}-{scenario_id}")
    with open(output_path, "wb") as f:
        f.write(rollouts.SerializeToString())
    
    # logging
    log_path = os.path.join(config["wosac"]["log_dir"], taskname, f"{global_idx}-{file_number}-{scenario_id}")
    with open(log_path, "w") as f:
        f.write(f"success.")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--idx', type=int)

    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

        # preparation: make sure all directories exist
        assert os.path.exists(config["wosac"]["dataset_dir"])
        assert os.path.exists(config["wosac"]["sumo_dir"])
        os.makedirs(os.path.join(config["wosac"]["output_dir"], config["wosac"]["taskname"]), exist_ok=True)
        os.makedirs(os.path.join(config["wosac"]["log_dir"], config["wosac"]["taskname"]), exist_ok=True)

        # generate rollouts
        gen_rollouts(config["wosac"]["idx_mapping"][args.idx], config)