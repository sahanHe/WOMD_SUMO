import os
import argparse
import re
import yaml

from waymo_open_dataset.protos import sim_agents_submission_pb2
from .utils import scenario_counts


def pack_submision(
    scenario_rollouts, metadata: dict
) -> sim_agents_submission_pb2.SimAgentsChallengeSubmission:

    return sim_agents_submission_pb2.SimAgentsChallengeSubmission(
        scenario_rollouts=scenario_rollouts,
        submission_type=sim_agents_submission_pb2.SimAgentsChallengeSubmission.SIM_AGENTS_SUBMISSION,
        account_name=metadata["account_name"],
        unique_method_name=metadata["unique_method_name"],
        authors=metadata["authors"],
        affiliation=metadata["affiliation"],
        description=metadata["description"],
        method_link=metadata["method_link"],
        # New REQUIRED fields.
        uses_lidar_data=False,
        uses_camera_data=False,
        uses_public_model_pretraining=False,
        num_model_parameters=metadata["num_model_parameters"],
        acknowledge_complies_with_closed_loop_requirement=True,
    )


def gen_shard_submission_file(shard_idx: int, config: dict):
    print(f" file-number: {shard_idx} ".center(50, "#"))

    scenario_rollouts: list[sim_agents_submission_pb2.ScenarioRollouts] = []
    # find all the rollouts for this shard
    for filename in os.listdir(config["rollouts_dir"]):
        i, scenario_id = filename.split("-")
        i = int(i)
        if i == shard_idx:
            with open(os.path.join(config["rollouts_dir"], filename), "rb") as f:
                scenario_rollout = sim_agents_submission_pb2.ScenarioRollouts()
                scenario_rollout.ParseFromString(f.read())
                scenario_rollouts.append(scenario_rollout)

    # ensure the number of rollouts is equal to the number of scenarios in the original file
    scenario_count = scenario_counts[config["dataset_type"]]
    print(
        f" total scenarios: {len(scenario_rollouts)} | should be {scenario_count[shard_idx]} ".center(70, "#")
    )
    assert scenario_count[shard_idx] == len(scenario_rollouts)

    # a single submission file
    shard_submission = pack_submision(scenario_rollouts, config["metadata"])
    output_filename = "submission.binproto-{:05d}-of-00150".format(shard_idx)
    assert re.fullmatch(r".*\.binproto(-\d{5}-of-\d{5})?", output_filename) is not None

    with open(os.path.join(config["submission_dir"], output_filename), "wb") as f:
        f.write(shard_submission.SerializeToString())

    # logging
    with open(os.path.join(config["log_dir"], str(shard_idx)), "w") as f:
        f.write("success.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--shard_idx", type=int)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

        # preparation: make sure all directories exist
        assert os.path.exists(config["rollouts_dir"])
        os.makedirs(config["submission_dir"], exist_ok=True)
        os.makedirs(config["log_dir"], exist_ok=True)

        gen_shard_submission_file(args.shard_idx, config)
