from waymo_open_dataset.protos import scenario_pb2
import tensorflow as tf

def read_data(file_path: str, scenario_id_set: set[str] = None) -> list:
    """Parse the Waymo data, and return the whole scenario object
    If scenario id set is not given, return all scenarios within the file given by file_num"""

    all_scenarios: list = []
    dataset = tf.data.TFRecordDataset(file_path, compression_type="")
    for i, data in enumerate(dataset):
        scenario = scenario_pb2.Scenario()
        scenario.ParseFromString(data.numpy())
        if scenario_id_set == None or scenario.scenario_id in scenario_id_set:
            all_scenarios.append(scenario)

    return all_scenarios