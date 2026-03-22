import tensorflow as tf
from waymo_open_dataset.protos import scenario_pb2

# --- Configuration ---
INPUT_FILE = '/home/sahan/Desktop/PhD/waymo_motion_training/uncompressed_scenario_validation_validation.tfrecord-00000-of-00150'
OUTPUT_FILE = 'unsignalized_scenarios.tfrecord-00000-of-00150'

def is_unsignalized(scenario):
    """
    Returns True if the scenario has NO active traffic light states.
    We define 'unsignalized' as a scenario where no lane has a dynamic 
    traffic signal state associated with it throughout the timeline.
    """
    # Iterate through all time steps in the scenario
    for step in scenario.dynamic_map_states:
        # lane_states contains traffic light data (color, state, etc.)
        if len(step.lane_states) > 0:
            return False  # Found a signal, so it's signalized
    return True


def filter_and_save():
    # Initialize the reader
    raw_dataset = tf.data.TFRecordDataset(INPUT_FILE)
    
    count_total = 0
    count_saved = 0

    print(f"Starting filtration on {INPUT_FILE}...")

    f = open("unsignalized_scenarios_log.txt", "w")

    with tf.io.TFRecordWriter(OUTPUT_FILE) as writer:
        for raw_record in raw_dataset:
            count_total += 1
            
            # Parse the binary record into a Scenario proto
            scenario = scenario_pb2.Scenario()
            scenario.ParseFromString(raw_record.numpy())
            
            # Apply our filter
            if is_unsignalized(scenario):

                # Write the original binary data (raw_record) to keep it fast
                writer.write(raw_record.numpy())
                count_saved += 1
                f.write(f"{scenario.scenario_id}\n")
            
            if count_total % 100 == 0:
                print(f"Processed {count_total} scenarios... Saved: {count_saved}")

    print(f"\nDone!")
    print(f"Total scenarios processed: {count_total}")
    print(f"Unsignalized scenarios saved: {count_saved}")

if __name__ == "__main__":
    filter_and_save()