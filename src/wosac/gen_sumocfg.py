import os
import xml.etree.ElementTree as ET
import subprocess
import argparse

def gen_sumocfg(root_dir: str, scenario_id: str, custom_root_dir: str = None) -> None:

    complete_dir = os.path.join(root_dir, scenario_id)
    assert os.path.isdir(complete_dir)

    sumo_net_path = os.path.join(complete_dir, f"{scenario_id}.net.xml")
    sumo_route_path = os.path.join(complete_dir, f"{scenario_id}.rou.xml")
    sumocfg_path = os.path.join(complete_dir, f"{scenario_id}.sumocfg")

    # I. write an empty .rou.xml
    root = ET.Element("routes")
    tree = ET.ElementTree(root)
    tree.write(sumo_route_path, encoding='utf-8', xml_declaration=True)

    # II. generate a .sumocfg with sumo
    bash_command = [
        "sumo",
        #input & output
        f"-n={sumo_net_path}",
        f"-r={sumo_route_path}",
        f"--save-configuration={sumocfg_path}",
        # other simulation settings
        f"--step-length=0.1",
        f"--random=true",
        f"--time-to-teleport=-1",
    ]
    process= subprocess.Popen(bash_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    print(f"[file output] Generated .rou.xml and .sumocfg for scenario {scenario_id}:")
    print(f"  {sumo_route_path}")
    print(f"  {sumocfg_path}")

    if custom_root_dir:
        modify_sumocfg(root_dir, scenario_id, old_root=root_dir, new_root=custom_root_dir)

def modify_sumocfg(root_dir: str, scenario_id: str, old_root: str, new_root: str):

    sumocfg_path = os.path.join(root_dir, scenario_id, f"{scenario_id}.sumocfg")
    if not os.path.exists(sumocfg_path):
        print(f"[warning] sumocfg path not found")
        return
    
    tree = ET.parse(sumocfg_path)
    root_element = tree.getroot()
    input_element = root_element.find("input")
    assert input_element is not None
    
    net_file_element = input_element.find("net-file")
    assert net_file_element is not None
    old_netfile_path = net_file_element.get("value")
    if old_netfile_path.startswith(old_root):
        new_netfile_path = os.path.join(new_root, old_netfile_path[len(old_root):])
        net_file_element.set('value', new_netfile_path)

    route_files_element = input_element.find("route-files")
    assert route_files_element is not None
    old_route_files_path = route_files_element.get('value')
    if old_route_files_path.startswith(old_root):
        new_route_files_path = os.path.join(new_root , old_route_files_path[len(old_root):])
        route_files_element.set('value', new_route_files_path)

    tree.write(sumocfg_path, encoding="UTF-8", xml_declaration=True)
    print(f"[file updated] updated sumocfg file {sumocfg_path}")


"""
The main function is used for parallel processing
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--operation', type=str, default="modify_sumocfg")
    parser.add_argument('--root_dir', type=str)
    parser.add_argument('--old_root', type=str, default="")
    parser.add_argument('--new_root', type=str, default="")

    args = parser.parse_args()
    all_subdirs = sorted(os.listdir(args.root_dir))
    for scenario_id in all_subdirs:
        if args.operation == "gen_sumocfg":
            gen_sumocfg(args.root_dir, scenario_id)
        elif args.operation == "modify_sumocfg":
            modify_sumocfg(args.root_dir, scenario_id, old_root=args.old_root, new_root=args.new_root)
        else:
            raise ValueError("--operation only gen_sumocfg or modify_sumocfg")
