import glob
import numpy as np
import os
import sys

def main():
    tool_type="gripper_sym_plane_robot_v4"
    perception_dir=[
        "./dump/perception/gripper_sym_plane_robot_v4_18-Sep-2022-10:32:22.876372"
    ]

    # tool_type += '_surf_nocorr'
    cd = os.path.dirname(os.path.realpath(sys.argv[0]))
    data_root_dir = os.path.join(cd, "..", "..", "data", "gt", f"data_{tool_type}")
    if type(perception_dir) is list:
        perception_data_list = []
        for p_dir in perception_dir:
            p_dir = os.path.join(cd, "..", "..", p_dir)
            perception_data_list += sorted(glob.glob(os.path.join(p_dir, '*')))
    else:
        perception_dir = os.path.join(cd, "..", "..", perception_dir)
        perception_data_list = sorted(glob.glob(os.path.join(perception_dir, '*')))

    np.random.seed(0)
    np.random.shuffle(perception_data_list)
    for path in perception_data_list:
        if 'archive' in path: 
            perception_data_list.remove(path)

    dataset_size = len(perception_data_list)
    valid_set_size = int(dataset_size * 0.1)
    test_set_size = int(dataset_size * 0.1)
    training_set_size = dataset_size - valid_set_size - test_set_size

    print(f"Training set size: {training_set_size}")
    print(f"Valid set size: {valid_set_size}")
    print(f"Test set size: {test_set_size}")
    
    dataset_dict = {"train": training_set_size, "valid": valid_set_size, "test": test_set_size}
    p_idx = 0
    for dataset, size in dataset_dict.items():
        dataset_dir = os.path.join(data_root_dir, dataset)
        if os.path.exists(dataset_dir):
            print(f'The {dataset} set already exists!')
        else:
            os.system('mkdir -p ' + dataset_dir)
            for i in range(size):
                p_name = os.path.basename(perception_data_list[p_idx])
                data_name = str(i).zfill(3)
                print(f'{p_name} -> {data_name}')
                os.system(f'cp -r {perception_data_list[p_idx]} {os.path.join(dataset_dir, data_name)}')
                p_idx += 1


if __name__ == "__main__":
    main()
