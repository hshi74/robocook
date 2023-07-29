#!/usr/bin/env python

import copy
import csv
import glob
import os
import shutil
import sys

from yaml import dump


def main():
    cd = os.path.dirname(os.path.realpath(sys.argv[0]))
    date = "Oct-30"
    tool_arr = ["gripper_sym_rod", "gripper_asym", "gripper_sym_plane"]
    root = os.path.join(cd, "..", "..")
    inspect_dir = f"{root}/data/planning_data/inspect/{date}"
    os.system(f"mkdir -p {inspect_dir}/anim")
    dump_dir_list = []
    for tool_type in tool_arr:
        dy_dir = f"{root}/data/planning_data/data_{tool_type}_robot_v4_surf_nocorr_full_normal_keyframe=16_action=2_p=300/dump/"
        dump_dir_list_glob = sorted(glob.glob(os.path.join(dy_dir, "*")))
        for dump_dir in dump_dir_list_glob:
            if (
                os.path.exists(f"{dump_dir}/eval")
                and date in dump_dir
                and not "debug" in dump_dir
            ):
                dump_dir_list.append(dump_dir)

    stats = []
    for dump_dir in dump_dir_list:
        for tool_type_candi in tool_arr:
            if tool_type_candi in dump_dir:
                tool_type = tool_type_candi
                break
        test_name = dump_dir.split("/")[-1]
        print(test_name)

        dataset_list = ["train", "valid", "test"]
        n_rollout = 3
        for dataset in dataset_list:
            for i in range(n_rollout):
                i = str(i).zfill(3)
                res_anim = f"{dump_dir}/eval/anim/{dataset}_{i}_{str(0).zfill(3)}.mp4"
                if os.path.exists(res_anim):
                    shutil.copyfile(
                        res_anim,
                        f"{inspect_dir}/anim/{test_name}_{dataset}_{i}_{str(0).zfill(3)}.mp4",
                    )

        shutil.copyfile(
            f"{dump_dir}/eval/eval.txt", f"{inspect_dir}/eval_{test_name}.txt"
        )

        with open(f"{inspect_dir}/eval_{test_name}.txt", "r") as log_file:
            row = {"tool": tool_type, "name": test_name}
            log = log_file.readlines()
            i = 0
            loss_all = 0.0
            while i < len(log):
                line = log[i]

                if "train phase" in line:
                    row["dataset"] = "train"
                if "valid phase" in line:
                    row["dataset"] = "valid"
                if "test phase" in line:
                    row["dataset"] = "test"

                if len(row) > 0 and "loss" in line.lower():
                    loss = line.strip().split(":")[1].strip()
                    row["loss"] = loss
                    loss_all += float(loss)
                    stats.append(row)
                    if "test" in row["dataset"]:
                        row_copy = copy.deepcopy(row)
                        row_copy["dataset"] = "all"
                        row_copy["loss"] = str(loss_all / 3)
                        loss_all = 0.0
                        stats.append(row_copy)
                    row = {"tool": tool_type, "name": test_name}

                i += 1

    fields = ["tool", "name", "dataset", "loss"]
    with open(f"{inspect_dir}/metrics.csv", "w") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        for row in stats:
            writer.writerow(row)


if __name__ == "__main__":
    main()
