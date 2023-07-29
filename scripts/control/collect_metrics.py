#!/usr/bin/env python

import copy
import csv
import glob
import os
import shutil
import sys

from string import ascii_uppercase
from yaml import dump


def main():
    if len(sys.argv) < 3:
        print("Please enter the name of the tool and the current date!")
        return

    tool_type = sys.argv[1]
    date_now = sys.argv[2]
    # date = "Jul-22"
    keyword = "CEM_40_0.5"
    test_type = "alphabet"

    cd = os.path.dirname(os.path.realpath(sys.argv[0]))
    root = os.path.join(cd, "..", "..")
    inspect_dir = f"{root}/dump/control/inspect/{date_now}"
    os.system(f"mkdir -p {inspect_dir}/anim")
    os.system(f"mkdir -p {inspect_dir}/soln")

    dump_dir_list = []
    control_dir = f"{root}/dump/control/control_{tool_type}"
    letter_names = list(ascii_uppercase)
    for letter in letter_names:
        dump_dir_list_glob = sorted(
            glob.glob(os.path.join(control_dir, test_type, letter, "*"))
        )
        for dump_dir in dump_dir_list_glob:
            # import pdb; pdb.set_trace()
            if (
                os.path.exists(f"{dump_dir}/anim_args/MPC_anim_args.pkl")
                and keyword in dump_dir
                and not "debug" in dump_dir
            ):  # date in dump_dir and
                dump_dir_list.append(dump_dir)

    # import pdb; pdb.set_trace()
    stats = []
    for dump_dir in dump_dir_list:
        test_name = dump_dir.split("/")[-1]
        target_name = "_".join(dump_dir.split("/")[-3:-1])
        print(f"Target: {target_name} -> {test_name}")

        res_anim = f"{dump_dir}/anim/MPC_anim.mp4"
        if not os.path.exists(res_anim):
            os.system(
                f"python utils/visualize.py {dump_dir}/anim_args/MPC_anim_args.pkl"
            )

        shutil.copyfile(res_anim, f"{inspect_dir}/anim/{target_name}_{test_name}.mp4")
        shutil.copyfile(
            f"{dump_dir}/param_seqs/MPC_param_seq.yml",
            f"{inspect_dir}/soln/{target_name}_{test_name}.yml",
        )

        with open(f"{dump_dir}/control.txt", "r") as log_file:
            row = {"name": test_name, "target": target_name}
            log = log_file.readlines()
            last_line = log[-1]
            if "lprof" in last_line:
                last_line = log[-2]

            row["control_loss"] = last_line.split(";")[0].split(":")[-1].strip()
            row["chamfer"] = last_line.split(";")[1].split(":")[-1].strip()
            row["emd"] = last_line.split(";")[2].split(":")[-1].strip()

            stats.append(row)

    fields = ["name", "target", "control_loss", "chamfer", "emd"]
    with open(f"{inspect_dir}/metrics.csv", "w") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        for row in stats:
            writer.writerow(row)


if __name__ == "__main__":
    main()
