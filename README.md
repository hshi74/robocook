# RoboCook: Long-Horizon Elasto-Plastic Object Manipulation with Diverse Tools


https://github.com/hshi74/robocook/assets/40679006/ddfaa3ee-21d4-4b13-936b-93114238afe6


**RoboCook: [Website](https://hshi74.github.io/robocook/) |  [Paper](https://arxiv.org/abs/2306.14447) | [Data](https://drive.google.com/drive/folders/1kEw4rnFWnYpkelfucvtJMEYYwA5P_0CK?usp=sharing)**

If you use this code for your research, please cite:

```
@article{shi2023robocook,
  title={RoboCook: Long-Horizon Elasto-Plastic Object Manipulation with Diverse Tools}, 
  author={Shi, Haochen and Xu, Huazhe and Clarke, Samuel and Li, Yunzhu and Wu, Jiajun},
  journal={arXiv preprint arXiv:2306.14447},
  year={2023},
}
```

## Prerequisites

- Ubuntu 18.04 or 20.04
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

## Installation

- Clone this repo:
    ```bash
    git clone https://github.com/hshi74/robocook.git
    cd robocook
    git submodule update --init --recursive
    ```

- Create the conda environment:
    ```bash
    conda env create -f robocook.yml
    conda activate robocook
    ```

- Install requirements for the simulator.
    ```bash
    cd simulator
    pip install -e .
    ```

- Add this line to your ~/.bashrc
    ```bash
    export PYTHONPATH="${PYTHONPATH}:[path/to/robocook]"
    ```

## Data

We share the real-world data we collected for this project at [this link](https://drive.google.com/drive/folders/1kEw4rnFWnYpkelfucvtJMEYYwA5P_0CK?usp=sharing). Please download and extract the files into the `data` folder, e.g., 
the folder structure should look like `data/dynamics/data_gripper_sym_rod_robot_v4_surf_nocorr_full_normal_keyframe=16`.

- raw: because the raw ROSbag data are huge, we share an episode of random exploration with the two-rod symmetric gripper as a sample. The episode has five sequences, each with around ~50 frames.

- dynamics: the processed datasets are built to train the GNN-based dynamics model. Each tool has a different dataset and each dataset is split into `train`, `valid`, and `test`. There are seven subfolders, and each of them represents a different tool. We merge `press_square` and `press_circle` into one dataset for easy organizing. The same applies to `punch_saqure` and `punch_circle`. An example trajectory is in `data/dynamics/data_gripper_sym_rod_robot_v4_surf_nocorr_full_normal_keyframe=16/train/000`, where each `.h5` file represents a frame in the trajectory, and the `repr.mp4` video shows the visualization of the trajectory. Sections 3.1 and 6.1 of the [paper](https://arxiv.org/abs/2306.14447) discuss the process of building the dynamics dataset.

- tool_classification: this processed dataset is built to train the tool classifier. The dataset is split into `train`, `valid`, and `test`. Each split has 15 subfolders, each of which denotes a different tool. This is an imbalanced dataset - each tool has different examples due to how we collect the data as described in Section 6.1.1 of the [paper](https://arxiv.org/abs/2306.14447). An example is in `data/tool_classification/train/gripper_sym_rod/000`, where `in.ply` is the point cloud observation of the elasto-plastic object before the robot applies the tool and `out.ply` is that after the robot applies the tool. The first few paragraphs in Section 3.3 of the [paper](https://arxiv.org/abs/2306.14447) describe the details.

- planning: these datasets are synthesized by the learned dynamics model. Again, each subfolder represents a dataset for a different tool, and each dataset is split into `train`, `valid`, and `test`. Under `data/planning/data_gripper_sym_rod_robot_v4_surf_nocorr_full_normal_keyframe=16_action=2_p=300`, `dy_model.pth` is a checkpoint of the dynamics model and `args.npy` records some configurations of the dynamics model. The subfolders of each split (e.g., `train/000`) represent different starting configurations of the dough, and the subfolders of them (e.g., `train/000/000`) represent different sampled random actions and have the same starting configuration of the dough. Sections 3.3 and 6.3 of the [paper](https://arxiv.org/abs/2306.14447) discuss the details.


## Perception
To run the perception module in RoboCook, run `bash scripts/perception/perception.sh` and enter the start index and the range as prompted in the command line. There are a couple of tunable parameters in the shell script. Since only five sequences are in `data/raw`, this script can sample at most five trajectories. To visualize each step of the sampling process, set `visualize = True` in the `main` function of `sample.py`. The results are stored in `dump/perception/[tool_type]_[time_stamp]`.

## Dynamics
To run the perception module in RoboCook, run `bash scripts/dynamics/run_train.sh`. Many tunable parameters are in the shell script and some inline comments are provided there. The results are in `dump/dynamics/dump_[tool_type]_[time_stamp]`, and the evaluation results are stored in the `eval` subfolder.


## Tool Classification
To run the tool classification module in RoboCook, run `bash planning/pcd_tool_classifier/run_train.sh`. There are some tunable parameters in the shell script and some inline comments are provided there. The results are in `dump/tool_classification/cls_[test_name]_[time_stamp]`.

### Planning
To run the closed-loop control module in RoboCook, run `bash scripts/control/run_control.sh`. There are some tunable parameters in the shell script and some inline comments are provided there. The results are in `dump/control/control[test_name]_[time_stamp]`.
