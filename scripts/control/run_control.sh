tool_type="gripper_sym_rod_robot_v4_surf_nocorr_full"
debug=0
tool_model_name="default"
active_tool_list="default"
target_shape_name="alphabet/R"
optim_algo="RS"  # RS, CEM, or GD, only used if planner_type is not learned
CEM_sample_size=20 # only used for CEM
control_loss_type="chamfer"
subtarget=0 # 1 for explict subtarget, 0 for no explict subtarget
close_loop=0 # 1 for close loop, 0 for open loop
cls_type='pcd' # pcd or img
planner_type='learned' # gnn for GNN-based planner, learned for self-supervised planner, RL for RL-based planner, sim for MPM-based planner

bash ./scripts/control/control.sh $tool_type $debug $tool_model_name $active_tool_list $target_shape_name $optim_algo $CEM_sample_size $control_loss_type $subtarget $close_loop $cls_type $planner_type
