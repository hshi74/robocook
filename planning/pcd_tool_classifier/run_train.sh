n_particles=4096 # the number of particles to sample from the point cloud
early_fusion=0 # 1 for early fusion, 0 for late fusion
use_rgb=1 # 1 for using rgb features, 0 for not using rgb features
debug=0

bash ./planning/pcd_tool_classifier/train.sh $n_particles $early_fusion $use_rgb $debug