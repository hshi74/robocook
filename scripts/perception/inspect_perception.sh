tool_type="press_square_robot_v4"
perception_dir="./dump/perception/press_square_robot_v4_29-Aug-2022-01:11:10.030099"

mkdir -p ./dump/perception/inspect

# echo $perception_dir
for sub_dir in $(find $perception_dir -maxdepth 1 -type d); do
    file="$sub_dir/repr.mp4"
    if test -f "$file"; then        
        echo $sub_dir
        vid_idx=$(basename -- "$sub_dir")
        cp $file ./dump/perception/inspect/$tool_type/$vid_idx.mp4
    fi
done

# touch ./dump/perception/inspect/inspect.txt