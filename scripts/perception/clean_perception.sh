tool_type="gripper_sym_plane_robot_v4"
perception_dir="./dump/perception/gripper_sym_plane_robot_v4_18-Sep-2022-10:32:22.876372"

last_dir=349

mkdir -p $perception_dir/archive
echo "archiving..."
while IFS='\n' read -ra arr; do
    for idx in ${arr[@]}; do
        echo $idx
        mv $perception_dir/$idx $perception_dir/archive/$idx
    done
done < ./dump/perception/inspect/$tool_type/inspect.txt

echo "patching..."
while IFS='\n' read -ra arr; do
    for ((i=${#arr[@]}-1; i>=0; i--)); do
        while [ ! -d "$perception_dir/$last_dir" ]; do
            last_dir=$((last_dir - 1))
        done
        if [ $last_dir -gt ${arr[i]} ]; then
            echo "$last_dir -> ${arr[i]}"
            mv $perception_dir/$last_dir $perception_dir/${arr[i]}
        fi
    done
done < ./dump/perception/inspect/$tool_type/inspect.txt
