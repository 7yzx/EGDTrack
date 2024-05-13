#!/bin/bash
export CUDA_HOME=/usr/local/cuda
# 定义其他固定参数
dataset_name="depthtrack"
input_dtype="rgbcolormap"

# 使用 for 循环遍历从 51 到 91，每次增加 5
for i in $(seq 51 5 81); do
    # 构造 tracker_param 参数
    tracker="mycocoDeT_DiMP50_Mean_${i}"
    echo $tracker_param
    # 执行 Python 程序
    python run_tracker.py dimp "$tracker" --dataset_name "$dataset_name" --input_dtype "$input_dtype"

    # 检查 Python 程序的退出码
    if [ $? -eq 0 ]; then
        # 程序成功执行
        echo "Python 程序执行成功，tracker_param = $tracker"
    else
        # 程序执行失败
        echo "Python 程序执行失败，tracker_param = $tracker"
        # 根据需要在这里可以进行错误处理，例如停止脚本运行
        exit 1
    fi
done
python run_tracker.py dimp mycocoDeT_DiMP50_Mean_85 --dataset_name "$dataset_name" --input_dtype "$input_dtype"


for i in $(seq 51 5 91); do
    # 构造 tracker_param 参数
    tracker_param="mygot10kcoco_DiMP_Mean_${i}"
    echo $tracker_param
    # 执行 Python 程序
    python run_tracker.py dimp "$tracker_param" --dataset_name "$dataset_name" --input_dtype "$input_dtype"

    # 检查 Python 程序的退出码
    if [ $? -eq 0 ]; then
        # 程序成功执行
        echo "Python 程序执行成功，tracker_param = $tracker_param"
    else
        # 程序执行失败
        echo "Python 程序执行失败，tracker_param = $tracker_param"
        # 根据需要在这里可以进行错误处理，例如停止脚本运行
        exit 1
    fi
done
#python run_tracker_debug.py --tracker_name atom --tracker_param DeT_ATOM_Max --dataset_name depthtrack --input_dtype rgbcolormap
#python run_tracker_debug.py --tracker_name atom --tracker_param DeT_ATOM_Max_mycoco_57 --dataset_name depthtrack --input_dtype rgbcolormap
#python run_tracker_debug.py --tracker_name atom --tracker_param DeT_ATOM_Mean --dataset_name depthtrack --input_dtype rgbcolormap
#python run_tracker_debug.py --tracker_name atom --tracker_param DeT_ATOM_Mean_retrain_54 --dataset_name depthtrack --input_dtype rgbcolormap