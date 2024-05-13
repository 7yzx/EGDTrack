#!/bin/bash
export CUDA_HOME=/usr/local/cuda
# 定义其他固定参数
dataset_name="depthtrack"
input_dtype="rgbcolormap"


python run_tracker.py dimp mycocoDeT_DiMP50_Mean_74 --dataset_name "$dataset_name" --input_dtype "$input_dtype"
if [ $? -eq 0 ]; then
    # 程序成功执行
    echo "Python 程序执行成功"
else
    # 程序执行失败
    echo "Python 程序执行失败"
    # 根据需要在这里可以进行错误处理，例如停止脚本运行
    exit 1
fi
python run_tracker.py dimp mycocoDeT_DiMP50_Mean_75 --dataset_name "$dataset_name" --input_dtype "$input_dtype"
if [ $? -eq 0 ]; then
    # 程序成功执行
    echo "Python 程序执行成功"
else
    # 程序执行失败
    echo "Python 程序执行失败"
    # 根据需要在这里可以进行错误处理，例如停止脚本运行
    exit 1
fi
python run_tracker.py dimp mycocoDeT_DiMP50_Mean_79 --dataset_name "$dataset_name" --input_dtype "$input_dtype"
if [ $? -eq 0 ]; then
    # 程序成功执行
    echo "Python 程序执行成功"
else
    # 程序执行失败
    echo "Python 程序执行失败"
    # 根据需要在这里可以进行错误处理，例如停止脚本运行
    exit 1
fi
python run_tracker.py dimp mycocoDeT_DiMP50_Mean_80 --dataset_name "$dataset_name" --input_dtype "$input_dtype"
if [ $? -eq 0 ]; then
    # 程序成功执行
    echo "Python 程序执行成功"
else
    # 程序执行失败
    echo "Python 程序执行失败"
    # 根据需要在这里可以进行错误处理，例如停止脚本运行
    exit 1
fi

#for i in $(seq 1 1 4); do
#    # 构造 tracker_param 参数
#    tracker_param="mygot10kcoco_DiMP_Mean_${i}"
#    echo $tracker_param
#    # 执行 Python 程序
##    python run_tracker.py dimp "$tracker_param" --dataset_name "$dataset_name" --input_dtype "$input_dtype"
##
##    # 检查 Python 程序的退出码
##    if [ $? -eq 0 ]; then
##        # 程序成功执行
##        echo "Python 程序执行成功，tracker_param = $tracker_param"
##    else
##        # 程序执行失败
##        echo "Python 程序执行失败，tracker_param = $tracker_param"
##        # 根据需要在这里可以进行错误处理，例如停止脚本运行
##        exit 1
##    fi
#done
#python run_tracker_debug.py --tracker_name atom --tracker_param DeT_ATOM_Max --dataset_name depthtrack --input_dtype rgbcolormap
#python run_tracker_debug.py --tracker_name atom --tracker_param DeT_ATOM_Max_mycoco_57 --dataset_name depthtrack --input_dtype rgbcolormap
#python run_tracker_debug.py --tracker_name atom --tracker_param DeT_ATOM_Mean --dataset_name depthtrack --input_dtype rgbcolormap
#python run_tracker_debug.py --tracker_name atom --tracker_param DeT_ATOM_Mean_retrain_54 --dataset_name depthtrack --input_dtype rgbcolormap