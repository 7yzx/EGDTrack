#!/bin/bash
export CUDA_HOME=/usr/local/cuda
# 定义其他固定参数
dataset_name="depthtrack"
input_dtype="rgbcolormap"
number1="69 91 99"

# 使用 for 循环遍历从 51 到 91，每次增加 5
for i in $number1; do
    echo $i
    # 构造 tracker_param 参数
    tracker_dir="depthtrack_cocograd1_Mean_test"
    tracker="mycocoDeT_DiMP50_Mean_${i}"
    # 执行 Python 程序
    python my_run_tracker.py dimp "$tracker_dir" "$tracker" "$dataset_name"

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

number="51 54 57 60 66 69 86 91 96"
for i in $number; do
    # 构造 tracker_param 参数
    tracker_dir="depthtrack_cocoorin_Mean_test"
    tracker="orincoco_DiMP_Mean_${i}"
    # 执行 Python 程序
    python my_run_tracker.py dimp "$tracker_dir" "$tracker" "$dataset_name"

    # 检查 Python 程序的退出码
    if [ $? -eq 0 ]; then
        # 程序成功执行
        echo "Python 程序执行成功"
    else
        # 程序执行失败
        echo "Python 程序执行失败"
        # 根据需要在这里可以进行错误处理，例如停止脚本运行
        exit 1
    fi
done