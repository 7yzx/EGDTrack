import json
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, Union,List
import matplotlib.font_manager as fm
from matplotlib.font_manager import FontProperties
font = FontProperties(fname="/home/whuai/simhei.ttf", size=14)  # 步骤二
font_legend = FontProperties(fname="/home/whuai/simhei.ttf", size=8)  # 步骤二

import random
# 设置中文字体为黑体
_PALETTE = {
    "white": (1, 1, 1),
    "black": (0, 0, 0),
    "red": (1, 0, 0),
    "green": (0, 1, 0),
    "blue": (0, 0, 1),
    "cyan": (0, 1, 1),
    "magenta": (1, 0, 1),
    "yellow": (1, 1, 0),
    "gray": (0.5, 0.5, 0.5),
}


def resolve_color(color: Union[Tuple[float, float, float], str]) -> Tuple[float, float, float]:
    if isinstance(color, str):
        return _PALETTE.get(color, (0, 0, 0))

    return (np.clip(color[0], 0, 1), np.clip(color[1], 0, 1), np.clip(color[2], 0, 1))


def generate_low_saturation_color() -> Tuple[float, float, float]:
    hue = np.random.rand()  # 随机选择色调
    saturation = np.random.uniform(0.2, 0.5)  # 在较低的饱和度范围内随机选择
    value = np.random.uniform(0.7, 1.0)  # 随机选择明度
    return resolve_color((hue, saturation, value))


def generate_random_colors(num_colors):
    # colors = ['#2878b5', '#9ac9db', '#f8ac8c', '#c82423', '#F9A31A']
    colors = ['#FF6100','#5086C4']
    return random.sample(colors, num_colors)


def generate_special_color(num_colors: int) -> List[Tuple[float, float, float]]:
    colors = []
    if num_colors >= 4:
        for _ in range(num_colors):
            hue = np.random.rand()  # 随机选择色调
            saturation = np.random.uniform(0.2, 0.5)  # 在较低的饱和度范围内随机选择
            value = np.random.uniform(0.7, 1.0)  # 随机选择明度
            colors.append(resolve_color((hue, saturation, value)))
    else:
        # 保证至少返回4种颜色，具有高区分度
        for _ in range(4):
            hue = np.random.rand()  # 随机选择色调
            saturation = np.random.uniform(0.5, 1.0)  # 在较高的饱和度范围内随机选择
            value = np.random.uniform(0.7, 1.0)  # 随机选择明度
            colors.append(resolve_color((hue, saturation, value)))
    return colors

def plot_tracker_result(cocojson,orinjson):
    #读取所有的信息
    with open(cocojson) as f:
        file = json.load(f)
        coco_results = file['results']['rgbd-unsupervised']['results']
        coco_trackers = file['trackers']
        coco_tracker_names = list(coco_trackers.keys())
        coco_Tracking_PRF = coco_results[0]
        coco_results_PR_curve = coco_results[1]
        coco_results_F_curve = coco_results[2]

    with open(orinjson) as f:
        file = json.load(f)
        orin_results = file['results']['rgbd-unsupervised']['results']
        orin_trackers = file['trackers']
        orin_tracker_names = list(orin_trackers.keys())
        orin_Tracking_PRF = orin_results[0]
        orin_results_PR_curve = orin_results[1]
        orin_results_F_curve = orin_results[2]

    fig, (ax3,ax1, ax2) = plt.subplots(3, 1, figsize=(15, 20),dpi= 100)
    plt.subplots_adjust(hspace=1)

    # fig2, (ax_pr,ax_f) = plt.subplots(1,2,figsize=(15, 20))
    # color_coco,color_orin = generate_random_colors(2)[0],generate_special_color(2)[1]
    color_coco = '#d80552'
    color_orin = '#4b65fe'

    plt.subplots_adjust(hspace=0.5)

    Pv_coco, Rv_coco, Fv_coco = [], [], []
    Pv_orin, Rv_orin, Fv_orin = [], [], []
    Presion_y = []
    Recall_x = []
    tracker_list_coco = []
    fx_coco = []
    tracker_list_orin = []
    fx_orin = []
    for i in range(0, 100):
        fx_coco.append(i)

    for i in range(0, len(coco_tracker_names)):
        tracker_name = int(coco_tracker_names[i].split('_')[-1])
        tracker_list_coco.append(tracker_name-50)
        Pv_coco.append(coco_Tracking_PRF[i][0])
        Rv_coco.append(coco_Tracking_PRF[i][1])
        Fv_coco.append(coco_Tracking_PRF[i][2])
        Recall_x=[data[0] for data in coco_results_PR_curve[i][0]]
        Presion_y=[data[1] for data in coco_results_PR_curve[i][0]]
        # ax_pr.plot(Recall_x,Presion_y,label=str(tracker_name-50))


    for i in range(0, len(orin_tracker_names)):
        tracker_name = int(orin_tracker_names[i].split('_')[-1])
        tracker_list_orin.append(tracker_name-50)
        Pv_orin.append(orin_Tracking_PRF[i][0])
        Rv_orin.append(orin_Tracking_PRF[i][1])
        Fv_orin.append(orin_Tracking_PRF[i][2])



    ax1.plot(tracker_list_coco, Pv_coco, color=color_coco,label="EGDTrack")
    ax1.plot(tracker_list_orin, Pv_orin, color=color_orin,label="BTSTrack")
    # ax1.set_xlabel("训练轮数",fontproperties=font,loc='right',fontsize=10)
    # ax1.set_ylabel("精准度",fontproperties=font,loc='top',rotation='horizontal',fontsize=10)
    # ax1.set_title('训练轮数与精准度曲线',fontproperties=font)

    ax2.plot(tracker_list_coco, Rv_coco, color=color_coco,label="EGDTrack")
    ax2.plot(tracker_list_orin, Rv_orin, color=color_orin,label="BTSTrack")
    # ax2.set_xlabel("训练轮数",fontproperties=font,loc='right',fontsize=10)
    # ax2.set_ylabel("召回率",fontproperties=font,loc='top',rotation='horizontal',fontsize=10)
    # ax2.set_title('训练轮数与召回率曲线',fontproperties=font)
    ax3.plot(tracker_list_coco, Fv_coco, color=color_coco,label="EGDTrack")
    ax3.plot(tracker_list_orin, Fv_orin, color=color_orin,label="BTSTrack")
    # ax3.set_xlabel("训练轮数",fontproperties=font,loc='right',fontsize=10)
    # ax3.set_ylabel("F分数",fontproperties=font,loc='top',rotation='horizontal',fontsize=10)
    # ax3.set_title('训练轮数与F分数曲线',fontproperties=font)

    # 标记最大值
    sorted_indices_coco = sorted(range(len(Fv_coco)), key=lambda i: Fv_coco[i], reverse=True)
    max_coco_index = sorted_indices_coco[0]
    fv_coco_max = Fv_coco[max_coco_index]
    max_coco_index = tracker_list_coco[max_coco_index]
    second_coco_index = sorted_indices_coco[1]
    fv_coco_second = Fv_coco[second_coco_index]
    second_coco_index = tracker_list_coco[second_coco_index]
    third_coco_index = sorted_indices_coco[2]
    fv_coco_third = Fv_coco[third_coco_index]
    third_coco_index = tracker_list_coco[third_coco_index]
    # orin
    sorted_indices_orin = sorted(range(len(Fv_orin)), key=lambda i: Fv_orin[i], reverse=True)
    max_orin_index = sorted_indices_orin[0]
    fv_orin_max = Fv_orin[max_orin_index]
    max_orin_index = tracker_list_orin[max_orin_index]
    second_orin_index = sorted_indices_orin[1]  # Index of the second maximum value
    fv_orin_second = Fv_orin[second_orin_index]
    second_orin_index = tracker_list_orin[second_orin_index]
    third_orin_index = sorted_indices_orin[2]
    fv_orin_third = Fv_orin[third_orin_index]
    third_orin_index = tracker_list_orin[third_orin_index]

    # max_color = '#228B22'#yelloe
    max_color = 'red'
    second_color = '#A020EF'#purple
    third_color = '#228B22'#blue
    #max coco
    ax3.scatter(max_coco_index, fv_coco_max, marker='^', color=color_coco, s=30,zorder=2,label='排名1')
    ax3.annotate(f'{fv_coco_max:.3f}', xy=(max_coco_index, fv_coco_max),
                 xytext=(max_coco_index-0.5, fv_coco_max + 0.01),
                  fontsize='small')

    ax3.scatter(second_coco_index, fv_coco_second+0.001, marker='+', color=color_coco, s=30,zorder=2,label='排名2')
    ax3.annotate(f'{fv_coco_second:.3f}', xy=(second_coco_index,fv_coco_second),
                 xytext=(second_coco_index, fv_coco_second + 0.01),
                  fontsize='small')

    ax3.scatter(third_coco_index, fv_coco_third, marker='s', color=color_coco, s=25,zorder=2,label='排名3')
    ax3.annotate(f'{fv_coco_third:.3f}', xy=(third_coco_index,fv_coco_third),
                 xytext=(third_coco_index, fv_coco_third + 0.01),
                  fontsize='small')


    # max orin
    ax3.scatter(max_orin_index, fv_orin_max, marker='^', color=color_orin, s=30,zorder=2,label='排名1')
    ax3.annotate(f'{fv_orin_max:.3f}', xy=(max_orin_index, fv_orin_max),
                 xytext=(max_orin_index-1, fv_orin_max - 0.02),
                  fontsize='small')

    ax3.scatter(second_orin_index, fv_orin_second, marker='+', color=color_orin, s=30,zorder=2,label='排名2')
    ax3.annotate(f'{fv_orin_second:.3f}', xy=(second_orin_index,fv_orin_second),
                     xytext=(second_orin_index-0.5, fv_orin_second - 0.03),
                  fontsize='small')
    ax3.scatter(third_orin_index, fv_orin_third, marker='s', color=color_orin, s=25,zorder=2,label='排名3')
    ax3.annotate(f'{fv_orin_third:.3f}', xy=(third_orin_index,fv_orin_third),
                 xytext=(third_orin_index-0.3, fv_orin_third - 0.015),
                  fontsize='small')

    # Lighten borders
    # 设置边缘透明度
    for ax in [ax1, ax2, ax3]:
        ax.grid(True, which='both', ls='dashed')
        ax.spines["top"].set_alpha(0)
        ax.spines["bottom"].set_alpha(0.3)
        ax.spines["right"].set_alpha(0)
        ax.spines["left"].set_alpha(0.3)
        ax.set_ylim(0.5, 0.7)
        ax.set_xlim(0, 55)


    font1 = FontProperties(fname="/home/whuai/simhei.ttf", size=10)  # 步骤二

    ax1.legend(loc='upper right',prop=font1)
    ax2.legend(loc='upper right',prop=font1)

    ax3.legend(loc='upper right',prop=font_legend)
    # for ax in [ax_pr, ax_f]:
    #     # ax.grid(True, which='both', ls='dashed')
    #     ax.set_ylim(0,1)
    #
    #     ax.legend(loc='lower left', fontsize='small')

    # 显示图形
    plt.show()
    print("result")


def plot_orin_coco_result(cocojson, orinjson):
        # 读取所有的信息
        with open(cocojson) as f:
            file = json.load(f)
            coco_results = file['results']['rgbd-unsupervised']['results']
            coco_trackers = file['trackers']
            coco_tracker_names = list(coco_trackers.keys())
            coco_Tracking_PRF = coco_results[0]
            coco_results_PR_curve = coco_results[1]
            coco_results_F_curve = coco_results[2]

        with open(orinjson) as f:
            file = json.load(f)
            orin_results = file['results']['rgbd-unsupervised']['results']
            orin_trackers = file['trackers']
            orin_tracker_names = list(orin_trackers.keys())
            orin_Tracking_PRF = orin_results[0]
            orin_results_PR_curve = orin_results[1]
            orin_results_F_curve = orin_results[2]

        fig2, (ax_pr, ax_f) = plt.subplots(1, 2, figsize=(15, 4),gridspec_kw={'width_ratios':[1,2]})
        color_coco, color_orin = generate_random_colors(2)[0], generate_special_color(2)[1]

        plt.subplots_adjust(hspace=1)

        Pv_coco, Rv_coco, Fv_coco = [], [], []
        Pv_orin, Rv_orin, Fv_orin = [], [], []
        Presion_y = []
        Recall_x = []
        tracker_list_coco = []
        fx_coco = []
        tracker_list_orin = []
        fx_orin = []
        for i in range(0, 100):
            fx_coco.append(i)

        for i in range(0, len(coco_tracker_names)):
            tracker_name = int(coco_tracker_names[i].split('_')[-1])
            tracker_list_coco.append(tracker_name - 50)
            Recall_x = [data[0] for data in coco_results_PR_curve[i][0]]
            Presion_y = [data[1] for data in coco_results_PR_curve[i][0]]
            ax_pr.plot(Recall_x, Presion_y, label='EGDTrack')
            ax_f.plot(fx_coco,coco_results_F_curve[i][0],label='EGDTrack')

        for i in range(0, len(orin_tracker_names)):
            tracker_name = int(orin_tracker_names[i].split('_')[-1])
            tracker_list_orin.append(tracker_name - 50)

            Recall_x = [data[0] for data in orin_results_PR_curve[i][0]]
            Presion_y = [data[1] for data in orin_results_PR_curve[i][0]]
            ax_pr.plot(Recall_x, Presion_y, label='BTSTrack')
            ax_f.plot(fx_coco,orin_results_F_curve[i][0], label='BTSTrack')


        ax_pr.grid(True, which='both', ls='dashed')
        ax_pr.set_ylim(0.4, 1)
        ax_pr.legend(loc='upper right', fontsize='small')
        ax_pr.set_linewidth = 2
        # ax_pr.set_xlabel("召回率", fontproperties=font, loc='right', fontsize=10)
        # ax_pr.set_ylabel("精准度", fontproperties=font, loc='top', rotation='horizontal', fontsize=10)
        # ax_pr.set_title("精准度-召回率曲线",fontproperties=font,fontsize=15)
        ax_f.grid(True, which='both', ls='dashed')
        ax_f.set_ylim(0, 1)
        ax_f.legend(loc='upper right', fontsize='small')
        ax_f.set_linewidth = 2
        # ax_f.set_xlabel("阈值", fontproperties=font, loc='right', fontsize=10)
        # ax_f.set_ylabel("F分数", fontproperties=font, loc='top', rotation='horizontal', fontsize=10)
        # ax_f.set_title("F分数曲线",fontproperties=font,fontsize=15)

        # 显示图形
        plt.show()
        print("result")

def plot_one_result(json_file):
    with open(json_file,'r') as f:
        fig, (ax1, ax2) = plt.subplots(2)

        result = json.load(f)
        results = result['results']['rgbd-unsupervised']['results']
        Tracking_PR = results[0]
        PR_curve = results[1][0][0]
        F_curve = results[2][0][0]

    Pv,Rv,Fv = Tracking_PR[0][0],Tracking_PR[0][1],Tracking_PR[0][2]
    fx=[]
    for i in range(0,len(F_curve)):
        fx.append(i)
    # plt.plot(fx,F_curve)

    pr_x = [data[0] for data in PR_curve]
    pr_y = [data[1] for data in PR_curve]
    # plt.ylim(0,1)
    # plt.plot(pr_x,pr_y)

    ax1.plot(pr_x, pr_y, color=generate_low_saturation_color())
    ax1.set_title('PR_curve')
    ax1.set_ylim(0,1)

    ax2.plot(fx, F_curve, color=generate_low_saturation_color())
    ax2.set_title('F_curve')

    # 调整布局
    plt.tight_layout()

    # 显示图形
    plt.show()
    print("result")


def plot_csv(file):
    with open(file,'r') as f:
        lines = [f.strip() for f in f.readlines()]
    step1 = []
    value1= []
    for i in range(1,len(lines)):
        step1.append(int(lines[i].split(',')[1])-50)
        value1.append(float(lines[i].split(',')[2]))
    plt.grid(True, which='both', ls='dashed')
    plt.plot(step1,value1,color='#425066')
    # plt.xlabel("训练轮数",fontproperties=font,loc='right',fontsize=10)
    # plt.ylabel("损失",fontproperties=font,loc='top',rotation='horizontal',fontsize=10)
    # plt.title('训练轮数与损失',fontproperties=font)
    plt.show()


json_file = './votd2020_test/analysis/cdtb_all.json'
cdtb_coco = './votd2020_test/analysis/cdtb_coco.json'
cdtb_orin = './votd2020_test/analysis/cdtb_orin.json'
DepthTrack_coco = './depthtrack_ws/analysis/depthtrack_coco.json'
DepthTrack_orin = './depthtrack_ws/analysis/depthtrack_orin.json'
DepthTrack_orin_test = './depthtrack_ws/analysis/detorin79.json'
DepthTrack_coco_test = './depthtrack_ws/analysis/detcoco78.json'
csv_file = '/home/whuai/Documents/csv.csv'

# plot_csv(csv_file)# plot loss
# plot_one_result('./votd2020_test/analysis/2024-04-23T23-20-29.444081/results.json')
# plot_tracker_result(cdtb_coco,cdtb_orin)
# plot_tracker_result(DepthTrack_coco,DepthTrack_orin)
plot_tracker_result(cdtb_coco,cdtb_orin)
# plot_orin_coco_result(DepthTrack_coco_test,DepthTrack_orin_test)

