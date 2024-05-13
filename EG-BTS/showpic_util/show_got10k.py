import cv2
from PIL import Image
import os
import matplotlib.pyplot as plt
from PIL import Image
import svgwrite


def extract_first_images(dataset_path, output_path):
    class_first_images = {}

    list_path = os.path.join(dataset_path,"list.txt")
    class_names = []
    with open(list_path) as ff:
        for line in ff:
            class_names.append(line.rstrip())
    class_names = class_names[:30]
    num_class = len(class_names)

    for i in range(0,num_class):
        first_img_path = os.path.join(dataset_path,'train',class_names[i],"00000001.jpg")
        if not os.path.exists(first_img_path):
            print("no ",first_img_path)

        img = plt.imread(first_img_path)
        # img = cv2.imread(first_img_path)#改变格式后大小变大了
        output_svg_path = os.path.join(output_path,class_names[i]+".png")

        # 保存图像
        plt.imsave(output_svg_path, img)
        # cv2.imwrite(output_svg_path,img)
        # img = plt.imread(first_img_path)

        # output_svg_path = os.path.join(output_path+class_names[i]+".svg")
        # # 创建SVG图像
        # dwg = svgwrite.Drawing(output_svg_path, profile='tiny')
        # dwg.add(svgwrite.image.Image(first_img_path, insert=(0, 0)))
        #
        # # 保存SVG图像
        # dwg.save()


def main():
    dataset_path = '/home/whuai/dataset_track/GOT-10K'
    output_path = '../got10k/'
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    extract_first_images(dataset_path, output_path)
    print("已提取每一类的第一张图片并保存到指定目录中。")


if __name__ == "__main__":
    main()
