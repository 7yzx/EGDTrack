import os

# tracking_result = './votd2020_test/results/coco'
tracking_result = './depthtrack_ws/results'
names = [f.name for f in os.scandir(tracking_result) if f.is_dir()]
names = sorted(names)
name_str = str()
for i in range(0,len(names)):
    name_str = name_str +names[i]+' '
print(name_str)
with open('./depthtrack_ws/initdet.txt' , 'w') as f :
    for i in range(0,len(names)):
        f.write('['+names[i]+']'+
                '\n'+'protocol = traxpython'+'\n'
                +'command = jjjjj'+'\n'
                )

