import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# 加载图片
image_name = 'MOT17-02-000451'
image_path = '/home/livion/Documents/github/dataset/MOT_yolo/images/val/' + image_name + '.jpg'
gt_path = '/home/livion/Documents/github/dataset/MOT_yolo/labels/val/' + image_name + '.txt'
image = Image.open(image_path)
img_width, img_height = image.size

# 读取YOLO格式的Bounding Box坐标
with open(gt_path, 'r') as f:
    lines = f.readlines()
    boxes = [list(map(float, line.strip().split()[1:])) for line in lines]

# 使用matplotlib绘制Bounding Box
fig, ax = plt.subplots(1)
ax.imshow(image)

for box in boxes:
    x_center, y_center, width, height = box
    x = (x_center - width / 2) * img_width
    y = (y_center - height / 2) * img_height
    rect = patches.Rectangle((x, y), width * img_width, height * img_height, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

plt.tight_layout()
# plt.show()
plt.savefig('visualization/images/'+image_name+'.png')
