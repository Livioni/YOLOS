import matplotlib.pyplot as plt
import json

# 从txt文件中读取数据
with open('results/MOT15Det_tiny/log.txt', 'r') as f:
    lines = f.readlines()
    data = [json.loads(line) for line in lines]

# 提取需要的数据
epochs = [d['epoch'] for d in data]
train_loss = [d['train_loss'] for d in data]
test_loss = [d['test_loss'] for d in data]
train_class_error = [d['train_class_error'] for d in data]
test_class_error = [d['test_class_error'] for d in data]

# 绘制训练和测试的损失曲线
plt.figure(figsize=(10, 5))
plt.plot(epochs, train_loss, label='Train Loss', color='blue')
plt.plot(epochs, test_loss, label='Test Loss', color='red')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Train and Test Loss over Epochs')
plt.legend()
plt.grid(True)
# plt.show()
plt.tight_layout()
plt.savefig('results/MOT15Det_tiny/loss.png')

# 绘制训练和测试的分类错误曲线
plt.figure(figsize=(10, 5))
plt.plot(epochs, train_class_error, label='Train Class Error', color='blue')
plt.plot(epochs, test_class_error, label='Test Class Error', color='red')
plt.xlabel('Epochs')
plt.ylabel('Class Error')
plt.title('Train and Test Class Error over Epochs')
plt.legend()
plt.grid(True)
# plt.show()
plt.tight_layout()
plt.savefig('results/MOT15Det_tiny/class_error.png')

# 你可以继续为其他指标添加更多的绘图代码...
