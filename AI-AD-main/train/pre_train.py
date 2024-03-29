# 学   校：华南师范大学
# 学   生：温志森
# 开发时间：2023/5/15 1:24
import torch
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np
model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
model.eval()

# 定义图像预处理函数
transform_train = Compose([
    Resize((299, 299)),
    ToTensor(),
    Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# 创建训练集和测试集的ImageFolder数据集对象
train_dataset = ImageFolder('/home/bzq/wzs/OCT/train', transform_train)
test_dataset = ImageFolder('/home/bzq/wzs/OCT/test', transform_train)

# 创建训练集和测试集的DataLoader，指定batch_size
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 创建Inception-v3模型，并将最后一层fc层输出改为2类
# model = models.inception_v3(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 2)

# 定义损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# 将模型转移到GPU上进行训练
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# # 训练模型
# for epoch in range(10):
#     running_loss = 0.0
#     for i, data in enumerate(train_loader, 0):
#         inputs, labels = data
#         inputs, labels = inputs.to(device), labels.to(device)
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#
#         running_loss += loss.item()
#         if i % 10 == 9:
#             print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))
#             running_loss = 0.0
#
# # 测试模型性能
# correct = 0
# total = 0
# top1_cnt = 0
# top5_cnt = 0
# with torch.no_grad():
#     for data in test_loader:
#         images, labels = data
#         images, labels = images.to(device), labels.to(device)
#         outputs = model(images)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#
#         # 统计Top1和Top5准确率
#         top5_probs, top5_locs = torch.topk(outputs.data, k=5, dim=1)
#         for i, pred_label in enumerate(predicted):
#             if pred_label == labels[i]:
#                 top1_cnt += 1
#             if labels[i] in top5_locs[i]:
#                 top5_cnt += 1
#
# # 计算微查准律、微召回率和微F1得分
# tp = top1_cnt
# fp = total - tp
# fn = total - top1_cnt
# precision = tp / (tp + fp)
# recall = tp / (tp + fn)
# f1 = 2 * precision * recall / (precision + recall)
#
# # 打印模型的性能指标
# print('Accuracy on test set: %.2f %%' % (100 * correct / total))
# print('Top1 accuracy on test set: %.2f %%' % (100 * top1_cnt / total))
# print('Top5 accuracy on test set: %.2f %%' % (100 * top5_cnt / total))
# print('Micro-precision: %.4f' % precision)
# print('Micro-recall: %.4f' % recall)
# print('Micro-F1 score: %.4f' % f1)
for epoch in range(10):
    running_loss = 0.0
    top1_cnt_train = 0
    # top3_cnt_train = 0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 9:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))
            running_loss = 0.0

        # 统计Top1
        # top3_probs, top3_locs = torch.topk(outputs.data, k=3, dim=1)
        for i, pred_label in enumerate(torch.argmax(outputs, dim=1)):
            if pred_label == labels[i]:
                top1_cnt_train += 1

    # 在部分测试集上测试模型性能
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    top1_acc_train = 100 * top1_cnt_train / len(train_loader.dataset)
    # top3_acc_train = 100 * top3_cnt_train / len(train_loader.dataset)
    acc_test = 100 * correct / total

    # 打印模型的性能指标
    print('Epoch %d:' % (epoch + 1))
    print('Training Top1 accuracy: %.2f %%' % (top1_acc_train))
    # print('Training Top3 accuracy: %.2f %%' % (top3_acc_train))
    print('Testing accuracy: %.2f %%' % (acc_test))

# 计算微查准律、微召回率和微F1得分
tp = correct
fp = total - tp
fn = total - correct
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1 = 2 * precision * recall / (precision + recall)
print('微查准率: %.2f%%' % (precision * 100))
print('微召回率: %.2f%%' % (recall * 100))
print('微F1得分: %.2f%%' % (f1 * 100))

# 绘制条形图
x = ['micro-precision', 'micro-recall', 'micro-F1 score']
y = [precision, recall, f1]
plt.bar(x, y, color='b')
plt.ylim((0, 1))
plt.title('Inception-v3 Performance on OCT Dataset')
plt.savefig('performance.png', dpi=300, bbox_inches='tight')
plt.show()