from data import ReadData
from model import Discriminator, Generator, weights_init
from net import DCGAN
import torch

ngpu=1
ngf=64 #生成器卷积核个数
ndf=64  #判别器卷积核个数
nc=3  #通道数
nz=100  #噪声维度
lr=0.001 #学习率
beta1=0.5 # 正则化系数，Adam优化器参数，稳定训练，0.9会发生震荡
batch_size=200 #一批一百个
num_showimage=100 #一次一百张图

data_path="./data"  #原图片路径
model_save_path="./models1/"   #模型保存路径
figure_save_path="./figures-100epoch/"  #数据保存路径
# 选择设备（MPS 优先）
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# 检查设备
print(f"Using device: {device}")

dataset=ReadData(data_path)  #加载数据
dataloader=dataset.getdataloader(batch_size=batch_size)

G = Generator(nz,ngf,nc).apply(weights_init) #实例化生成器
print(G)  #实例化判别器
D = Discriminator(ndf,nc).apply(weights_init)
print(D)

dcgan=DCGAN( lr,beta1,nz,batch_size,num_showimage,device, model_save_path,figure_save_path,G, D, dataloader) #实例化

dcgan.train(num_epochs=100)
# dcgan.test(epoch=0)




