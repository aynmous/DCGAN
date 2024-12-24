import torch
import torch.nn as nn
from torchvision import utils, datasets, transforms
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import os


class DCGAN():
    def __init__(self,lr,beta1,nz, batch_size,num_showimage,device, model_save_path,figure_save_path,generator, discriminator, data_loader,):
        self.real_label=1  #真标签
        self.fake_label=0  #假标签
        self.nz=nz
        self.batch_size=batch_size
        self.num_showimage=num_showimage
        self.device = device
        self.model_save_path=model_save_path
        self.figure_save_path=figure_save_path

        self.G = generator.to(device) 
        self.D = discriminator.to(device)
        self.opt_G = torch.optim.Adam(self.G.parameters(), lr=lr, betas=(beta1, 0.999))  #使用Adam优化算法，是带动量的惯性梯度下降算法
        self.opt_D = torch.optim.Adam(self.D.parameters(), lr=lr, betas=(beta1, 0.999))  #同上
        self.criterion = nn.BCELoss().to(device)     #损失函数 BCEloss -w(ylog x +(1 - y)log(1 - x))
                                                        #二分类 y为真实标签，x为判别器打分（sigmiod，1为真0为假），加上负号，等效于求对应标签下的最大得分

        self.dataloader=data_loader
        self.fixed_noise = torch.randn(self.num_showimage, nz, 1, 1, device=device)# 生成满足N(1,1)标准正态分布，nz维（100维），num_showimage个数的随机噪声
                                                                #固定噪声，每个EPOCH使用相同的噪声，可以观察不同EPOCH的变化

        self.img_list = []  #图
        self.G_loss_list = []  #判别器损失
        self.D_loss_list = []   #生成器损失
        self.D_x_list = []
        self.D_z_list = []



    def train(self,num_epochs):
        loss_tep = 10
        G_loss=0 #损失值初始化
        D_loss=0
        print("Starting Training Loop...")

        for epoch in range(num_epochs):

            beg_time = time.time()

            for i, data in enumerate(self.dataloader):

                #D(x)代表真实图片的概率，越大越好；D(x)固定后，D(G(z))代表生成的图片为真的概率，越小越好
                #训练判别器 maximize log(D(x)) + log(1 - D(G(z))),分成两段，越小越好

                #计算真实图像误差
                x = data[0].to(self.device)  #获取到真实的数据 device=self.device都是为了转到gpu运行
                b_size = x.size(0) #batch_size的值
                lbx = torch.full((b_size,), self.real_label, dtype=torch.float, device=self.device)  #float类型 返回指定tensor形状，填充为real_lable
                D_x = self.D(x).view(-1)    #获得输出 .view重构成一维张量
                LossD_x = self.criterion(D_x, lbx)  #损失函数计算输出和lable之间的损失
                D_x_item = D_x.mean().item()  #获得算术平均值
                
                #计算生成图像误差
                z = torch.randn(b_size, self.nz, 1, 1, device=self.device)  #生成一个噪声簇
                gz = self.G(z)  #传入生成器，生成假图

                lbz1 = torch.full((b_size,), self.fake_label, dtype=torch.float, device=self.device) #float类型 返回指定tensor形状，填充为fake_lable
                D_gz1 = self.D(gz.detach()).view(-1) #固定生成器参数，输出预测概率;.detach()做截断操作
                LossD_gz1 = self.criterion(D_gz1, lbz1) #计算输出和真实标签之间的损失
                D_gz1_item = D_gz1.mean().item() #获得算术平均值
                
                LossD = LossD_x + LossD_gz1 #完整误差
                
                self.opt_D.zero_grad()  #更新判别器
                LossD.backward() #累积梯度
                self.opt_D.step() #判别器参数优化更新

                D_loss+=LossD

                #D(G(z))越大越好
                #生成器 max log(D(G(z)))

                lbz2 = torch.full((b_size,), self.real_label, dtype=torch.float, device=self.device) # 对生成器来说生成的图是真的，lable=real_lable
                D_gz2 = self.D(gz).view(-1) #传入判别器，得到预测结果（概率）
                D_gz2_item = D_gz2.mean().item() #获得算术平均值
                LossG = self.criterion(D_gz2, lbz2)  #计算预测结果和真实标签之间的误差

                self.opt_G.zero_grad()  #梯度清零
                LossG.backward()  #反向传播
                self.opt_G.step()  #生成器参数优化更新，梯度下降
                G_loss+=LossG

                end_time = time.time()
            #-----------------------计时-----------------------
                run_time = round(end_time - beg_time)
                
                if i % 100 == 0:

                    print(
                        f'Epoch: [{epoch + 1:0>{len(str(num_epochs))}}/{num_epochs}]',
                        f'Step: [{i + 1:0>{len(str(len(self.dataloader)))}}/{len(self.dataloader)}]',
                        f'Loss-D: {LossD.item():.4f}',
                        f'Loss-G: {LossG.item():.4f}',
                        f'D(x): {D_x_item:.4f}',
                        f'D(G(z)): [{D_gz1_item:.4f}/{D_gz2_item:.4f}]',
                        f'Time: {run_time}s',
                        end='\r\n'
                    )
                

                # 保存损失值
                self.G_loss_list.append(LossG.item())
                self.D_loss_list.append(LossD.item())

                # 保存 D(X) 和 D(G(z)) 
                self.D_x_list.append(D_x_item)
                self.D_z_list.append(D_gz2_item)

            if not os.path.exists(self.model_save_path):  #如果不存在模型保存路径就新建
                os.makedirs(self.model_save_path)

            torch.save(self.D.state_dict(), self.model_save_path + 'disc_{}.pth'.format(epoch))
            torch.save(self.G.state_dict(), self.model_save_path + 'gen_{}.pth'.format(epoch))
                
            with torch.no_grad():
                fake = self.G(self.fixed_noise).detach().cpu()
                
            self.img_list.append(utils.make_grid(fake * 0.5 + 0.5, nrow=10))
            print()

        if not os.path.exists(self.figure_save_path): #如果不存在数据保存路径，就新建
            os.makedirs(self.figure_save_path)

        #绘制判别器和生成器的损失函数曲线
        plt.figure(1,figsize=(8, 4))  #编号和大小
        plt.title("Generator and Discriminator Loss During Training") #标题
        plt.plot(self.G_loss_list[::10], label="G") 
        plt.plot(self.D_loss_list[::10], label="D")
        plt.xlabel("iterations")  #“迭代” x轴
        plt.ylabel("Loss")  #“损失” y轴
        plt.axhline(y=0, label="0", c="g")  # y=0的渐近线
        plt.legend() #图例
        plt.savefig(self.figure_save_path + str(num_epochs) + 'epochs_' + 'loss.jpg', bbox_inches='tight') #将图保存到文件中：命名（路径+epoch数+epochs_+loss.jpg);bbox_inches删除图周空白


        plt.figure(2,figsize=(8, 4)) #编号和大小
        plt.title("D(x) and D(G(z)) During Training")#标题
        plt.plot(self.D_x_list[::10], label="D(x)") 
        plt.plot(self.D_z_list[::10], label="D(G(z))")
        plt.xlabel("iterations")  #“迭代” x轴
        plt.ylabel("Probability") #“概率” y轴
        plt.axhline(y=0.5, label="0.5", c="g")  # y=0.5渐近线
        plt.legend() #图例
        plt.savefig(self.figure_save_path + str(num_epochs) + 'epochs_' + 'D(x)D(G(z)).jpg', bbox_inches='tight')#将图保存到文件中：命名（路径+epoch数+epochs_+D(x)D(G(z)).jpg);bbox_inches删除图周空白

        fig = plt.figure(3,figsize=(5, 5)) #编号和大小
        plt.axis("off")  #关闭坐标轴
        ims = [[plt.imshow(item.permute(1, 2, 0), animated=True)] for item in self.img_list]
        ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
        HTML(ani.to_jshtml())
        ani.save(self.figure_save_path + str(num_epochs) + 'epochs_' + 'generation.gif')


        plt.figure(4,figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.axis("off")
        plt.title("Real Images")
        real = next(iter(self.dataloader))  # real[0]image,real[1]label
        plt.imshow(utils.make_grid(real[0][:self.num_showimage] * 0.5 + 0.5, nrow=10).permute(1, 2, 0))

        self.G.eval()
        # 生成假图
        with torch.no_grad():
            fake = self.G(self.fixed_noise).cpu()
        # 显示假图片
        plt.subplot(1, 2, 2)
        plt.axis("off")
        plt.title("Fake Images")
        fake = utils.make_grid(fake[:self.num_showimage] * 0.5 + 0.5, nrow=10).permute(1, 2, 0)
        plt.imshow(fake)

        # 保存结果
        plt.savefig(self.figure_save_path + str(num_epochs) + 'epochs_' + 'result.jpg', bbox_inches='tight')
        plt.show()




    def test(self,epoch):
        # 设置图片尺寸
        plt.figure(figsize=(8, 4))

        # 显示真实图像
        plt.subplot(1, 2, 1)
        plt.axis("off")
        plt.title("Real Images")
        real = next(iter(self.dataloader))
        plt.imshow(utils.make_grid(real[0][:self.num_showimage] * 0.5 + 0.5, nrow=10).permute(1, 2, 0))

        # 加载最优模型
        # self.G.load_state_dict(torch.load(self.model_save_path + 'disc_{}.pth'.format(epoch), map_location=torch.device(self.device)))


        # 加载预训练模型
        checkpoint = torch.load(self.model_save_path + 'disc_{}.pth'.format(epoch),
                                map_location=torch.device(self.device))

        # 检查checkpoint中是否存在'state_dict'键
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # 筛选出与模型参数相关的键，并忽略形状不匹配的参数
        pretrained_dict = {}
        for k, v in state_dict.items():
            if k in self.G.state_dict() and v.size() == self.G.state_dict()[k].size():
                pretrained_dict[k] = v

        # 加载筛选后的参数到模型中，忽略不匹配的键
        self.G.load_state_dict(pretrained_dict, strict=False)

        # 将模型设置为评估模式
        self.G.eval()
        # 生成假图片
        with torch.no_grad():
            fake = self.G(self.fixed_noise.to(self.device))
        # 显示生成的假图片
        plt.subplot(1, 2, 2)
        plt.axis("off") #不显示坐标轴
        plt.title("Fake Images") #标题
        fake = utils.make_grid(fake * 0.5 + 0.5, nrow=10)
        plt.imshow(fake.permute(1, 2, 0).cpu())  #交换tensor维度后输出

        # 保存
        plt.savefig(self.figure_save_path+'result.jpg', bbox_inches='tight')
        plt.show()




