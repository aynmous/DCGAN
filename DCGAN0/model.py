import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, nz,ngf,nc):
        super(Generator, self).__init__()
        self.nz = nz
        self.ngf = ngf #生成器卷积核个数
        self.nc=nc #图片通道数

        self.main = nn.Sequential(
            #反卷积
            nn.ConvTranspose2d(self.nz, self.ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.ngf * 8),  #进行数据的归一化处理，这使得数据在进行Relu之前不会因为数据过大而导致网络性能的不稳定
            nn.ReLU(True),
            # #输出(ngf*16) × 4 × 4
            # nn.ConvTranspose2d(self.ngf * 8, self.ngf * 16, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(self.ngf * 16),
            # nn.ReLU(True),
            #  输出(ngf*8) x 4 x 4
            nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),
            #  输出(ngf*4) x 8 x 8
            nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),
            #  输出(ngf*2) x 16 x 16
            nn.ConvTranspose2d(self.ngf * 2, self.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),
            #  输出(ngf) x 32 x 32
            nn.ConvTranspose2d(self.ngf, self.nc, kernel_size=4, stride=2, padding=1, bias=False),
            # nn.ConvTranspose2d(self.ngf, self.nc, 4, 2, 1, bias=False),
            nn.Tanh()
            #  输出(nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)



class Discriminator(nn.Module):
    def __init__(self, ndf,nc):
        super(Discriminator, self).__init__()
        self.ndf=ndf #判别器卷积核个数
        self.nc=nc #图片通道数
        self.main = nn.Sequential(
            #  (nc) x 64 x 64
            nn.Conv2d(self.nc, self.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),  #0.2为负斜率的角度；防止梯度稀疏
            #  (ndf) x 32 x 32
            nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            #  (ndf*2) x 16 x 16
            nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            #  (ndf*4) x 8 x 8
            nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            #  (ndf*8) x 4 x 4
            nn.Conv2d(self.ndf * 8, 1, 4, 1, 0, bias=False),
            #  (1) x 1 x 1
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

#权重初始化
#所有的权重都以均值为0，标准差为0.2的正态分布随机初始化。
#weights_init 函数读取一个已初始化的模型并重新初始化卷积层
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)