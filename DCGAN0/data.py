from torch.utils.data import DataLoader
from torchvision import utils, datasets, transforms

class ReadData():
    def __init__(self,data_path,image_size=64):
        self.root=data_path
        self.image_size=image_size
        self.dataset=self.getdataset()
    def getdataset(self):

        dataset = datasets.ImageFolder(root=self.root, 
                                       transform=transforms.Compose([   #组合多个tv.transforms操作,定义好transforms组合操作后，直接传入图片即可进行处理
                                       transforms.Resize(self.image_size), #重置尺寸64*64
                                       transforms.CenterCrop(self.image_size), #中心裁剪
                                       transforms.ToTensor(), #将读到的图片转为torch image类型（通道，像素，像素）,且把像素范围转为[0，1]
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), #执行image = (image - mean)/std 数据归一化操作，一参数是mean,mean = (0.5, 0.5, 0.5)，二参数std
                                   ]))
        

        print(f'Total Size of Dataset: {len(dataset)}')
        return dataset  #dataset是一个包装类，将数据包装成Dataset类，方便之后传入DataLoader中

    def getdataloader(self,batch_size=128):
        dataloader = DataLoader(
            self.dataset, #数据加载
            batch_size=batch_size, #批处理大小
            shuffle=True,  #是否进行随机洗牌
            num_workers=0) #每一轮迭代时，只有主进程加载BATCH

        return dataloader

if __name__ =='__main__':
    dset=ReadData('./oxford17_all')
    print('lalala')
    dloader=dset.getdataloader()
