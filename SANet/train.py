import argparse
import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data as data
from PIL import Image
from PIL import ImageFile
from torchvision import transforms
from tqdm import tqdm
from sampler import InfiniteSamplerWrapper
from net import decoder, vgg, Net

cudnn.benchmark = True
Image.MAX_IMAGE_PIXELS = None  # 避免解压错误
ImageFile.LOAD_TRUNCATED_IMAGES = True  # 避免系统错误


# 对图像的转换
def train_transform():
    transform_list = [
        transforms.Resize(size=(512, 512)),
        transforms.RandomCrop(256),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)


# 加载数据集
class FlatFolderDataset(data.Dataset):
    def __init__(self, root, transform):
        super(FlatFolderDataset, self).__init__()
        self.root = root
        self.paths = os.listdir(self.root)
        self.transform = transform

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'FlatFolderDataset'


# 调整学习率
def adjust_learning_rate(optimizer, iteration_count):
    lr = args.lr / (1.0 + args.lr_decay * iteration_count)  # 随着迭代次数的增长，学习率会逐渐降低
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


parser = argparse.ArgumentParser()
# 基础指令
parser.add_argument('--content_dir', type=str, default='../data/content', help='内容图片目录')
parser.add_argument('--style_dir', type=str, default='../data/style', help='风格图片目录')
parser.add_argument('--vgg', type=str, default='models/vgg_normalised.pth')

# 训练指令
parser.add_argument('--save_dir', default='./experiments', help='模型保存目录')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--lr_decay', type=float, default=5e-5)
parser.add_argument('--max_iter', type=int, default=160000)
parser.add_argument('--batch_size', type=int, default=5)
parser.add_argument('--style_weight', type=float, default=3.0)
parser.add_argument('--content_weight', type=float, default=1.0)
parser.add_argument('--n_threads', type=int, default=16)
parser.add_argument('--save_model_interval', type=int, default=1000)
parser.add_argument('--start_iter', type=float, default=0)

args = parser.parse_args('')

device = torch.device('cuda')

decoder = decoder

vgg = vgg  # VGG19的网络结构
vgg.load_state_dict(torch.load(args.vgg))  # 从权重文件导入权重
vgg = nn.Sequential(*list(vgg.children())[:44])  # 取前44层
network = Net(vgg, decoder, args.start_iter)  # 初始化网络
network.train()  # 训练网络
network.to(device)

# 图像转换
content_tf = train_transform()
style_tf = train_transform()

# 图像数据集
content_dataset = FlatFolderDataset(args.content_dir, content_tf)
style_dataset = FlatFolderDataset(args.style_dir, style_tf)

# 构建数据集的迭代器
content_iter = iter(data.DataLoader(content_dataset,
                                    batch_size=args.batch_size,
                                    sampler=InfiniteSamplerWrapper(content_dataset),
                                    num_workers=args.n_threads))
style_iter = iter(data.DataLoader(style_dataset,
                                  batch_size=args.batch_size,
                                  sampler=InfiniteSamplerWrapper(style_dataset),
                                  num_workers=args.n_threads))

# 构建优化器，对解码器和转换器的参数进行优化
optimizer = torch.optim.Adam([{'params': network.decoder.parameters()}, {'params': network.transform.parameters()}],
                             lr=args.lr)

# 如果不是从头开始训练，则需要导入优化器权重
if args.start_iter > 0:
    optimizer.load_state_dict(torch.load('optimizer_iter_' + str(args.start_iter) + '.pth'))

for i in tqdm(range(args.start_iter, args.max_iter)):
    adjust_learning_rate(optimizer, iteration_count=i)  # 首先调整当前的学习率
    # 取下一批图片
    content_images = next(content_iter).to(device)
    style_images = next(style_iter).to(device)
    # 计算损失
    loss_c, loss_s, l_identity1, l_identity2 = network(content_images, style_images)
    loss_c = args.content_weight * loss_c
    loss_s = args.style_weight * loss_s
    loss = loss_c + loss_s + l_identity1 * 50 + l_identity2 * 1  # 计算加权后的损失

    # 梯度下降
    optimizer.zero_grad()  # 梯度归零
    loss.backward()  # 反向传播求梯度
    optimizer.step()  # 反向传播更新参数

    # 隔一段迭代保存模型
    if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:
        # 保存解码器模型
        state_dict = decoder.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict, '{:s}/decoder_iter_{:d}.pth'.format(args.save_dir, i + 1))
        # 保存转换器模型
        state_dict = network.transform.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict, '{:s}/transformer_iter_{:d}.pth'.format(args.save_dir, i + 1))
        # 保存优化器模型
        state_dict = optimizer.state_dict()
        torch.save(state_dict, '{:s}/optimizer_iter_{:d}.pth'.format(args.save_dir, i + 1))
