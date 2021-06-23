import argparse
from pathlib import Path
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

from net import decoder, vgg
from function import adaptive_instance_normalization


# 转换图像，可以指定大小和是否进行中心裁剪
def test_transform(size, crop):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform


# 风格转换器
def style_transfer(vgg, decoder, content, style, alpha=1.0, interpolation_weights=None):
    assert (0.0 <= alpha <= 1.0)
    content_f = vgg(content)  # 提取内容图片特征
    style_f = vgg(style)  # 提取风格图片特征
    if interpolation_weights:
        _, C, H, W = content_f.size()
        feat = torch.FloatTensor(1, C, H, W).zero_().to(device)
        base_feat = adaptive_instance_normalization(content_f, style_f)
        for i, w in enumerate(interpolation_weights):
            feat = feat + w * base_feat[i:i + 1]  # 对不同的风格加权求和
        content_f = content_f[0:1]
    else:
        feat = adaptive_instance_normalization(content_f, style_f)
    feat = feat * alpha + content_f * (1 - alpha)  # 使用 AdaIN 算法求迁移后的特征
    return decoder(feat)  # 使用解码器返回生成的图片


parser = argparse.ArgumentParser()

# 基础指令
parser.add_argument('--content', type=str, default='input/avril.jpg', help='内容图片路径')
parser.add_argument('--style', type=str, default='style/candy.jpg', help='风格图片路径')
parser.add_argument('--content_dir', type=str, help='内容图片目录')
parser.add_argument('--style_dir', type=str, help='风格图片目录')
parser.add_argument('--vgg', type=str, default='models/vgg_normalised.pth')
parser.add_argument('--decoder', type=str, default='models/decoder.pth')

# 附加指令
parser.add_argument('--content_size', type=int, default=512, help='新内容图片的（最小）尺寸，设为0则不变')
parser.add_argument('--style_size', type=int, default=512, help='新风格图片的（最小）尺寸，设为0则不变')
parser.add_argument('--crop', action='store_true', help='中心裁剪生成正方形图片')
parser.add_argument('--save_ext', default='.jpg', help='输出图片名')
parser.add_argument('--output', type=str, default='output', help='输出目录')

# 进阶指令
parser.add_argument('--alpha', type=float, default=0.8, help='风格化程度的权重，介于0和1')
parser.add_argument('--style_interpolation_weights', type=str, default='', help='融合多张风格图片的权重')

args = parser.parse_args('')

do_interpolation = False  # 默认不做多风格融合

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

output_dir = Path(args.output)
output_dir.mkdir(exist_ok=True, parents=True)

assert (args.content or args.content_dir)
if args.content:
    content_paths = [Path(args.content)]
else:
    content_dir = Path(args.content_dir)
    content_paths = [f for f in content_dir.glob('*')]

assert (args.style or args.style_dir)
if args.style:
    style_paths = args.style.split(',')
    if len(style_paths) == 1:
        style_paths = [Path(args.style)]
    else:
        do_interpolation = True
        assert (args.style_interpolation_weights != ''), 'Please specify interpolation weights'
        weights = [int(i) for i in args.style_interpolation_weights.split(',')]
        interpolation_weights = [w / sum(weights) for w in weights]  # 对权值做标准化处理
else:
    style_dir = Path(args.style_dir)
    style_paths = [f for f in style_dir.glob('*')]

decoder = decoder
vgg = vgg

decoder.eval()
vgg.eval()

# 从文件加载权重
decoder.load_state_dict(torch.load(args.decoder))  # 加载解码器权值
vgg.load_state_dict(torch.load(args.vgg))  # 加载 VGG 权值
vgg = nn.Sequential(*list(vgg.children())[:31])  # 取 VGG 前31层

vgg.to(device)
decoder.to(device)

# 图像预处理
content_tf = test_transform(args.content_size, args.crop)
style_tf = test_transform(args.style_size, args.crop)

for content_path in content_paths:
    # 多风格融合
    if do_interpolation:
        # 图像转换
        style = torch.stack([style_tf(Image.open(str(p))) for p in style_paths])
        content = content_tf(Image.open(str(content_path))).unsqueeze(0).expand_as(style)
        style = style.to(device)
        content = content.to(device)
        # 风格迁移
        with torch.no_grad():
            output = style_transfer(vgg, decoder, content, style, args.alpha, interpolation_weights)
        output = output.cpu()
        output_name = output_dir / '{:s}_interpolation{:s}'.format(content_path.stem, args.save_ext)
        # 保存图片
        save_image(output, str(output_name))
    # 单风格迁移
    else:
        for style_path in style_paths:
            # 图像转换
            content = content_tf(Image.open(str(content_path)))
            style = style_tf(Image.open(str(style_path)))
            # 增加维度
            content = content.to(device).unsqueeze(0)
            style = style.to(device).unsqueeze(0)
            # 风格迁移
            with torch.no_grad():
                output = style_transfer(vgg, decoder, content, style, args.alpha)
            output = output.cpu()
            output_name = output_dir / '{:s}_stylized_{:s}{:s}'.format(content_path.stem, style_path.stem, args.save_ext)
            # 保存图片
            save_image(output, str(output_name))
