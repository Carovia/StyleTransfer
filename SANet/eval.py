import argparse
import os
import torch
import torch.nn as nn
from PIL import Image
from os.path import basename
from os.path import splitext
from torchvision import transforms
from torchvision.utils import save_image

from net import decoder, vgg, Transform


def test_transform():
    transform_list = [transforms.ToTensor()]
    transform = transforms.Compose(transform_list)
    return transform


parser = argparse.ArgumentParser()

# 基础指令
parser.add_argument('--content', type=str, default='input/avril.jpg', help='内容图片路径')
parser.add_argument('--style', type=str, default='style/mess.jpg', help='风格图片路径')
parser.add_argument('--steps', type=str, default=1)
parser.add_argument('--vgg', type=str, default='models/vgg_normalised.pth')
parser.add_argument('--decoder', type=str, default='models/decoder_iter_500000.pth')
parser.add_argument('--transform', type=str, default='models/transformer_iter_500000.pth')

# 附加指令
parser.add_argument('--save_ext', default='.jpg', help='输出图片名')
parser.add_argument('--output', type=str, default='output', help='输出目录')

args = parser.parse_args('')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.exists(args.output):
    os.mkdir(args.output)

decoder = decoder
transform = Transform(in_planes=512)
vgg = vgg

decoder.eval()
transform.eval()
vgg.eval()

decoder.load_state_dict(torch.load(args.decoder))
transform.load_state_dict(torch.load(args.transform))
vgg.load_state_dict(torch.load(args.vgg))

norm = nn.Sequential(*list(vgg.children())[:1])
enc_1 = nn.Sequential(*list(vgg.children())[:4])  # input -> relu1_1
enc_2 = nn.Sequential(*list(vgg.children())[4:11])  # relu1_1 -> relu2_1
enc_3 = nn.Sequential(*list(vgg.children())[11:18])  # relu2_1 -> relu3_1
enc_4 = nn.Sequential(*list(vgg.children())[18:31])  # relu3_1 -> relu4_1
enc_5 = nn.Sequential(*list(vgg.children())[31:44])  # relu4_1 -> relu5_1

norm.to(device)
enc_1.to(device)
enc_2.to(device)
enc_3.to(device)
enc_4.to(device)
enc_5.to(device)
transform.to(device)
decoder.to(device)

content_tf = test_transform()
style_tf = test_transform()

content = content_tf(Image.open(args.content))
style = style_tf(Image.open(args.style))

style = style.to(device).unsqueeze(0)
content = content.to(device).unsqueeze(0)

with torch.no_grad():
    for x in range(args.steps):
        print('iteration ' + str(x))
        
        Content4_1 = enc_4(enc_3(enc_2(enc_1(content))))
        Content5_1 = enc_5(Content4_1)
    
        Style4_1 = enc_4(enc_3(enc_2(enc_1(style))))
        Style5_1 = enc_5(Style4_1)
    
        content = decoder(transform(Content4_1, Style4_1, Content5_1, Style5_1))

        content.clamp(0, 255)

    content = content.cpu()
    
    output_name = '{:s}/{:s}_stylized_{:s}{:s}'.format(
                args.output, splitext(basename(args.content))[0],
                splitext(basename(args.style))[0], args.save_ext
            )
    save_image(content, output_name)
