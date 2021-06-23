# Style Transfer 风格迁移

## 数据集

- 使用了 MS COCO 和 WikiArt 数据集的部分，前者对应内容图片，后者对应风格图片
- 可以直接去官网下载，也可以下载本实验使用的数据
  - coco2017
    - 链接：https://pan.baidu.com/s/11gZi6nrpIltCRkK28sv3ew 
    - 提取码：cis7
  - 本实验
    - 链接：https://pan.baidu.com/s/11Jy6fO9VziFkg3KBM4eDYg 
    - 提取码：4x0x
- 下载后解压到 data 文件夹下（即 data 下包含 content 和 style 两个文件夹，分别存放训练用的内容图片和风格图片）

## AdaIN

- 实现参考：https://github.com/naoto0804/pytorch-AdaIN

- 风格迁移

  - 进入 AdaIN 目录
  - 根据需要下载预训练模型到 models 文件夹
    - 链接：https://pan.baidu.com/s/1pRSSSd0cf-G-FS0QqsJMIg 
    - 提取码：djun
  - 运行 eval.py 文件，可以命令行指定输入图像路径（--content）和风格图像路径（--style），也可以修改代码中的默认值
  - 输出图片默认在 output 文件夹

- 训练
    - 进入 AdaIN 目录
    - 按要求准备好数据集
    - 运行 train.py 文件，需要 GPU 环境

## SANet

- 实现参考：https://github.com/GlebBrykin/SANET
- 使用方法同上
  - 预训练模型
    - 链接：https://pan.baidu.com/s/116tPY4zNS4AzLRiPQen8Ew 
    - 提取码：ae01