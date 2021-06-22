import torch


# 计算均值和标准差
def calc_mean_std(feat, eps=1e-5):
    # eps 是添加到方差的一个小值，以避免被零除
    size = feat.size()
    assert (len(size) == 4)
    n, c = size[:2]
    feat_var = feat.view(n, c, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(n, c, 1, 1)
    feat_mean = feat.view(n, c, -1).mean(dim=2).view(n, c, 1, 1)
    return feat_mean, feat_std


def mean_variance_norm(feat):
    size = feat.size()
    mean, std = calc_mean_std(feat)
    normalized_feat = (feat - mean.expand(size)) / std.expand(size)
    return normalized_feat


def calc_feat_flatten_mean_std(feat):
    assert (feat.size()[0] == 3)  # 采用 3D 特征：通道、高、宽，返回通道内数组的均值和标准差
    assert (isinstance(feat, torch.FloatTensor))
    feat_flatten = feat.view(3, -1)
    mean = feat_flatten.mean(dim=-1, keepdim=True)
    std = feat_flatten.std(dim=-1, keepdim=True)
    return feat_flatten, mean, std
