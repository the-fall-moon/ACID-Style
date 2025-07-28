"""Stripped version of https://github.com/richzhang/PerceptualSimilarity/tree/master/models"""

from annotated_types import T
import torch
import torch.nn as nn
from torchvision import models
from collections import namedtuple
import math
# from ldm.util import get_ckpt_path

def get_ckpt_path(name, root, check=False):
    assert name in URL_MAP
    path = os.path.join(root, CKPT_MAP[name])
    if not os.path.exists(path) or (check and not md5_hash(path) == MD5_MAP[name]):
        print("Downloading {} model from {} to {}".format(name, URL_MAP[name], path))
        download(URL_MAP[name], path)
        md5 = md5_hash(path)
        assert md5 == MD5_MAP[name], md5
    return path


class LPIPS(nn.Module):
    # Learned perceptual metric
    def __init__(self, use_dropout=True):
        super().__init__()
        self.scaling_layer = ScalingLayer()
        self.chns = [64, 128, 256, 512, 512]  # vg16 features
        self.net = vgg16(pretrained=True, requires_grad=False)
        self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
        self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
        self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
        self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
        self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
        self.load_from_pretrained()
        for param in self.parameters():
            param.requires_grad = False

    def load_from_pretrained(self, name="vgg_lpips"):
        ckpt = get_ckpt_path(name, "taming/modules/autoencoder/lpips")
        self.load_state_dict(torch.load(ckpt, map_location=torch.device("cpu")), strict=False)
        print("loaded pretrained LPIPS loss from {}".format(ckpt))

    @classmethod
    def from_pretrained(cls, name="vgg_lpips"):
        if name != "vgg_lpips":
            raise NotImplementedError
        model = cls()
        ckpt = get_ckpt_path(name)
        model.load_state_dict(torch.load(ckpt, map_location=torch.device("cpu")), strict=False)
        return model

    def forward(self, input, target):
        in0_input, in1_input = (self.scaling_layer(input), self.scaling_layer(target))
        outs0, outs1 = self.net(in0_input), self.net(in1_input)
        feats0, feats1, diffs = {}, {}, {}
        lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        for kk in range(len(self.chns)):
            feats0[kk], feats1[kk] = normalize_tensor(outs0[kk]), normalize_tensor(outs1[kk])
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2

        res = [spatial_average(lins[kk].model(diffs[kk]), keepdim=True) for kk in range(len(self.chns))]
        val = res[0]
        for l in range(1, len(self.chns)):
            val += res[l]
        return val


class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer('shift', torch.Tensor([-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer('scale', torch.Tensor([.458, .448, .450])[None, :, None, None])

    def forward(self, inp):
        return (inp - self.shift) / self.scale
    
class ScalingLayer_v2(nn.Module):
    def __init__(self):
        super(ScalingLayer_v2, self).__init__()
        self.register_buffer('shift', torch.Tensor([0.485, 0.456, 0.406])[None, :, None, None])
        self.register_buffer('scale', torch.Tensor([.229, .224, .225])[None, :, None, None])

    def forward(self, inp):
        return (inp - self.shift) / self.scale


class NetLinLayer(nn.Module):
    """ A single linear layer which does a 1x1 conv """
    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()
        layers = [nn.Dropout(), ] if (use_dropout) else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False), ]
        self.model = nn.Sequential(*layers)


class vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super(vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)
        return out


def normalize_tensor(x,eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(x**2,dim=1,keepdim=True))
    return x/(norm_factor+eps)


def spatial_average(x, keepdim=True):
    return x.mean([2,3],keepdim=keepdim)


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def normalize(feat, eps=1e-5):
    feat_mean, feat_std= calc_mean_std(feat, eps)
    normalized=(feat-feat_mean)/feat_std
    return normalized 

def build_vgg():
    vgg = nn.Sequential(
        nn.Conv2d(3, 3, (1, 1)),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(3, 64, (3, 3)),
        nn.ReLU(),  # relu1-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 64, (3, 3)),
        nn.ReLU(),  # relu1-2
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 128, (3, 3)),
        nn.ReLU(),  # relu2-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 128, (3, 3)),
        nn.ReLU(),  # relu2-2
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 256, (3, 3)),
        nn.ReLU(),  # relu3-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),  # relu3-2
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),  # relu3-3
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),  # relu3-4
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 512, (3, 3)),
        nn.ReLU(),  # relu4-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),  # relu4-2
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),  # relu4-3
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),  # relu4-4
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),  # relu5-1, this is the last layer used
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),  # relu5-2
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),  # relu5-3
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU()  # relu5-4
    )
    return vgg


class VGG19(nn.Module):
    def __init__(self, checkpoint):
        super(VGG19, self).__init__()
        vgg = build_vgg()
        vgg.load_state_dict(torch.load(checkpoint))
        vgg_layers = list(vgg.children())
        self.enc_1 = nn.Sequential(*vgg_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*vgg_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*vgg_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*vgg_layers[18:31])  # relu3_1 -> relu4_1
        self.enc_5 = nn.Sequential(*vgg_layers[31:44])  # relu4_1 -> relu5_1

        # fix the encoder
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4', 'enc_5']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

        self.mse_loss = nn.MSELoss()

    # extract relu1_1 - relu(n_layer)_1 from input image
    def encode_with_intermediate(self, x, n_layer=5):
        results = [x]
        for i in range(n_layer):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    # extract relu(n_layer)_1 from input image
    def encode(self, x, n_layer=5):
        for i in range(n_layer):
            x = getattr(self, 'enc_{:d}'.format(i + 1))(x)
        return x

    # def reduce_high_attention_regions_v1(self, features, threshold=0.8):
    #     features_flatten = features.view(features.size(0),-1)
    #     cdf = torch.sort(features_flatten, descending=True)[0].cumsum(dim=1)
    #     cdf = cdf / cdf[:, -1:].expand_as(cdf)
    #     indices = cdf > threshold
    #     # 查询indcices中 值为True的数量
    #     indices_num = torch.sum(indices, dim=1)
    #     print(indices_num)
    #     features_flatten = torch.where(indices, torch.full_like(features_flatten, threshold), features_flatten)
    #     return features_flatten.view(features.size())
    
    def reduce_high_attention_regions(self, features, threshold=1):
        if threshold >= 1:
            return features
        features_flatten = features.view(features.size(0),features.size(1),-1)
        cdf = torch.sort(features_flatten, descending=False)[0]
        threshold_num = int(cdf.size(-1)*threshold)
        threshold_maps = cdf[:,:,threshold_num:threshold_num+1].expand_as(cdf)
        features_flatten = torch.where(features_flatten > threshold_maps, threshold_maps, features_flatten)
        return features_flatten.view(features.size())

    def calc_content_loss(self, input, target,normal=True):
        assert (input.size() == target.size())
        if normal == False:
            return self.mse_loss(input, target)
        else:
            return self.mse_loss(normalize(input),normalize(target))

    def calc_style_loss(self, input, target):
        assert (input.size() == target.size())
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + self.mse_loss(input_std, target_std)

    def forward(self, content_images, style_images, stylized_images, 
                content_layer=5, style_layer=5, wo_content=False, wo_style=False, content_threshold:float=None):
        
        max_layer = content_layer if content_layer > style_layer else style_layer
        
        stylized_feats = self.encode_with_intermediate(stylized_images, max_layer)
        #print("style_feats",style_feats)

        # content loss
        loss_c = 0
        if not wo_content:
            content_feat = self.encode_with_intermediate(content_images, content_layer) # self.encode(content_images,n_layer)
            for i in range(3,content_layer):
                if content_threshold is not None:
                    content_feat[i] = self.reduce_high_attention_regions(content_feat[i],threshold=content_threshold)
                loss_c += self.calc_content_loss(stylized_feats[i], content_feat[i])    # relu4_1 relu5_1

        # style loss
        loss_s = 0
        if not wo_style:
            style_feats = self.encode_with_intermediate(style_images, style_layer)
            for i in range(0, style_layer):
                loss_s += self.calc_style_loss(stylized_feats[i], style_feats[i])
        
        return loss_c, loss_s

    def gram_matric(self,x, normalize=True):
        '''
        Generate gram matrices of the representations of content and style images.
        '''
        (b, ch, h, w) = x.size()
        features = x.view(b, ch, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t)
        if normalize:
            gram /= ch * h * w
        return gram
    
    def calc_style_loss_gram(self, input, target):
        assert (input.size() == target.size())
        # assert (target.requires_grad is False)
        return self.mse_loss(self.gram_matric(input),self.gram_matric(target))
    
    def forward_gram(self, style_images, stylized_images, n_layer=5):
       
        style_feats = self.encode_with_intermediate(style_images, n_layer)
        stylized_feats = self.encode_with_intermediate(stylized_images, n_layer)
        #print("style_feats",style_feats)

        # style gram loss
        loss_style_gram = self.calc_style_loss_gram(stylized_feats[0], style_feats[0])
        for i in range(1, n_layer):
            loss_style_gram += self.calc_style_loss_gram(stylized_feats[i], style_feats[i])
        
        return loss_style_gram


if __name__ == '__main__':
   
    pass