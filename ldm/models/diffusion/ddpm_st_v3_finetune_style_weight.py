"""
wild mixture of
https://github.com/lucidrains/denoising-diffusion-pytorch/blob/7706bdfc6f527f58d33f84b7b522e61e6e3164b3/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
https://github.com/openai/improved-diffusion/blob/e94489283bb876ac1477d5dd7709bbbd2d9902ce/improved_diffusion/gaussian_diffusion.py
https://github.com/CompVis/taming-transformers
-- merci
"""

from re import S
import re
import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from torch.optim.lr_scheduler import LambdaLR
from einops import rearrange, repeat
from contextlib import contextmanager
from functools import partial
from tqdm import tqdm
from torchvision.utils import make_grid
# from pytorch_lightning.utilities.distributed import rank_zero_only
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from ldm.models.diffusion.dpm_solver.sampler import DPMSolverSampler
from ldm.models.diffusion.dpm_solver.dpm_solver import model_wrapper, DPM_Solver
from ldm.models.diffusion.plms import PLMSSampler
from ldm.modules.losses.lpips import vgg16, ScalingLayer,VGG19
from ldm.util import exists, default, mean_flat, count_params, instantiate_from_config
from ldm.modules.ema import LitEma
from ldm.modules.distributions.distributions import normal_kl, DiagonalGaussianDistribution
from ldm.models.autoencoder import VQModelInterface, IdentityFirstStage, AutoencoderKL
from ldm.modules.diffusionmodules.util import make_beta_schedule, extract_into_tensor, noise_like
from ldm.models.diffusion.ddim import DDIMSampler
import cv2
from model import DexiNed
from torchvision.utils import save_image
import random
import torch.nn.functional as F
import time
import copy
import os
import cv2
import matplotlib.pyplot as plt
from ldm.modules.losses import lpips
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

__conditioning_keys__ = {'concat': 'c_concat',
                         'crossattn': 'c_crossattn',
                         'adm': 'y'}
def get_attr_recursive(obj,attr):
    attrs = attr.split(".")
    for i in attrs:
        if isinstance(obj,dict):
            if i not in obj:
                return None
            obj = obj[i]
        else:
            try:
                obj = getattr(obj,i)
            except AttributeError:
                return None
    return obj

def torch2img(input):
    input_ = input[0]
    input_ = input_.permute(1,2,0)
    input_ = input_.data.cpu().numpy()
    input_ = (input_ + 1.0) / 2
    cv2.imwrite('./test.png', input_[:,:,::-1]*255.0)


def visualize_fea(save_path, fea_img):
    fig = plt.figure(figsize = (fea_img.shape[1]/10, fea_img.shape[0]/10)) # Your image (W)idth and (H)eight in inches
    plt.subplots_adjust(left = 0, right = 1.0, top = 1.0, bottom = 0)
    im = plt.imshow(fea_img, vmin=0.0, vmax=1.0, cmap='jet', aspect='auto') # Show the image
    plt.savefig(save_path)
    plt.clf()

def calc_mean_std(feat, eps=1e-5):
    """Calculate mean and std for adaptive_instance_normalization.
    Args:
        feat (Tensor): 4D tensor.
        eps (float): A small value added to the variance to avoid
            divide-by-zero. Default: 1e-5.
    """
    size = feat.size()
    assert len(size) == 4, 'The input feature should be 4D tensor.'
    b, c = size[:2]
    feat_var = feat.view(b, c, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(b, c, 1, 1)
    feat_mean = feat.view(b, c, -1).mean(dim=2).view(b, c, 1, 1)
    return feat_mean, feat_std

def adaptive_instance_normalization(content_feat, style_feat):
    """Adaptive instance normalization.
    Adjust the reference features to have the similar color and illuminations
    as those in the degradate features.
    Args:
        content_feat (Tensor): The reference feature.
        style_feat (Tensor): The degradate features.
    """
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)

def space_timesteps(num_timesteps, section_counts):
    """
    Create a list of timesteps to use from an original diffusion process,
    given the number of timesteps we want to take from equally-sized portions
    of the original process.

    For example, if there's 300 timesteps and the section counts are [10,15,20]
    then the first 100 timesteps are strided to be 10 timesteps, the second 100
    are strided to be 15 timesteps, and the final 100 are strided to be 20.

    If the stride is a string starting with "ddim", then the fixed striding
    from the DDIM paper is used, and only one section is allowed.

    :param num_timesteps: the number of diffusion steps in the original
                          process to divide up.
    :param section_counts: either a list of numbers, or a string containing
                           comma-separated numbers, indicating the step count
                           per section. As a special case, use "ddimN" where N
                           is a number of steps to use the striding from the
                           DDIM paper.
    :return: a set of diffusion steps from the original process to use.
    """
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim"):])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(
                f"cannot create exactly {num_timesteps} steps with an integer stride"
            )
        section_counts = [int(x) for x in section_counts.split(",")]   #[250,]
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

def uniform_on_device(r1, r2, shape, device):
    return (r1 - r2) * torch.rand(*shape, device=device) + r2

class DDPM(pl.LightningModule):
    # classic DDPM with Gaussian diffusion, in image space
    def __init__(self,
                 unet_config,
                 timesteps=1000,
                 beta_schedule="linear",
                 loss_type="l2",
                 ckpt_path=None,
                 ignore_keys=[],
                 load_only_unet=False,
                 monitor="train_loss",
                 use_ema=True,
                 first_stage_key="image",
                 image_size=256,
                 channels=3,
                 log_every_t=100,
                 clip_denoised=True,
                 linear_start=1e-4,
                 linear_end=2e-2,
                 cosine_s=8e-3,
                 given_betas=None,
                 original_elbo_weight=0.,
                 v_posterior=0.,  # weight for choosing posterior variance as sigma = (1-v) * beta_tilde + v * beta
                 l_simple_weight=1.,
                 conditioning_key=None,
                 parameterization="eps",  # all assuming fixed variance schedules
                 scheduler_config=None,
                 use_positional_encodings=False,
                 learn_logvar=False,
                 logvar_init=0.,
                 ):
        super().__init__()
        assert parameterization in ["eps", "x0", "v"], 'currently only supporting "eps" and "x0" and "v"'
        self.parameterization = parameterization
        print(f"{self.__class__.__name__}: Running in {self.parameterization}-prediction mode")
        self.cond_stage_model = None
        self.clip_denoised = clip_denoised
        self.log_every_t = log_every_t
        self.first_stage_key = first_stage_key
        self.image_size = image_size  # try conv?
        self.channels = channels
        self.use_positional_encodings = use_positional_encodings
        self.model = DiffusionWrapper(unet_config, conditioning_key)
        count_params(self.model, verbose=True)
        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self.model)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        self.use_scheduler = scheduler_config is not None
        if self.use_scheduler:
            self.scheduler_config = scheduler_config

        self.v_posterior = v_posterior
        self.original_elbo_weight = original_elbo_weight
        self.l_simple_weight = l_simple_weight

        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys, only_model=load_only_unet)

        self.register_schedule(given_betas=given_betas, beta_schedule=beta_schedule, timesteps=timesteps,
                               linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s)

        self.loss_type = loss_type

        self.learn_logvar = learn_logvar
        self.logvar = torch.full(fill_value=logvar_init, size=(self.num_timesteps,))
        if self.learn_logvar:
            self.logvar = nn.Parameter(self.logvar, requires_grad=True)


    def register_schedule(self, given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        if exists(given_betas):
            betas = given_betas
        else:
            betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end,
                                       cosine_s=cosine_s)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (
                    1. - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

        if self.parameterization == "eps":
            lvlb_weights = self.betas ** 2 / (
                        2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod))
        elif self.parameterization == "x0":
            lvlb_weights = 0.5 * np.sqrt(torch.Tensor(alphas_cumprod)) / (2. * 1 - torch.Tensor(alphas_cumprod))
        elif self.parameterization == "v":
            lvlb_weights = torch.ones_like(self.betas ** 2 / (
                    2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod)))
        else:
            raise NotImplementedError("mu not supported")
        # TODO how to choose this term
        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer('lvlb_weights', lvlb_weights, persistent=False)
        assert not torch.isnan(self.lvlb_weights).all()

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def init_from_ckpt(self, path, ignore_keys=list(), only_model=False):
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False) if not only_model else self.model.load_state_dict(sd, strict=False)
        print('<<<<<<<<<<<<>>>>>>>>>>>>>>>')
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).
        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start)
        variance = extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract_into_tensor(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool):
        model_out = self.model(x, t)
        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        elif self.parameterization == "v":
            x_recon = self.predict_start_from_z_and_v(x, model_out, t)
        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, repeat_noise=False):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, clip_denoised=clip_denoised)
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, return_intermediates=False):
        device = self.betas.device
        b = shape[0]
        img = torch.randn(shape, device=device)
        intermediates = [img]
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='Sampling t', total=self.num_timesteps):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long),
                                clip_denoised=self.clip_denoised)
            if i % self.log_every_t == 0 or i == self.num_timesteps - 1:
                intermediates.append(img)
        if return_intermediates:
            return img, intermediates
        return img

    @torch.no_grad()
    def sample(self, batch_size=16, return_intermediates=False):
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, image_size, image_size),
                                  return_intermediates=return_intermediates)

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    def q_sample_respace(self, x_start, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (extract_into_tensor(sqrt_alphas_cumprod.to(noise.device), t, x_start.shape) * x_start +
                extract_into_tensor(sqrt_one_minus_alphas_cumprod.to(noise.device), t, x_start.shape) * noise)

    def get_v(self, x, noise, t):
        return (
                extract_into_tensor(self.sqrt_alphas_cumprod, t, x.shape) * noise -
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x.shape) * x
        )

    def predict_start_from_z_and_v(self, x, v, t):
        return (
                extract_into_tensor(self.sqrt_alphas_cumprod, t, x.shape) * x -
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x.shape) * v
        )

    def get_loss(self, pred, target, mean=True):
        if self.loss_type == 'l1':
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == 'l2':
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target, pred, reduction='none')
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        return loss

    def p_losses(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_out = self.model(x_noisy, t)

        loss_dict = {}
        if self.parameterization == "eps":
            target = noise
        elif self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "v":
            target = self.get_v(x_start, noise, t)
        else:
            raise NotImplementedError(f"Paramterization {self.parameterization} not yet supported")

        loss = self.get_loss(model_out, target, mean=False).mean(dim=[1, 2, 3])

        log_prefix = 'train' if self.training else 'val'

        loss_dict.update({f'{log_prefix}/loss_simple': loss.mean()})
        loss_simple = loss.mean() * self.l_simple_weight

        loss_vlb = (self.lvlb_weights[t] * loss).mean()
        loss_dict.update({f'{log_prefix}/loss_vlb': loss_vlb})

        loss = loss_simple + self.original_elbo_weight * loss_vlb

        loss_dict.update({f'{log_prefix}/loss': loss})

        return loss, loss_dict

    def forward(self, x, *args, **kwargs):
        # b, c, h, w, device, img_size, = *x.shape, x.device, self.image_size
        # assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        return self.p_losses(x, t, *args, **kwargs)

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = rearrange(x, 'b h w c -> b c h w')
        x = x.to(memory_format=torch.contiguous_format).float()
        return x

    def shared_step(self, batch):
        x = self.get_input(batch, self.first_stage_key)
        loss, loss_dict = self(x)
        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch)
        self.log('train_loss', loss,
                 on_epoch=True, prog_bar=True, logger=True)
        self.log_dict(loss_dict, prog_bar=True,
                      logger=True, on_step=True, on_epoch=True)

        self.log("global_step", self.global_step,
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)

        if self.use_scheduler:
            lr = self.optimizers().param_groups[0]['lr']
            self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        _, loss_dict_no_ema = self.shared_step(batch)
        with self.ema_scope():
            _, loss_dict_ema = self.shared_step(batch)
            loss_dict_ema = {key + '_ema': loss_dict_ema[key] for key in loss_dict_ema}
        self.log_dict(loss_dict_no_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log_dict(loss_dict_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self.model)

    def _get_rows_from_list(self, samples):
        n_imgs_per_row = len(samples)
        denoise_grid = rearrange(samples, 'n b c h w -> b n c h w')
        denoise_grid = rearrange(denoise_grid, 'b n c h w -> (b n) c h w')
        denoise_grid = make_grid(denoise_grid, nrow=n_imgs_per_row)
        return denoise_grid

    @torch.no_grad()
    def log_images(self, batch, N=8, n_row=2, sample=True, return_keys=None, **kwargs):
        log = dict()
        x = self.get_input(batch, self.first_stage_key)
        N = min(x.shape[0], N)
        n_row = min(x.shape[0], n_row)
        x = x.to(self.device)[:N]
        log["inputs"] = x

        # get diffusion row
        diffusion_row = list()
        x_start = x[:n_row]

        for t in range(self.num_timesteps):
            if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                t = t.to(self.device).long()
                noise = torch.randn_like(x_start)
                x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
                diffusion_row.append(x_noisy)

        log["diffusion_row"] = self._get_rows_from_list(diffusion_row)

        if sample:
            # get denoise row
            with self.ema_scope("Plotting"):
                samples, denoise_row = self.sample(batch_size=N, return_intermediates=True)

            log["samples"] = samples
            log["denoise_row"] = self._get_rows_from_list(denoise_row)

        if return_keys:
            if np.intersect1d(list(log.keys()), return_keys).shape[0] == 0:
                return log
            else:
                return {key: log[key] for key in return_keys}
        return log

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())
        if self.learn_logvar:
            params = params + [self.logvar]
        opt = torch.optim.AdamW(params, lr=lr)
        return opt



class LatentDiffusiontransferV2(DDPM):
    """main class"""
    def __init__(self,
                 first_stage_config,
                 cond_stage_config,
                 structcond_stage_config,
                #  style_flag_key='style_flag',
                #  content_flag_key='content_flag',
                 num_timesteps_cond=None,
                 cond_stage_key="image",
                 cond_stage_trainable=False, 
                 concat_mode=True,
                 cond_stage_forward=None,
                 conditioning_key=None,
                 vgg_encoder_ckpt_path=None, # VGG19
                 scale_factor=1.0,
                 scale_by_std=False,
                 unfrozen_diff=False,
                 random_size=False,
                 p2_gamma=None,
                 p2_k=None,
                 time_replace=None,
                 sampler=None,
                 total_ddim_steps=None,
                 use_usm=False,
                 mix_ratio=0.0,
                 *args, **kwargs):
        # put this in your init
        self.num_timesteps_cond = default(num_timesteps_cond, 1)
        self.scale_by_std = scale_by_std
        self.unfrozen_diff = unfrozen_diff
        self.random_size = random_size
        self.time_replace = time_replace
        self.use_usm = use_usm

        self.sampler=sampler
        self.total_ddim_steps=total_ddim_steps
        
        self.mix_ratio = mix_ratio
        assert self.num_timesteps_cond <= kwargs['timesteps']
        # for backwards compatibility after implementation of DiffusionWrapper
        if conditioning_key is None:
            conditioning_key = 'concat' if concat_mode else 'crossattn'
        if cond_stage_config == '__is_unconditional__':
            conditioning_key = None
        ckpt_path = kwargs.pop("ckpt_path", None)
        ignore_keys = kwargs.pop("ignore_keys", [])
        resume_ckpt_path = kwargs.pop("resume_ckpt_path", None)
        self.content_threshold = kwargs.pop("content_threshold", 1)
        self.style_weight_ = kwargs.pop("style_weight", None)
        super().__init__(conditioning_key=conditioning_key, *args, **kwargs)
        self.vgg = vgg16(pretrained=True, requires_grad=False)
        self.vgg_scaling_layer = ScalingLayer() 
        self.struct_stage_trainable=True
        self.concat_mode = concat_mode
        self.cond_stage_trainable = cond_stage_trainable
        self.cond_stage_key = cond_stage_key
        try:
            self.num_downs = len(first_stage_config.params.ddconfig.ch_mult) - 1
        except:
            self.num_downs = 0
        if not scale_by_std:
            self.scale_factor = scale_factor
        else:
            self.register_buffer('scale_factor', torch.tensor(scale_factor))
        self.instantiate_first_stage(first_stage_config)
        self.instantiate_cond_stage(cond_stage_config)
        self.instantiate_structcond_stage(structcond_stage_config)
        self.instantiate_vgg_encoder(vgg_encoder_ckpt_path)
    
        self.cond_stage_forward = cond_stage_forward
        self.clip_denoised = False
        
        self.bbox_tokenizer = None
        self.restarted_from_ckpt = False

        # init the style-cross-attention
        if ckpt_path is not None:
            sd = torch.load(ckpt_path, map_location="cpu")
            if "state_dict" in list(sd.keys()):
                sd = sd["state_dict"]
            for key,value in sd.items():
                if ".attn2" in key:
                    assert "weight" in key or "bias" in key
                    if "weight" in key:
                        layer_name = key.split(".weight")[0].replace("attn2","style_attn2")
                        layer = get_attr_recursive(self,layer_name)
                        layer.weight.data.copy_(sd[key])
                    elif "bias" in key:
                        layer_name = key.split(".bias")[0].replace("attn2","style_attn2")
                        layer = get_attr_recursive(self,layer_name)
                        layer.bias.data.copy_(sd[key])

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys)
        if resume_ckpt_path is not None:
            self.init_from_ckpt(resume_ckpt_path, ignore_keys)
            self.restarted_from_ckpt = True

        if not self.unfrozen_diff:
            self.model.eval()
            # self.model.train = disabled_train
            for name, param in self.model.named_parameters():
                if 'spade' in name :
                    param.requires_grad = True
                    #param.requires_grad = False
                elif 'style_proj' in name:
                    param.requires_grad =True
                elif 'style_norm' in name:
                    param.requires_grad =True
                elif 'style_attn2.to_k' in name:
                    param.requires_grad =True
                elif 'style_attn2.to_v' in name:
                    param.requires_grad =True
                elif 'style_tokens_scale' in name:
                    param.requires_grad =True
                else:
                    param.requires_grad = False

        print('>>>>>>>>>>>>>>>>model>>>>>>>>>>>>>>>>>>>>')
        param_list = []
        for name, params in self.model.named_parameters():
            if params.requires_grad:
                param_list.append(name)
        print(param_list)
        param_list = []
        print('>>>>>>>>>>>>>>>>>cond_stage_model>>>>>>>>>>>>>>>>>>>')
        for name, params in self.cond_stage_model.named_parameters():
            if params.requires_grad:
                param_list.append(name)
        print(param_list)
        param_list = []
        print('>>>>>>>>>>>>>>>>structcond_stage_model>>>>>>>>>>>>>>>>>>>>')
        for name, params in self.structcond_stage_model.named_parameters():
            if params.requires_grad:
                param_list.append(name)
        print(param_list)

        # P2 weighting: https://github.com/jychoi118/P2-weighting
        if p2_gamma is not None:
            assert p2_k is not None
            self.p2_gamma = p2_gamma
            self.p2_k = p2_k
            self.snr = 1.0 / (1 - self.alphas_cumprod) - 1
        else:
            self.snr = None

        # Support time respacing during training
        if self.time_replace is None:
            self.time_replace = kwargs['timesteps']
        use_timesteps = set(space_timesteps(kwargs['timesteps'], [self.time_replace]))
        last_alpha_cumprod = 1.0
        new_betas = []
        timestep_map = []
        for i, alpha_cumprod in enumerate(self.alphas_cumprod):
            if i in use_timesteps:
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                timestep_map.append(i)
        new_betas = [beta.data.cpu().numpy() for beta in new_betas]
        self.register_schedule(given_betas=np.array(new_betas), timesteps=len(new_betas), linear_start=kwargs['linear_start'], linear_end=kwargs['linear_end'])
        self.ori_timesteps = list(use_timesteps)
        self.ori_timesteps.sort()

    def make_cond_schedule(self, ):
        self.cond_ids = torch.full(size=(self.num_timesteps,), fill_value=self.num_timesteps - 1, dtype=torch.long)
        ids = torch.round(torch.linspace(0, self.num_timesteps - 1, self.num_timesteps_cond)).long()
        self.cond_ids[:self.num_timesteps_cond] = ids

    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_start(self, batch, batch_idx, dataloader_idx):
        # only for very first batch
        if self.scale_by_std and self.current_epoch == 0 and self.global_step == 0 and batch_idx == 0 and not self.restarted_from_ckpt:
            assert self.scale_factor == 1., 'rather not use custom rescaling and std-rescaling simultaneously'
            # set rescale weight to 1./std of encodings
            print("### USING STD-RESCALING ###")
            x = super().get_input(batch, self.first_stage_key)
            x = x.to(self.device)
            encoder_posterior = self.encode_first_stage(x)
            z = self.get_first_stage_encoding(encoder_posterior).detach()
            del self.scale_factor
            self.register_buffer('scale_factor', 1. / z.flatten().std())
            print(f"setting self.scale_factor to {self.scale_factor}")
            print("### USING STD-RESCALING ###")

    def register_schedule(self,
                          given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        super().register_schedule(given_betas, beta_schedule, timesteps, linear_start, linear_end, cosine_s)

        self.shorten_cond_schedule = self.num_timesteps_cond > 1
        if self.shorten_cond_schedule:
            self.make_cond_schedule()

    def on_save_checkpoint(self,checkpoint):
        state_dict= checkpoint['state_dict']
        trainable_dict = {k:v for k,v in state_dict.items() if "spade" in k or "style" in k or "structcond_stage_model" in k}
        checkpoint['state_dict']= trainable_dict

    def instantiate_vgg_encoder(self, vgg_encoder_ckpt_path):
        self.vgg_encoder = lpips.VGG19(checkpoint=vgg_encoder_ckpt_path)
        self.vgg_encoder.eval()
        for param in self.vgg_encoder.parameters():
            param.requires_grad = False


    def instantiate_first_stage(self, config):
        model = instantiate_from_config(config)
        self.first_stage_model = model.eval()
        self.first_stage_model.train = disabled_train
        for param in self.first_stage_model.parameters():
            param.requires_grad = False

    def instantiate_cond_stage(self, config):
        if not self.cond_stage_trainable:
            if config == "__is_first_stage__":
                print("Using first stage also as cond stage.")
                self.cond_stage_model = self.first_stage_model
            elif config == "__is_unconditional__":
                print(f"Training {self.__class__.__name__} as an unconditional model.")
                self.cond_stage_model = None
                # self.be_unconditional = True
            else:
                model = instantiate_from_config(config)
                self.cond_stage_model = model.eval()
                # self.cond_stage_model.train = disabled_train
                for name, param in self.cond_stage_model.named_parameters():
                    if 'final_projector' not in name:
                        param.requires_grad = False
        else:
            assert config != '__is_first_stage__'
            assert config != '__is_unconditional__'
            model = instantiate_from_config(config)
            self.cond_stage_model = model
            self.cond_stage_model.train()

    def instantiate_structcond_stage(self, config):
        if self.struct_stage_trainable:
            model = instantiate_from_config(config)
            self.structcond_stage_model = model
            self.structcond_stage_model.train()
        else :
            model = instantiate_from_config(config)
            self.structcond_stage_model = model.eval()
                # self.cond_stage_model.train = disabled_train
            for name, param in self.structcond_stage_model.named_parameters():
                param.requires_grad = False

    def get_first_stage_encoding(self, encoder_posterior):
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.sample()
        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented")
        return self.scale_factor * z

    def get_learned_conditioning(self, c):
        if self.cond_stage_forward is None:
            if hasattr(self.cond_stage_model, 'encode') and callable(self.cond_stage_model.encode):
                c = self.cond_stage_model.encode(c)
                if isinstance(c, DiagonalGaussianDistribution):
                    c = c.mode()
            else:
                c = self.cond_stage_model(c)
        else:
            assert hasattr(self.cond_stage_model, self.cond_stage_forward)
            c = getattr(self.cond_stage_model, self.cond_stage_forward)(c)
        return c
    
    def get_style_features(self, features,):
         # [fea1(B C H W),...,fea5] -> [fea1(B C 1),...fea5] -> [fea1(B C),...fea5] -> B C(c1+...+c5)
        style_features = torch.cat([torch.cat(torch.std_mean(f, dim=[-1, -2]), dim=1) for f in features], dim=1)
        return style_features

    def get_content_features(self, features):
        content_features = features[:, :self.first_stage_model.embed_dim]
        std, mean = torch.std_mean(content_features, dim=[-1, -2], keepdim=True)
        content_features = (content_features - mean) / std
        return content_features
    
    def get_image_input(self, batch): 
            content = batch["content"]
            style = batch["style"]
            
            if len(content.shape) == 3:
                content = content[..., None]
            if len(style.shape) == 3:
                style = style[..., None]

            content = rearrange(content, 'b h w c -> b c h w')
            style = rearrange(style, 'b h w c -> b c h w')

            content = content.to(memory_format=torch.contiguous_format).float()
            style = style.to(memory_format=torch.contiguous_format).float()
            
            return content,style
    
    
    @torch.no_grad()
    def get_input(self, batch, k=None, return_first_stage_outputs=False, return_content_features=False,
                  bs=None, *args, **kwargs):
 
        content, style = self.get_image_input(batch)

        if bs is not None:
            content = content[:bs]

        content = content.to(self.device)
        style = style.to(self.device)
        

        contents=self.encode_first_stage(content)
        contents=self.get_first_stage_encoding(contents) 
        c_content = self.get_content_features(contents)

        vgg_style = self.vgg_scaling_layer(style)
        vgg_features = self.vgg(vgg_style)
        c_style = self.get_style_features(vgg_features)

        text_cond=['']
        while len(text_cond) < content.size(0):
            text_cond.append(text_cond[-1])
        if len(text_cond) > content.size(0):
            text_cond = text_cond[:content.size(0)]
        assert len(text_cond) == content.size(0)

        c_text = self.cond_stage_model(text_cond)

        c = {'c':c_text ,'c1': c_content, 'c2': c_style}
        out = [content,style, c]
        return out
    
    # @torch.no_grad()
    def decode_first_stage(self, z, predict_cids=False, force_not_quantize=False):
        if predict_cids:
            if z.dim() == 4:
                z = torch.argmax(z.exp(), dim=1).long()
            z = self.first_stage_model.quantize.get_codebook_entry(z, shape=None)
            z = rearrange(z, 'b h w c -> b c h w').contiguous()

        z = 1. / self.scale_factor * z

        if isinstance(self.first_stage_model, VQModelInterface):
            return self.first_stage_model.decode(z, force_not_quantize=predict_cids or force_not_quantize)
        else:
            return self.first_stage_model.decode(z)


    # same as above but without decorator
    def differentiable_decode_first_stage(self, z, predict_cids=False, force_not_quantize=False):
        if predict_cids:
            if z.dim() == 4:
                z = torch.argmax(z.exp(), dim=1).long()
            z = self.first_stage_model.quantize.get_codebook_entry(z, shape=None)
            z = rearrange(z, 'b h w c -> b c h w').contiguous()

        z = 1. / self.scale_factor * z

        if isinstance(self.first_stage_model, VQModelInterface):
            return self.first_stage_model.decode(z, force_not_quantize=predict_cids or force_not_quantize)
        else:
            return self.first_stage_model.decode(z)

    @torch.no_grad()
    def encode_first_stage(self, x):
        return self.first_stage_model.encode(x)

    def shared_step(self, batch, **kwargs):
        content,style, c = self.get_input(batch)
        loss,loss_dict = self(content,style, c)
        return loss,loss_dict

    def forward(self,content, style, cond, *args, **kwargs):
        content=self.tensor_to_rgb(content)
        style=self.tensor_to_rgb(style)

        if self.training:
            mid_timestep = random.randint(0, int(self.total_ddim_steps)-1)
        else:
            mid_timestep = 0
        shape = (self.channels, self.image_size//8, self.image_size//8)
        bs=content.shape[0]
        # if self.sampler =="dpm-solver":
        #     sampler = DPMSolverSampler(self)
        #     with torch.no_grad():
        #         early_stop_step = int(self.total_ddim_steps)-mid_timestep-1
        #         x, _ ,timesteps = sampler.sample(S=self.total_ddim_steps,batch_size = bs,shape = shape,
        #                                                     conditioning = cond,verbose=False,train_mode=False,
        #                                                     early_stop_step = early_stop_step)

        #     with torch.enable_grad():
        #         x = x.requires_grad_(True)
        #         t_cond = cond.copy()
        #         t_cond["c1"].requires_grad_(True)
        #         t_cond["c"].requires_grad_(True)
        #         t_cond["c2"].requires_grad_(True)
        #         # time_range = timesteps
        #         # total_step = len(time_range)
        #         vec_t = timesteps[early_stop_step].expand((x.shape[0]))

        #         model_fn = model_wrapper(
        #             sampler.model,
        #             sampler.ns,
        #             model_type="noise",
        #             guidance_type="classifier-free",
        #             condition=t_cond,
        #             unconditional_condition=None,
        #             guidance_scale=1.,
        #         )
        #         dpm_solver = DPM_Solver(model_fn, sampler.ns, predict_x0=True, thresholding=False)
        #         pred_x0 = dpm_solver.model_fn(x,vec_t)
        #         output_img = self.decode_first_stage(pred_x0)
        #         output_img = self.tensor_to_rgb(output_img)
        # elif self.sampler =="plms":
        #     plms_sampler = PLMSSampler(self)
        #     with torch.no_grad():
        #         latent_samples, _ ,old_eps = plms_sampler.sample(S=self.total_ddim_steps,batch_size = bs,shape = shape,
        #                                     conditioning = cond,verbose=False,train_mode=False,
        #                                     early_stop_step = mid_timestep+1)

        #     with torch.enable_grad():
        #         latent_model_input = latent_samples.requires_grad_(True)
        #         t_cond = cond.copy()
        #         t_cond["c1"].requires_grad_(True)
        #         t_cond["c"].requires_grad_(True)
        #         t_cond["c2"].requires_grad_(True)

        #         t_next_cond = cond.copy()
        #         t_next_cond["c1"].requires_grad_(True)
        #         t_next_cond["c"].requires_grad_(True)
        #         t_next_cond["c2"].requires_grad_(True)
                
        #         time_range = plms_sampler.ddim_timesteps
        #         total_step = len(time_range)
        #         index = mid_timestep
        #         ts = torch.full((bs,), time_range[mid_timestep], device=self.device, dtype=torch.long)
        #         ts_next = torch.full((bs,), time_range[min(mid_timestep + 1, total_step - 1)], device=self.device, dtype=torch.long)
        #         t_cond["c1"] = self.structcond_stage_model(t_cond["c1"], ts)
        #         t_next_cond["c1"] = self.structcond_stage_model(t_next_cond["c1"], ts_next)

        #         _, pred_x0, _ = plms_sampler.p_sample_plms(latent_model_input, t_cond, ts, index=index, use_original_steps=False,
        #                                                    old_eps=old_eps,t_next=ts_next,t_next_cond=t_next_cond,train_mode=True,)
        #         output_img = self.decode_first_stage(pred_x0)
        #         output_img = self.tensor_to_rgb(output_img)
        # el
        if self.sampler =="ddim":
            ddim_sampler = DDIMSampler(self)
            with torch.no_grad():
                latent_samples = ddim_sampler.sample(s = self.total_ddim_steps,batch_size = bs,shape = shape,
                                            conditioning = cond,verbose=False,train_mode=False,eta=1.0,
                                            early_stop_step = mid_timestep+1 )[0]

            with torch.enable_grad():
                latent_model_input = latent_samples.requires_grad_(True)
                t_cond = cond.copy()
                t_cond["c1"].requires_grad_(True)
                t_cond["c"].requires_grad_(True)
                t_cond["c2"].requires_grad_(True)

                index = mid_timestep
                ts = torch.full((bs,), ddim_sampler.ddim_timesteps[mid_timestep], device=self.device, dtype=torch.long)
                t_cond["c1"] = self.structcond_stage_model(t_cond["c1"], ts)

                _, pred_x0 = ddim_sampler.p_sample_ddim(latent_model_input, t_cond, ts, index=index, use_original_steps=False,train_mode=True,)
                output_img = self.decode_first_stage(pred_x0)
                output_img = self.tensor_to_rgb(output_img)
        else:
            raise Exception("You choose wrong sampler!")


        loss_c, loss_s = self.vgg_encoder(content, style, output_img, content_threshold=self.content_threshold)
        t_index = ddim_sampler.ddim_timesteps[mid_timestep]
        alphas_cumprod_t_index = self.alphas_cumprod[t_index]
        
        vgg_loss_style_weight = self.style_weight_ * torch.sqrt(alphas_cumprod_t_index) # V3
        vgg_loss_content_weight = 1 - vgg_loss_style_weight

        loss = vgg_loss_content_weight * loss_c + loss_s * vgg_loss_style_weight

        
        loss_dict = {}
        loss_dict.update({"total_loss": loss})
        loss_dict.update({"content weight": vgg_loss_content_weight})
        loss_dict.update({f"loss_c": loss_c})
        loss_dict.update({f"loss_s": loss_s})

        return loss, loss_dict

    def apply_model(self, x_noisy, t, c, only_style=False, only_content=False, return_ids=False):   #
    
        cond=c['c']
        if isinstance(cond, dict):
            pass
        else:
            if not isinstance(cond, list):
                cond = [cond]
            key = 'c_concat' if self.model.conditioning_key == 'concat' else 'c_crossattn'
            cond = {key: cond}

        cond['struct_cond'] = c['c1']
        style=c['c2']
        x_recon = self.model(x_noisy, t, **cond,seg_cond=style, only_style=only_style, only_content=only_content)

        if isinstance(x_recon, tuple) and not return_ids:
            return x_recon[0]
        else:
            return x_recon

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - pred_xstart) / \
               extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)


    def tensor_to_rgb(self,x):
        return torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
    
    def p_losses(self, x_start,c,t, t_ori, z_gt, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        #vgg19.to(self.device)
        
        if self.mix_ratio > 0:
            if random.random() < self.mix_ratio:
                noise_new = default(noise, lambda: torch.randn_like(x_start))
                noise = noise_new * 0.5 + noise * 0.5
                x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        #print(x_start.shape[0])
        model_output = self.apply_model(x_noisy, t_ori, c)
        
        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        elif self.parameterization == "v":
            target = self.get_v(x_start, noise, t)
        else:
            raise NotImplementedError()

        model_output_ = model_output
        
        loss_simple = self.get_loss(model_output_, target, mean=False).mean([1, 2, 3])
        loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})

        #P2 weighting
        if self.snr is not None:
            self.snr = self.snr.to(loss_simple.device)
            weight = extract_into_tensor(1 / (self.p2_k + self.snr)**self.p2_gamma, t, target.shape)
            loss_simple = weight * loss_simple

        logvar_t = self.logvar[t.cpu()].to(self.device)
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        #loss = loss_simple / torch.exp(self.logvar) + self.logvar
        if self.learn_logvar:
            loss_dict.update({f'{prefix}/loss_gamma': loss.mean()})
            loss_dict.update({'logvar': self.logvar.data.mean()})

        loss = self.l_simple_weight * loss.mean()

        loss_vlb = self.get_loss(model_output_, target, mean=False).mean(dim=(1, 2, 3))
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})
        loss += (self.original_elbo_weight * loss_vlb)
        
        loss_dict.update({f'{prefix}/loss': loss})
        
        return loss, loss_dict

    @torch.no_grad()
    def sample_log(self,cond,batch_size,sampler=None,height=None,width=None, ddim_steps=20,log_every_t=100,**kwargs):
        if sampler is None:
            sampler =self.sampler

        if height is not None and width is not None:
            shape = (self.channels, height//8, width//8)
        else:
            shape = (self.channels, self.image_size//8, self.image_size//8)

        if sampler == "dpm-solver":
            dpm_solver_sampler = DPMSolverSampler(self)
            samples, intermediates, _ = dpm_solver_sampler.sample(ddim_steps,batch_size,
                                                            shape,cond,verbose=False,
                                                            log_every_t=log_every_t,**kwargs)
        elif sampler == "plms":
            plms_sampler = PLMSSampler(self)
            samples, intermediates, _ = plms_sampler.sample(ddim_steps,batch_size,
                                            shape,cond,verbose=False,
                                            log_every_t=log_every_t,**kwargs)
        elif sampler == "ddim":  
            ddim_sampler = DDIMSampler(self)
            samples, intermediates =ddim_sampler.sample(ddim_steps,batch_size,
                                                        shape,cond,verbose=False,
                                                        log_every_t=log_every_t,**kwargs)
        else:
            raise Exception("You choose wrong sampler!")

        return samples, intermediates
    
    @torch.no_grad()
    def log_images(self, batch, max_image=None, return_keys=None, **kwargs):
        log = dict()
        content,style,c = self.get_input(batch)
        if max_image:
            N = min(content.shape[0],max_image)
            content = content[:N]
            style = style[:N]
            c["c"] = c["c"][:N]
            c["c1"] = c["c1"][:N]
            c["c2"] = c["c2"][:N]
        
        log["content"] = self.tensor_to_rgb(content)
        log["style"] = self.tensor_to_rgb(style)

        eta = 1. if self.sampler =="ddim" else 0.
        size = self.image_size
        
        total_samples = list()
        for i in range(len(content)):
            c_in = {'c':c["c"][i:i+1], 'c1':c["c1"][i:i+1], "c2":c["c2"][i:i+1]}
            samples = self.sample_log(c_in,batch_size=1,ddim_steps=self.total_ddim_steps,
                                        height=size,width=size,eta=eta)[0]
            x_samples = self.decode_first_stage(samples)
            x_samples = self.tensor_to_rgb(x_samples)

            total_samples.append(x_samples)
        total_samples = torch.cat(total_samples,dim=0)
        log["sample"] = total_samples
        return log


    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())
        params = params + list(self.structcond_stage_model.parameters())
        if self.learn_logvar:
            assert not self.learn_logvar
            print('Diffusion model optimizing logvar')
            params.append(self.logvar)
        opt = torch.optim.AdamW(params, lr=lr)
        if self.use_scheduler:
            assert 'target' in self.scheduler_config
            scheduler = instantiate_from_config(self.scheduler_config)

            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    'scheduler': LambdaLR(opt, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                }]
            return [opt], scheduler
        return opt

    @torch.no_grad()
    def to_rgb(self, x):
        x = x.float()
        if not hasattr(self, "colorize"):
            self.colorize = torch.randn(3, x.shape[1], 1, 1).to(x)
        x = nn.functional.conv2d(x, weight=self.colorize)
        x = 2. * (x - x.min()) / (x.max() - x.min()) - 1.
        return x


class DiffusionWrapper(pl.LightningModule):
    def __init__(self, diff_model_config, conditioning_key):
        super().__init__()
        self.diffusion_model = instantiate_from_config(diff_model_config)
        self.conditioning_key = conditioning_key
        assert self.conditioning_key in [None, 'concat', 'crossattn', 'hybrid', 'adm']

    def forward(self, x, t, c_concat: list = None, c_crossattn: list = None, struct_cond=None, seg_cond=None, only_style=False, only_content=False):
        if self.conditioning_key is None:
            out = self.diffusion_model(x, t)
        elif self.conditioning_key == 'concat':
            xc = torch.cat([x] + c_concat, dim=1)
            out = self.diffusion_model(xc, t)
        elif self.conditioning_key == 'crossattn':         #crossattn
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(x, t, context=cc, struct_cond=struct_cond, seg_cond=seg_cond)   #into it
        elif self.conditioning_key == 'hybrid':
            xc = torch.cat([x] + c_concat, dim=1)
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(xc, t, context=cc)
        elif self.conditioning_key == 'adm':
            cc = c_crossattn[0]
            out = self.diffusion_model(x, t, y=cc)
        else:
            raise NotImplementedError()

        return out
