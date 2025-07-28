"""SAMPLING ONLY."""
from asyncio import FastChildWatcher
import copy
from turtle import Turtle

import torch
import numpy as np
from tqdm import tqdm
from functools import partial

from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like


class DDIMSampler(object):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
     
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)
        # print(self.ddim_timesteps)

        alphas_cumprod = self.model.alphas_cumprod
        
        #assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

   
    def sample(self,
               s,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
            #    unconditional_guidance_scale_2=1.,
            #    unconditional_conditioning_2=None,
               early_stop_step = None,
               train_mode = False,
               only_style=False,
               only_content=False,
               s_c_mode=None,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):
        
        assert not train_mode or not only_style
        # if conditioning is not None:
        #     if isinstance(conditioning, dict):
        #         cbs = conditioning[list(conditioning.keys())[0]].shape[0]
        #         if cbs != batch_size:
        #             print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
        #     else:
        #         if conditioning.shape[0] != batch_size:
        #             print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=s, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        # print(f'Data shape for DDIM sampling is {size}, eta {eta}')
        # exit()
       
        samples, intermediates = self.ddim_sampling(conditioning, size,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    # unconditional_guidance_scale_2=unconditional_guidance_scale_2,
                                                    # unconditional_conditioning_2=unconditional_conditioning_2,
                                                    early_stop_step = early_stop_step,
                                                    train_mode=train_mode,
                                                    only_style=only_style,
                                                    only_content=only_content,
                                                    s_c_mode=s_c_mode,
                                                    )
        return samples, intermediates

    
    def ddim_sampling(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,
                    #   unconditional_guidance_scale_2=1., unconditional_conditioning_2=None,
                      early_stop_step = None, train_mode=False,only_style=False,only_content=False,s_c_mode=None,
                      ):
        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T
        
        if early_stop_step is not None:
            stop_step = early_stop_step
        else:
            stop_step = 0

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'x_inter': [img], 'pred_x0': [img],'e_t': [img]}
        time_range = reversed(range(stop_step,timesteps)) if ddim_use_original_steps else np.flip(timesteps[stop_step:])

        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        
        uncond_guidance = unconditional_conditioning is not None and unconditional_guidance_scale != 1.
        # uncond_guidance_2 = unconditional_conditioning_2 is not None and unconditional_guidance_scale_2 != 1.
        
       
        t_cond=cond.copy()
        
        struct_cond_input=cond["c1"]
        if uncond_guidance:  
            t_conditioning=unconditional_conditioning.copy()
            struct_cond_input_1=unconditional_conditioning['c1']
        # if uncond_guidance_2:
        #     t_conditioning_2=unconditional_conditioning_2.copy()
        #     struct_cond_input_2=unconditional_conditioning_2["c1"]
        #iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)
        iterator=tqdm(time_range, total=total_steps,disable=True)

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            # if index<2 and s_c_mode=="ns_nc":
            #     only_content = True
            #     only_style = True
            # elif index<2 and s_c_mode=="s_nc":
            #     only_content = False
            #     only_style = True
            # elif index<2 and s_c_mode=="ns_c":
            #     only_content = True
            #     only_style = False
            # else:
            #     only_content = only_style = False
            ts = torch.full((b,), step, device=device, dtype=torch.long)
            if uncond_guidance:    
                t_conditioning['c1']=self.model.structcond_stage_model(struct_cond_input_1,ts)
            # if uncond_guidance_2:    
            #     t_conditioning_2['c1']=self.model.structcond_stage_model(struct_cond_input_2,ts)
            struct_cond = self.model.structcond_stage_model(struct_cond_input, ts)
            t_cond["c1"]=struct_cond
            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                img = img_orig * mask + (1. - mask) * img

            outs = self.p_sample_ddim(img, t_cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=t_conditioning if uncond_guidance else unconditional_conditioning,
                                    #   unconditional_guidance_scale_2=unconditional_guidance_scale_2,
                                    #   unconditional_conditioning_2=t_conditioning_2 if uncond_guidance_2 else unconditional_conditioning_2,
                                      train_mode=train_mode,only_style=only_style,only_content=only_content,
                                      )
            # img, pred_x0, e_t = outs
            img, pred_x0 = outs
            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)
                # intermediates['e_t'].append(e_t)

        return img, intermediates

   
    def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,
                    #   unconditional_guidance_scale_2=1., unconditional_conditioning_2=None,
                      train_mode=False,only_style=False,only_content=False,):
        b, *_, device = *x.shape, x.device

        x_in = x
        t_in = t
        if not train_mode:
            c_in = copy.deepcopy(c) if isinstance(c, dict) else c
        else:
            c_in = c

        # if train_mode:
        #     print("x_in and :",x_in.requires_grad)

        uncond_guidance = unconditional_conditioning is not None and unconditional_guidance_scale != 1.
        # uncond_guidance_2 = unconditional_conditioning_2 is not None and unconditional_guidance_scale_2 != 1.
        if uncond_guidance:
            x_in = torch.cat([x_in, x])
            t_in = torch.cat([t_in, t])
            if isinstance(c_in, dict):
                for key in c_in.keys():
                    if key=='c'or key=="c2":
                        c_in[key] = torch.cat([c_in[key], unconditional_conditioning[key]])
                    if key=='c1':
                        for key2 in c_in[key].keys():
                            c_in[key][key2]=torch.cat([c_in[key][key2], unconditional_conditioning[key][key2]])
            else:
                c_in = torch.cat([c_in, unconditional_conditioning])
        # if uncond_guidance_2:
        #     x_in = torch.cat([x_in, x])
        #     t_in = torch.cat([t_in, t])
        #     if isinstance(c_in, dict):
        #         for key in c_in.keys():
        #             if key=='c' or key=="c2":
        #                 c_in[key] = torch.cat([c_in[key], unconditional_conditioning_2[key]])
        #             if key=='c1':
        #                 for key2 in c_in[key].keys():
        #                     c_in[key][key2]=torch.cat([c_in[key][key2], unconditional_conditioning_2[key][key2]])
        #     else:
        #         c_in = torch.cat([c_in, unconditional_conditioning_2])
        e_t = self.model.apply_model(x_in, t_in, c_in, only_style,only_content)
        # if train_mode:
        #     print("e_t:",e_t.requires_grad)
        if uncond_guidance:
            e_t, e_t_uncond = e_t.chunk(2)
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

        # if uncond_guidance and not uncond_guidance_2:
        #     e_t, e_t_uncond = e_t.chunk(2)
        #     e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)
        # elif not uncond_guidance and uncond_guidance_2:
        #     e_t, e_t_uncond = e_t.chunk(2)
        #     e_t = e_t_uncond + unconditional_guidance_scale_2 * (e_t - e_t_uncond)
        # elif uncond_guidance and uncond_guidance_2:
        #     e_t, e_t_uncond, e_t_uncond_2 = e_t.chunk(3)
        #     e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond) + \
        #           e_t_uncond_2 + unconditional_guidance_scale_2 * (e_t - e_t_uncond_2) - e_t

        if score_corrector is not None:
            assert self.model.parameterization == "eps"
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        # return x_prev, pred_x0, e_t
        return x_prev, pred_x0
