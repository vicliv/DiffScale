import os
import logging
import time
import glob

import numpy as np
import tqdm
import torch
import torch.utils.data as data

from models.diffusion import Model
from models.ema import EMAHelper
from functions import get_optimizer
from functions.losses import loss_registry
from datasets import get_dataset, data_transform, inverse_data_transform
from functions.ckpt_util import get_ckpt_path

import torchvision.utils as tvu
import matplotlib.pyplot as plt

f = 3

def torch2hwcuint8(x, clip=False):
    if clip:
        x = torch.clamp(x, -1, 1)
    x = (x + 1.0) / 2.0
    return x


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
            # torch.cat(
            # [posterior_variance[1:2], betas[1:]], dim=0).log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

    def train(self):
        print("train")
        args, config = self.args, self.config
        tb_logger = self.config.tb_logger
        dataset, test_dataset = get_dataset(args, config)
        train_loader = data.DataLoader(
            dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
        )
        model = Model(config)

        model = model.to(self.device)
        model = torch.nn.DataParallel(model)

        optimizer = get_optimizer(self.config, model.parameters())

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(model)
        else:
            ema_helper = None

        start_epoch, step = 0, 0
        if self.args.resume_training:
            states = torch.load(os.path.join(self.args.log_path, "ckpt.pth"))
            model.load_state_dict(states[0])

            states[1]["param_groups"][0]["eps"] = self.config.optim.eps
            optimizer.load_state_dict(states[1])
            start_epoch = states[2]
            step = states[3]
            if self.config.model.ema:
                ema_helper.load_state_dict(states[4])

        for epoch in range(start_epoch, self.config.training.n_epochs):
            data_start = time.time()
            data_time = 0
            for i, (x, y) in enumerate(train_loader):
                n = x.size(0)
                data_time += time.time() - data_start
                model.train()
                step += 1

                x = x.to(self.device)
                x = data_transform(self.config, x)
                b = self.betas

                # antithetic sampling
                # t = torch.randint(
                #     low=0, high=self.num_timesteps, size=(n // 2 + 1,)
                # ).to(self.device)
                # t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
                
                bin = torch.randint(0, f, (1,))
                t = torch.randint(self.num_timesteps//f*bin, self.num_timesteps//f*(bin+1), (n,), device=self.device).long()
                loss = loss_registry[config.model.type](model, x, t, b, bin)

                tb_logger.add_scalar("loss", loss, global_step=step)

                logging.info(
                    f"epoch: {epoch}, step: {step}, loss: {loss.item()}, data time: {data_time / (i+1)}"
                )

                optimizer.zero_grad()
                loss.backward()

                try:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.optim.grad_clip
                    )
                except Exception:
                    pass
                optimizer.step()

                if self.config.model.ema:
                    ema_helper.update(model)

                if step % self.config.training.snapshot_freq == 0 or step == 1:
                    states = [
                        model.state_dict(),
                        optimizer.state_dict(),
                        epoch,
                        step,
                    ]
                    if self.config.model.ema:
                        states.append(ema_helper.state_dict())

                    torch.save(
                        states,
                        os.path.join(self.args.log_path, "ckpt_{}.pth".format(step)),
                    )
                    torch.save(states, os.path.join(self.args.log_path, "ckpt.pth"))

                data_start = time.time()
            if epoch % 5 == 0:
                n = 9
                size = config.data.image_size // (2 ** (f-1))
                x = torch.randn(
                    n,
                    config.data.channels,
                    size,
                    size,
                    device=self.device,
                )
                
                x = self.generate_image(x, model)
                x = inverse_data_transform(config, x)

                merged = tvu.make_grid(x, 3)
                tvu.save_image(
                    merged, os.path.join(self.args.image_folder, f"{step}.png")
                )

    def sample(self):
        model = Model(self.config)

        if not self.args.use_pretrained:
            if getattr(self.config.sampling, "ckpt_id", None) is None:
                states = torch.load(
                    os.path.join(self.args.log_path, "ckpt.pth"),
                    map_location=self.config.device,
                )
            else:
                states = torch.load(
                    os.path.join(
                        self.args.log_path, f"ckpt_{self.config.sampling.ckpt_id}.pth"
                    ),
                    map_location=self.config.device,
                )
            model = model.to(self.device)
            model = torch.nn.DataParallel(model)
            model.load_state_dict(states[0], strict=True)

            if self.config.model.ema:
                ema_helper = EMAHelper(mu=self.config.model.ema_rate)
                ema_helper.register(model)
                ema_helper.load_state_dict(states[-1])
                ema_helper.ema(model)
            else:
                ema_helper = None
        else:
            # This used the pretrained DDPM model, see https://github.com/pesser/pytorch_diffusion
            if self.config.data.dataset == "CIFAR10":
                name = "cifar10"
            elif self.config.data.dataset == "LSUN":
                name = f"lsun_{self.config.data.category}"
            else:
                raise ValueError
            ckpt = get_ckpt_path(f"ema_{name}")
            print("Loading checkpoint {}".format(ckpt))
            model.load_state_dict(torch.load(ckpt, map_location=self.device))
            model.to(self.device)
            model = torch.nn.DataParallel(model)

        model.eval()

        if self.args.fid:
            self.sample_fid(model)
        elif self.args.interpolation:
            self.sample_interpolation(model)
        elif self.args.sequence:
            self.sample_sequence(model)
        elif self.args.reconstruct:
            self.sample_reconstruct(model)
        else:
            raise NotImplementedError("Sample procedeure not defined")

    def sample_fid(self, model):
        config = self.config
        img_id = len(glob.glob(f"{self.args.image_folder}/*"))
        print(f"starting from image {img_id}")
        total_n_samples = 300
        n_rounds = (total_n_samples - img_id) // config.sampling.batch_size

        with torch.no_grad():
            for _ in tqdm.tqdm(
                range(n_rounds), desc="Generating image samples for FID evaluation."
            ):
                n = config.sampling.batch_size
                x = torch.randn(
                    n,
                    config.data.channels,
                    config.data.image_size,
                    config.data.image_size,
                    device=self.device,
                )

                x = self.sample_image(x, model)
                x = inverse_data_transform(config, x)

                for i in range(n):
                    tvu.save_image(
                        x[i], os.path.join(self.args.image_folder, f"{img_id}.png")
                    )
                    img_id += 1

    def sample_sequence(self, model):
        config = self.config

        x = torch.randn(
            8,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )

        # NOTE: This means that we are producing each predicted x0, not x_{t-1} at timestep t.
        with torch.no_grad():
            _, x = self.sample_image(x, model, last=False)

        x = [inverse_data_transform(config, y) for y in x]

        for i in range(len(x)):
            for j in range(x[i].size(0)):
                tvu.save_image(
                    x[i][j], os.path.join(self.args.image_folder, f"{j}_{i}.png")
                )

    def sample_interpolation(self, model):
        config = self.config

        def slerp(z1, z2, alpha):
            theta = torch.acos(torch.sum(z1 * z2) / (torch.norm(z1) * torch.norm(z2)))
            return (
                torch.sin((1 - alpha) * theta) / torch.sin(theta) * z1
                + torch.sin(alpha * theta) / torch.sin(theta) * z2
            )

        z1 = torch.randn(
            1,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )
        z2 = torch.randn(
            1,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )
        alpha = torch.arange(0.0, 1.01, 0.1).to(z1.device)
        z_ = []
        for i in range(alpha.size(0)):
            z_.append(slerp(z1, z2, alpha[i]))

        x = torch.cat(z_, dim=0)
        xs = []

        # Hard coded here, modify to your preferences
        with torch.no_grad():
            for i in range(0, x.size(0), 8):
                xs.append(self.sample_image(x[i : i + 8], model))
        x = inverse_data_transform(config, torch.cat(xs, dim=0))
        for i in range(x.size(0)):
            tvu.save_image(x[i], os.path.join(self.args.image_folder, f"{i}.png"))
    
    def sample_reconstruct7(self, model):
        config = self.config
        img_id = len(glob.glob(f"{self.args.image_folder}/*"))-4
        print(f"starting from image {img_id}")
        total_n_samples = 1
        n_rounds = (total_n_samples - img_id) // config.sampling.batch_size
        
        n_rounds = 10
        
        def merge_images(image_batch, size):
            h,w = 64, 64
            c = 3
            img = torch.zeros((c, int(h*size[0]), w*size[1]))
            for idx, im in enumerate(image_batch):
                i = idx % size[1]
                j = idx // size[1]
                img[:, j*h:j*h+h, i*w:i*w+w] = im
            return img
        
        max_change = torch.zeros(1, config.data.channels,
                        config.data.image_size,
                        config.data.image_size)
        min_change = torch.zeros(1, config.data.channels,
                        config.data.image_size,
                        config.data.image_size)

        with torch.no_grad():
            im_da = []
            im_de = []
            for k in tqdm.tqdm(
                range(n_rounds), desc="Generating reconstructed images."
            ):
                n = config.sampling.batch_size
                x = torch.randn(
                    n,
                    config.data.channels,
                    config.data.image_size,
                    config.data.image_size,
                    device=self.device,
                )
                
                # x = torch.zeros(
                #     n,
                #     config.data.channels,
                #     config.data.image_size,
                #     config.data.image_size,
                #     device=self.device,
                # )
                og_noise = x.clone()
                
                if k == 0:
                    jacobian = self.get_jacobian(x, model, 500)
                    
                    U, S, Vt = torch.svd(jacobian.view(config.data.channels * 
                        config.data.image_size**2, config.data.channels *
                        config.data.image_size**2))
                    
                    print(torch.max(S))
                    print(torch.min(S))
                    print(S)
                    
                    max_change = Vt[0, :].view(1, config.data.channels,
                                    config.data.image_size,
                                    config.data.image_size)
                    sec_change = Vt[1, :].view(1, config.data.channels,
                                    config.data.image_size,
                                    config.data.image_size)
                    min_change = Vt[-1, :].view(1, config.data.channels,
                                    config.data.image_size,
                                    config.data.image_size)
                
                # max_change /= torch.norm(max_change)
                # sec_change /= torch.norm(sec_change)
                # min_change /= torch.norm(min_change)
                im = []
                for i in range(0, 20):
                    x1 = self.sample_image(x+2*i*max_change, model)
                    
                    x2 = inverse_data_transform(config, x1)
                    
                    tvu.save_image(
                        x2[0], os.path.join(self.args.image_folder, "generated", f"{k}_{img_id}.png")             
                    )
                    img_id += 1
                    im.append(x2[0])
                    im_da.append(x2[0])
                
                for i in range(0,20):
                    x1 = self.sample_image(x+2*i*min_change, model)
                    
                    x2 = inverse_data_transform(config, x1)
                    
                    tvu.save_image(
                        x2[0], os.path.join(self.args.image_folder, "generated", f"{k}_{img_id}.png")             
                    )
                    img_id += 1
                    im.append(x2[0])
                    im_de.append(x2[0])
                    
                # for latent in [x, x+scale*max_change, x+scale*sec_change, x+scale*min_change]:
                #     x1 = self.sample_image(latent, model)
                    
                #     x2 = inverse_data_transform(config, x1)
                    
                #     tvu.save_image(
                #         x2[0], os.path.join(self.args.image_folder, "generated", f"{img_id}.png")             
                #     )
                #     img_id += 1
                #     im.append(x2[0])
                
                im_merged = merge_images(im, [2,20])
                tvu.save_image(
                    im_merged, os.path.join(self.args.image_folder, "generated", f"{k}_face.png")
                )
            ims_merged = merge_images(im_da, [n_rounds,20])
            tvu.save_image(
                ims_merged, os.path.join(self.args.image_folder, "generated", f"allfacesmax.png")
            )
            ims_merged = merge_images(im_de, [n_rounds,20])
            tvu.save_image(
                ims_merged, os.path.join(self.args.image_folder, "generated", f"allfacesmin.png")
            )
                # for i in range(len(im)):
                #     y = torch.zeros(
                #         n,
                #         config.data.channels,
                #         config.data.image_size * 6,
                #         config.data.image_size * 6,
                #         device=self.device,
                #     )
                #     tvu.save_image(
                #         x1[i], os.path.join(self.args.image_folder, "generated", f"{img_id}.png")
                #     )
                #     img_id += 1
                # img_id -=n
                
                # x = self.get_latent(x.to(self.device), model)
                # x_latent = x
                
                # im = []
                # for j in range(64):
                #     x = x_latent.clone()
                #     x[:, :, j, j] += 1
                    
                #     x2 = inverse_data_transform(config, x)
                    
                #     for i in range(n):
                #         tvu.save_image(
                #             x2[i], os.path.join(self.args.image_folder, "noise", f"{img_id}_{j}.png")
                #         )
                #         img_id += 1
                #     img_id -=n

                #     x = self.sample_image(x.to(self.device), model)
                    
                #     x3 = inverse_data_transform(config, x)
                    
                #     for i in range(n):
                #         tvu.save_image(
                #             x3[i], os.path.join(self.args.image_folder, "reconstructed", f"{img_id}_{j}.png")
                #         )
                #         img_id += 1
                #     im.append(x3)
                #for j in range(64):
    
    def sample_reconstruct2(self, model):
        config = self.config
        img_id = len(glob.glob(f"{self.args.image_folder}/*"))-4
        print(f"starting from image {img_id}")
        total_n_samples = 1
        n_rounds = (total_n_samples - img_id) // config.sampling.batch_size
        
        n_rounds = 20
        mid = 10
        
        def merge_images(image_batch, size):
            h,w = 64, 64
            c = 3
            img = torch.zeros((c, int(h*size[0]), w*size[1]))
            for idx, im in enumerate(image_batch):
                i = idx % size[1]
                j = idx // size[1]
                img[:, j*h:j*h+h, i*w:i*w+w] = im
            return img
        n = config.sampling.batch_size
        x = torch.randn(
            n,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )
        og_noise = x.clone()
        
        jacobian = self.get_jacobian(x, model, 500)
            
        U, S, Vt = torch.svd(jacobian.view(config.data.channels * 
            config.data.image_size**2, config.data.channels *
            config.data.image_size**2))
        
        print(torch.max(S))
        print(torch.min(S))
        print(S)

        with torch.no_grad():
            im_da = []
            im_de = []
            for k in tqdm.tqdm(
                range(n_rounds), desc="Generating reconstructed images."
            ):
                
                if k < mid:
                    change = Vt[k, :].view(1, config.data.channels,
                                        config.data.image_size,
                                        config.data.image_size)
                else:
                    change = Vt[k-n_rounds, :].view(1, config.data.channels,
                                        config.data.image_size,
                                        config.data.image_size)
                print(torch.linalg.norm(change))
                
                im = []
                for i in range(0, 20):
                    x1 = self.sample_image(x+2*i*change, model)
                    
                    x2 = inverse_data_transform(config, x1)
                    
                    tvu.save_image(
                        x2[0], os.path.join(self.args.image_folder, "generated", f"{k}_{img_id}.png")             
                    )
                    img_id += 1
                    im.append(x2[0])
                    im_da.append(x2[0])
                    
            ims_merged = merge_images(im_da, [n_rounds,20])
            tvu.save_image(
                ims_merged, os.path.join(self.args.image_folder, "generated", f"allfacesmax.png")
            )

    def sample_reconstruct(self, model):
        config = self.config
        img_id = len(glob.glob(f"{self.args.image_folder}/*"))-4
        print(f"starting from image {img_id}")
        total_n_samples = 1
        n_rounds = (total_n_samples - img_id) // config.sampling.batch_size
        
        n_rounds = 100
        mid = 10
        
        def merge_images(image_batch, size):
            h,w = 64, 64
            c = 3
            img = torch.zeros((c, int(h*size[0]), w*size[1]))
            for idx, im in enumerate(image_batch):
                i = idx % size[1]
                j = idx // size[1]
                img[:, j*h:j*h+h, i*w:i*w+w] = im
            return img
        n = config.sampling.batch_size
        x = torch.randn(
            n,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )
        og_noise = x.clone()
        
        change = torch.zeros(1, config.data.channels,
                                config.data.image_size,
                                config.data.image_size)
        direc = torch.randn(1, config.data.channels,
                                config.data.image_size,
                                config.data.image_size)
        dots = []
        dists = []
        rat = [0, 1e-15,  1e-14, 1e-13,  1e-12, 1e-11,  1e-10, 1e-9,  1e-7,  1e-5,  1e-3,  1e-1, 1]
        with torch.no_grad():
            dir = []
            for k in tqdm.tqdm(
                rat, desc="Finding the distance"
            ):
                prev_change = change
                
                jacobian = self.get_jacobian(x + k*direc.to(self.device), model, 50)
                dis = torch.cdist(x.view((1, config.data.channels * config.data.image_size**2)),  (x+k*direc.to(self.device)).view((1, config.data.channels * config.data.image_size**2)))
                dis = torch.norm(k*direc)
                print(dis)
                dists.append(dis.cpu().squeeze())
            
                #U, S, Vt = torch.svd(jacobian)
                S, V = torch.lobpcg(jacobian.T * jacobian, k=1, largest = True)
                print(S)
                
                change = V[:, 0].view(1, config.data.channels,
                                        config.data.image_size,
                                        config.data.image_size)
                
                dir.append(change)
                dot = torch.dot(prev_change.cpu().view((config.data.channels * config.data.image_size**2)), change.cpu().view((config.data.channels * config.data.image_size**2)))
                print(dot)
                dots.append(torch.abs(dot))
            print(torch.dot(dir[0].cpu().view((config.data.channels * config.data.image_size**2)), dir[-1].cpu().view((config.data.channels * config.data.image_size**2))))
            
        plt.plot(dists, dots, "o-")
        plt.xscale("log")
        plt.xlabel("Distance to original point")
        plt.ylabel("Inner product with original point")
        plt.savefig("dots.png")
    
    # def sample_reconstruct(self, model):
    #     config = self.config
    #     img_id = len(glob.glob(f"{self.args.image_folder}/*"))-4
    #     print(f"starting from image {img_id}")
    #     total_n_samples = 1
    #     n_rounds = (total_n_samples - img_id) // config.sampling.batch_size
        
    #     n_rounds = 5
    #     mid = 10
        
    #     def merge_images(image_batch, size):
    #         h,w = 64, 64
    #         c = 3
    #         img = torch.zeros((c, int(h*size[0]), w*size[1]))
    #         for idx, im in enumerate(image_batch):
    #             i = idx % size[1]
    #             j = idx // size[1]
    #             img[:, j*h:j*h+h, i*w:i*w+w] = im
    #         return img
    #     n = config.sampling.batch_size
    #     x = torch.randn(
    #         n,
    #         config.data.channels,
    #         config.data.image_size,
    #         config.data.image_size,
    #         device=self.device,
    #     )
    #     og_noise = x.clone()
        
    #     change = torch.zeros(1, config.data.channels,
    #                             config.data.image_size,
    #                             config.data.image_size)
        
    #     #with torch.no_grad():
    #     if True:
    #         dir = []
    #         colors = ["red", "blue"]
    #         for k in tqdm.tqdm(
    #             range(n_rounds), desc="Generating reconstructed images."
    #         ):
    #             prev_change = change
                
    #             jacobian = self.get_jacobian(x, model, 50).view((config.data.channels * config.data.image_size**2,config.data.channels * config.data.image_size**2))
            
    #             #U, S, Vt = torch.svd(jacobian)
    #             S, V = torch.lobpcg(jacobian.T * jacobian, k=2, largest = True)
    #             #print(S)
    #             print(V)
    #             plt.plot(np.arange(0, 2), S.cpu(), "bo")
    #             plt.yscale("log")
    #             plt.savefig("plot" + str(k) + ".png")
    #             change = V[:, 0].view(1, config.data.channels,
    #                                     config.data.image_size,
    #                                     config.data.image_size)
    #             print(change)
    #             dir.append(change)
    #             print(torch.norm(prev_change.cpu().view((config.data.channels * config.data.image_size**2))- change.cpu().view((config.data.channels * config.data.image_size**2))))
    #             print(torch.dot(prev_change.cpu().view((config.data.channels * config.data.image_size**2)), change.cpu().view((config.data.channels * config.data.image_size**2))))
    #         #print(torch.dot(dir[0].cpu().view((config.data.channels * config.data.image_size**2)), dir[-1].cpu().view((config.data.channels * config.data.image_size**2))))
    #         plt.savefig("plot.png")
    
    def generate_image(self, x, model, last=False):
        try:
            skip = self.args.skip
        except Exception:
            skip = 1

        if self.args.sample_type == "generalized":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            else:
                raise NotImplementedError
            from functions.denoising import generalized_steps

            for bin in range(f-1, -1, -1):
                frames = []
                seq = range(self.num_timesteps//f*bin, self.num_timesteps//f*(bin+1), skip)
                xs = generalized_steps(x, seq, model, self.betas, eta=self.args.eta)
                x = xs[0][-1]
                
                if bin > 0:
                    _, x = model(x.to(self.device), (torch.ones(x.shape[0]) *  (self.num_timesteps//f*(bin) -1)).to(self.device).long())

        elif self.args.sample_type == "ddpm_noisy":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            else:
                raise NotImplementedError
            from functions.denoising import ddpm_steps

            x = ddpm_steps(x, seq, model, self.betas)
        else:
            raise NotImplementedError
        return x   

    def sample_image(self, x, model, last=True):
        try:
            skip = self.args.skip
        except Exception:
            skip = 1

        if self.args.sample_type == "generalized":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            else:
                raise NotImplementedError
            from functions.denoising import generalized_steps
            
            

            xs = generalized_steps(x, seq, model, self.betas, eta=self.args.eta)
            x = xs
        elif self.args.sample_type == "ddpm_noisy":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (
                    np.linspace(
                        0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                    )
                    ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from functions.denoising import ddpm_steps

            x = ddpm_steps(x, seq, model, self.betas)
        else:
            raise NotImplementedError
        if last:
            x = x[0][-1]
        return x
    
    def get_latent(self, x, model, last=True):
        try:
            skip = self.args.skip
        except Exception:
            skip = 1

        if self.args.sample_type == "generalized":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (
                    np.linspace(
                        0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                    )
                    ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from functions.denoising import reverse_generalized_steps

            xs = reverse_generalized_steps(x, seq, model, self.betas, eta=self.args.eta)
            x = xs
        else:
            raise NotImplementedError
        if last:
            x = x[0][-1]
        return x
    
    def get_jacobian(self, x, model, n_noise = 100, batch_size = 256):
        try:
            skip = self.args.skip
        except Exception:
            skip = 1

        if self.args.sample_type == "generalized":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (
                    np.linspace(
                        0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                    )
                    ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from functions.denoising import generalized_steps

            # xs = generalized_steps(x, seq, model, self.betas, eta=self.args.eta)
            # x = xs
            
            #vf = lambda vx: generalized_steps(vx.view((C, D1, D2)), seq, model, self.betas, eta=self.args.eta).view((FD))
            vf = lambda vx: generalized_steps(vx, seq, model, self.betas, eta=self.args.eta)
            
            N, C, D1, D2 = x.shape
            eps = 1e-15
            FD =  D1*D2*C

            # #J_approx = torch.zeros((FD, C, D1, D2)).to(self.device)
            # #noise = torch.eye(FD).to(self.device).view(FD, C, D1, D2)
            # # noise = torch.randn(n_noise, C, D1, D2, device=self.device)
            
            # # x_noisy = x.squeeze() + noise * eps

            
            # # A = (vf(x_noisy)[0][-1] - vf(x)[0][-1].repeat(n_noise, 1, 1, 1)) / eps
            # # Bt = noise.view(n_noise, FD)
            # # At = A.view(n_noise, FD).to(self.device)

            # # BAt = torch.matmul(Bt.T, At)
            # # BBt = torch.matmul(Bt.T, Bt)
            
            # # J_approx = torch.linalg.solve(BBt, BAt).T

            Bt = []
            At = []
            fx = vf(x)[0][-1]

            for i in tqdm.tqdm(range(0, n_noise, batch_size)):
                b = min(batch_size, n_noise - i)
                noise = torch.randn(b, C, D1, D2, device=self.device)
            
                x_noisy = x.squeeze() + noise * eps
            
                A = (vf(x_noisy)[0][-1] - fx.repeat(b, 1, 1, 1))  #/ eps
                B_t = x_noisy.view(b, FD).to("cpu")
                A_t = A.view(b, FD)
                At.append(A_t)
                Bt.append(B_t)
            
            Bt = torch.concatenate(Bt, 0)
            At = torch.concatenate(At, 0)
                
            BAt = torch.matmul(Bt.T, At)
            BBt = torch.matmul(Bt.T, Bt)
            
            J_approx = torch.linalg.solve(BBt, BAt).T
            #J_approx = torch.autograd.functional.jacobian(vf, x.view((FD)))
            
            print(J_approx.shape)
            print(J_approx)
            
            return J_approx.to(self.device)
        else:
            raise NotImplementedError

    def test(self):
        pass
