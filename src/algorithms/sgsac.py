import os
from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
import utils
import augmentations
import algorithms.modules as m
from algorithms.sac import SAC

from .rl_utils import (
    compute_attribution,
    compute_attribution_mask,
    make_attribution_pred_grid,
    make_obs_grid,
    make_obs_grad_grid,
)
import random


class SGSAC(SAC):
    def __init__(self, obs_shape, action_shape, args):
        super().__init__(obs_shape, action_shape, args)
        self.args = args

        self.attribution_predictor = m.AttributionPredictor(action_shape[0],self.critic.encoder).to(args.device) # .cuda()
        self.quantile = 0.9
        self.aux_update_freq = 2
        self.consistency = 1

        self.aux_optimizer = torch.optim.Adam(
            self.attribution_predictor.parameters(),
            lr=3e-4,
            betas=(0.9, 0.999),
        )

        tb_dir = os.path.join(
            args.work_dir,
            args.domain_name + "_" + args.task_name,
            args.algorithm,
            str(args.seed),
            "tensorboard",
        )
        self.writer = SummaryWriter(tb_dir)
        
        self.buffer_dir = os.path.join(args.work_dir,
            args.domain_name + "_" + args.task_name,
            args.algorithm,
            str(args.seed),
			'buffer')

    def update_critic(self, obs, action, reward, next_obs, not_done, L=None, step=None):
        with torch.no_grad():
            _, policy_action, log_pi, _ = self.actor(next_obs)
            target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
            target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (not_done * self.discount * target_V)

        current_Q1, current_Q2 = self.critic(obs, action)

        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q
        )
        if self.consistency:
            obs_grad = compute_attribution(self.critic,obs,action.detach())
            mask = compute_attribution_mask(obs_grad,self.quantile)
            masked_obs = obs*mask
            masked_obs[mask<1] = random.uniform(obs.view(-1).min(),obs.view(-1).max()) 
            masked_Q1,masked_Q2 = self.critic(masked_obs,action)
            critic_loss += 0.5 *(F.mse_loss(current_Q1,masked_Q1) + F.mse_loss(current_Q2,masked_Q2))
        if L is not None:
            L.log("train_critic/loss", critic_loss, step)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def update_aux(self, obs, action, obs_grad, mask, step=None, L=None):
        mask = compute_attribution_mask(obs_grad,self.quantile)
        s_prime = augmentations.attribution_augmentation(obs.clone(), mask.float())

        s_tilde = augmentations.random_overlay(obs.clone())
        self.aux_optimizer.zero_grad()
        pred_attrib, aux_loss = self.compute_attribution_loss(s_tilde,action, mask)
        aux_loss.backward()
        self.aux_optimizer.step()

        if L is not None:
            L.log("train/aux_loss", aux_loss, step)

        # if step % 10000 == 0:
        #     self.log_tensorboard(obs, action, step, prefix="original")
        #     self.log_tensorboard(s_tilde, action, step, prefix="augmented")
        #     self.log_tensorboard(s_prime, action, step, prefix="super_augmented")
        if step % 10000 == 0:
            self.save_image(obs, action, step, prefix="original")
            self.save_image(s_tilde, action, step, prefix="augmented")

    def log_tensorboard(self, obs, action, step, prefix="original"):
        obs_grad = compute_attribution(self.critic, obs, action.detach())
        mask = compute_attribution_mask(obs_grad, quantile=self.quantile)
        attrib = self.attribution_predictor(obs.detach(),action.detach())
        grid = make_obs_grid(obs)
        self.writer.add_image(prefix + "/observation", grid, global_step=step)
        grad_grid = make_obs_grad_grid(obs_grad.data.abs())
        self.writer.add_image(prefix + "/attributions", grad_grid, global_step=step)
        mask = torch.sigmoid(attrib)
        mask = (mask > 0.5).float()
        masked_obs = make_obs_grid(obs * mask)
        self.writer.add_image(prefix + "/masked_obs{}", masked_obs, global_step=step)
        attrib_grid = make_obs_grad_grid(torch.sigmoid(attrib))
        self.writer.add_image(
            prefix + "/predicted_attrib", attrib_grid, global_step=step
        )
        for q in [0.95, 0.975, 0.9, 0.995, 0.999]:
            mask = compute_attribution_mask(obs_grad, quantile=q)
            masked_obs = make_obs_grid(obs * mask)
            self.writer.add_image(
                prefix + "/attrib_q{}".format(q), masked_obs, global_step=step
            )

    def compute_attribution_loss(self, obs, action, mask):
        mask = mask.float()
        attrib = self.attribution_predictor(obs.detach(),action.detach())
        aux_loss = F.binary_cross_entropy_with_logits(attrib, mask.detach())
        return attrib, aux_loss


    def update(self, replay_buffer, L, step):
        obs, action, reward, next_obs, not_done = replay_buffer.sample_drq()

        self.update_critic(obs, action, reward, next_obs, not_done, L, step)
        obs_grad = compute_attribution(self.critic, obs, action.detach())
        mask = compute_attribution_mask(obs_grad, quantile=self.quantile)

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs, L, step)

        if step % self.critic_target_update_freq == 0:
            self.soft_update_critic_target()

        if step % self.aux_update_freq == 0:
            self.update_aux(obs, action, obs_grad, mask, step, L)


    def save_image(self, obs, action, step, prefix="original"):
		# 获取各个梯度图
        grid = make_obs_grid(obs)
        critic_obs_grad = compute_attribution(self.critic, obs, action.detach())
        critic_grad_grid = make_obs_grad_grid(critic_obs_grad.data.abs())
        # 获取遮挡图
        critic_obs_grad_mask = compute_attribution_mask(critic_obs_grad, quantile=0.9)
        critic_masked_obs = make_obs_grid(obs * critic_obs_grad_mask)	
        # 添加到tensorboard
        self.writer.add_image(prefix + "/observation", grid, global_step=step)
        self.writer.add_image(prefix + "/critic_grad", critic_grad_grid, global_step=step)
        self.writer.add_image(prefix + "/critic_grad_mask", critic_masked_obs, global_step=step)
		# 保存图片
        save_image(grid,os.path.join(self.buffer_dir, prefix +"_obs_"+str(step)+".jpg"))
        save_image(critic_grad_grid,os.path.join(self.buffer_dir, prefix +"_critic_grad_"+str(step)+".jpg"))
        save_image(critic_masked_obs,os.path.join(self.buffer_dir, prefix +"_critic_grad_mask_"+str(step)+".jpg"))
        for i in range(action.shape[1]):
            # 获取各个梯度图
            actor_obs_grad = compute_attribution(self.actor, obs, target=i)
            actor_grad_grid = make_obs_grad_grid(actor_obs_grad.data.abs())
            # 获取遮挡图
            actor_obs_grad_mask = compute_attribution_mask(actor_obs_grad, quantile=0.9)
            actor_masked_obs = make_obs_grid(obs * actor_obs_grad_mask)
            # 添加到tensorboard
            self.writer.add_image(prefix + "/actor_grad_"+str(i), actor_grad_grid, global_step=step)
            self.writer.add_image(prefix + "/actor_grad_mask_"+str(i), actor_masked_obs, global_step=step)
            # 保存图片
            save_image(actor_grad_grid,os.path.join(self.buffer_dir, prefix +"_actor_grad_"+str(step)+"_"+str(i)+".jpg"))
            save_image(actor_masked_obs,os.path.join(self.buffer_dir, prefix +"_actor_grad_mask_"+str(step)+"_"+str(i)+".jpg"))