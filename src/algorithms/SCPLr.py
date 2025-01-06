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
from torch.distributions import MultivariateNormal

class SCPLr(SAC):
    def __init__(self, obs_shape, action_shape, args):
        super().__init__(obs_shape, action_shape, args)
        self.args = args

        self.transition_model = m.My_TransitionModel(args.projection_dim, action_shape, 512).to(args.device)

        self.combined_transition_model = m.Combined_Transition_Model(self.critic.encoder, self.transition_model)
        
        self.attribution_predictor = m.AttributionPredictor(action_shape[0],self.critic.encoder).to(args.device) # .cuda()
        self.quantile = 0.9
        self.aux_update_freq = 2
        self.consistency = 1

        # decoder是奖励与世界模型
        self.decoder_optimizer = torch.optim.Adam(
            list(self.transition_model.parameters()), 
            lr=3e-4, 
            betas=(0.9, 0.999),
            )

        self.encoder_optimizer = torch.optim.Adam(
            self.critic.encoder.parameters(),
            lr=3e-4,
            betas=(0.9, 0.999),
        )

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

    def update_critic(self, obs, aug_obs, action, reward, next_obs, not_done, L=None, step=None):
        with torch.no_grad():
            _, policy_action, log_pi, _ = self.actor(next_obs)
            target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
            target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (not_done * self.discount * target_V)

        obs = utils.cat(obs, aug_obs)
        action = utils.cat(action, action)
        target_Q = utils.cat(target_Q, target_Q)
        
        current_Q1, current_Q2 = self.critic(obs, action)
        critic_loss = (F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q))

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

    def update_actor_and_alpha(self, obs, aug_obs, L=None, step=None, update_alpha=True):
        mu, pi, log_pi, log_std = self.actor(obs, detach=True)
        actor_Q1, actor_Q2 = self.critic(obs, pi, detach=True)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()

        if L is not None:
            L.log('train_actor/loss', actor_loss, step)
            entropy = 0.5 * log_std.shape[1] * (1.0 + np.log(2 * np.pi)
												) + log_std.sum(dim=-1)

        # 计算两个分布
        std = log_std.exp()
        cov_mat = torch.diag_embed(std).detach()
        target_distribution = MultivariateNormal(mu, cov_mat)
        aug_mu, aug_pi, aug_log_pi, aug_log_std = self.actor(aug_obs, detach=True)
        aug_std = aug_log_std.exp()
        aug_cov_mat = torch.diag_embed(aug_std)
        current_distribution = MultivariateNormal(aug_mu, aug_cov_mat)
        kl_loss = torch.distributions.kl_divergence(target_distribution, current_distribution).mean()
        if L is not None:
            L.log('train/kl_loss', kl_loss, step)
        # 添加kl损失
        actor_loss += kl_loss
		
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if update_alpha:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (self.alpha * (-log_pi - self.target_entropy).detach()).mean()

            if L is not None:
                L.log('train_alpha/loss', alpha_loss, step)
                L.log('train_alpha/value', self.alpha, step)

            alpha_loss.backward()
            self.log_alpha_optimizer.step()

    def update_transition_reward_model(self, obs, aug_obs, action, next_obs, aug_next_obs, reward, L, step):
        # 获取尺寸
        n, c, h, w = obs.size()
        # 分解并增强数据
        augment_obs = aug_obs
        augment_next_obs = aug_next_obs
        # 合并数据
        obs = torch.cat([obs, augment_obs], axis=0)
        next_obs = torch.cat([next_obs, augment_next_obs], axis=0)
        action = torch.cat([action, action], axis=0)
        reward = torch.cat([reward, reward], axis=0)
		# 计算真实下一状态的动作
        next_mu, next_policy_action, next_log_pi, _ = self.actor(next_obs)
        next_obs_grad = compute_attribution(self.critic, next_obs, next_mu.detach())
        next_mask = compute_attribution_mask(next_obs_grad, quantile=0.9)
        next_mask = next_mask.float()

        # 开始过encoder
        h = self.critic.encoder(obs)
        pred_next_latent_mu, pred_next_latent_sigma, pred_next_reward = self.transition_model(torch.cat([h, action], dim=1))
        if pred_next_latent_sigma is None:
            pred_next_latent_sigma = torch.ones_like(pred_next_latent_mu)

        # 下一状态预测
        next_h = self.critic.encoder(next_obs)
        diff = (pred_next_latent_mu - next_h.detach()) / pred_next_latent_sigma
        predict_loss = torch.mean(0.5 * diff.pow(2) + torch.log(pred_next_latent_sigma))
        
        # 下一状态的奖励输入到奖励模型，比较奖励值
        reward_loss = F.mse_loss(pred_next_reward, reward)
        total_loss = predict_loss + reward_loss


        L.log('train_ae/reward_loss', reward_loss, step)
        L.log('train_ae/transition_loss', total_loss, step)
        return total_loss

    def update_aux(self, obs, aug_obs, action, obs_grad, mask, step=None, L=None):
        # mask = compute_attribution_mask(obs_grad,self.quantile)
        # s_prime = augmentations.attribution_augmentation(obs.clone(), mask.float())

        s_tilde = aug_obs
        # self.aux_optimizer.zero_grad()
        # pred_attrib, aux_loss = self.compute_attribution_loss(s_tilde,action, mask)
        # aux_loss.backward()
        # self.aux_optimizer.step()

        # if L is not None:
        #     L.log("train/aux_loss", aux_loss, step)

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

        aug_obs = augmentations.random_overlay(obs.clone())
        # aug_obs, aug_next_obs = self.get_augmentation(obs, next_obs)

        self.update_critic(obs, aug_obs, action, reward, next_obs, not_done, L, step)
        obs_grad = compute_attribution(self.critic, obs, action.detach())
        mask = compute_attribution_mask(obs_grad, quantile=self.quantile)

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs, aug_obs, L, step)

        if step % self.critic_target_update_freq == 0:
            self.soft_update_critic_target()

        if step % self.aux_update_freq == 0:
            transition_reward_loss = self.update_transition_reward_model(obs, aug_obs, action, next_obs, aug_next_obs, reward, L, step)
            self.encoder_optimizer.zero_grad()
            self.decoder_optimizer.zero_grad()
            transition_reward_loss.backward()
            self.encoder_optimizer.step()
            self.decoder_optimizer.step()
            self.update_aux(obs, aug_obs, action, obs_grad, mask, step, L)

    def get_augmentation(self, obs, next_obs):
		# 获取尺寸
        n, c, h, w = obs.size()
        # 合并并增强数据
        augmented_obs = torch.cat([obs, next_obs], dim=1)
        augmented_obs = augmentations.random_overlay(augmented_obs.clone())
		# 分解并增强数据
        augment_obs = augmented_obs[:, :c, :, :]
        augment_next_obs = augmented_obs[:, c:, :, :]
        return augment_obs, augment_next_obs
        
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