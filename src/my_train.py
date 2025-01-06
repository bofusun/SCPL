import torch
import os
import pickle
import argparse
import numpy as np
import gym
import utils
import time
from arguments import parse_args
from env.wrappers import make_env
from algorithms.factory import make_agent
from logger import Logger
from video import VideoRecorder

def parse_args():
    parser = argparse.ArgumentParser()
    # environment
    parser.add_argument('--domain_name', default='walker')
    parser.add_argument('--task_name', default='walk')
    parser.add_argument('--workdir', default='523_save_grad_overlay')
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--algorithm', default='svea', type=str, choices=['sac', 'rad', 'curl', 'pad', 'soda', 'drq', 'svea', "sgsac", 'svea_value', 'svea_kl', \
                                                                          'svea_trans', 'svea_trans_conv', 'sac_trans', 'sac_trans_conv', 'svea_ave', \
                                                                          'my_svea_ave_shift', 'my_svea_ave_unshift', 'my_drq_shift', 'my_drq_unshift', \
                                                                          'my_svea_augshift', 'my_drq_augshift', 'my_svea_ave_shift_value', 'my_svea_ave_shift_kl', \
                                                                          'my_svea_ave_shift_trans', 'my_svea_ave_shift_trans_augconv', 'my_svea_ave_shift_trans_together', \
                                                                          'my_svea_ave_shift_trans_augconv_together'])
    parser.add_argument('--eval_mode', default='video_easy', type=str, choices=['train', 'color_easy', 'color_hard', 'video_easy', 'video_hard', 'distracting_cs', 'none'])
    args = parser.parse_args()
    return args


class Param:
    def __init__(self, domain_name, task_name, image_size, action_repeat, frame_stack, episode_length, eval_mode, replay_buffer_capacity, algorithm, \
                 init_steps, train_steps, discount, batch_size, hidden_dim, actor_lr, actor_beta, actor_log_std_min, actor_log_std_max, actor_update_freq, \
                 critic_lr, critic_beta, critic_tau, critic_target_update_freq, num_shared_layers, num_head_layers, num_filters, encoder_feature_dim, \
                 encoder_lr, encoder_tau, encoder_stride, decoder_type, decoder_lr, decoder_update_freq, decoder_weight_lambda, init_temperature, alpha_lr, \
                 alpha_beta, cfpred_lr, rsd_nstep, rsd_discount, cf_temp, cf_output_dim, elite_output_dim, output_mode, omega_num_sample, omega_output_mode, \
                 soda_batch_size, soda_tau, svea_alpha, svea_beta, save_freq, eval_freq, eval_episodes, distracting_cs_intensity, seed, work_dir, save_tb, \
                 save_model, save_buffer, save_video, render, port, device):
        self.domain_name = domain_name
        self.task_name = task_name
        self.image_size = image_size
        self.action_repeat = action_repeat
        self.frame_stack = frame_stack
        self.episode_length = episode_length
        self.eval_mode = eval_mode
        self.replay_buffer_capacity = replay_buffer_capacity
        self.algorithm = algorithm
        self.init_steps = init_steps
        self.train_steps = train_steps
        self.discount = discount
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.actor_lr = actor_lr
        self.actor_beta = actor_beta
        self.actor_log_std_min = actor_log_std_min
        self.actor_log_std_max = actor_log_std_max
        self.actor_update_freq = actor_update_freq
        self.critic_lr = critic_lr
        self.critic_beta = critic_beta
        self.critic_tau = critic_tau
        self.critic_target_update_freq = critic_target_update_freq
        self.num_shared_layers = num_shared_layers
        self.num_head_layers = num_head_layers
        self.num_filters = num_filters
        self.projection_dim = encoder_feature_dim
        self.encoder_lr = encoder_lr
        self.encoder_tau = encoder_tau
        self.encoder_stride = encoder_stride
        self.decoder_type = decoder_type
        self.decoder_lr = decoder_lr
        self.decoder_update_freq = decoder_update_freq
        self.decoder_weight_lambda = decoder_weight_lambda
        self.init_temperature = init_temperature
        self.alpha_lr = alpha_lr
        self.alpha_beta = alpha_beta
        self.cfpred_lr = cfpred_lr
        self.rsd_nstep = rsd_nstep
        self.rsd_discount = rsd_discount
        self.cf_temp = cf_temp
        self.cf_output_dim = cf_output_dim
        self.elite_output_dim = elite_output_dim
        self.output_mode = output_mode
        self.omega_num_sample = omega_num_sample
        self.omega_output_mode = omega_output_mode
        self.soda_batch_size = soda_batch_size
        self.soda_tau = soda_tau
        self.svea_alpha = svea_alpha
        self.svea_beta = svea_beta
        self.save_freq = save_freq
        self.eval_freq = eval_freq
        self.eval_episodes = eval_episodes
        self.distracting_cs_intensity = distracting_cs_intensity
        self.seed = seed
        self.work_dir = work_dir
        self.save_tb = save_tb
        self.save_model = save_model
        self.save_buffer = save_buffer
        self.save_video = save_video
        self.render = render
        self.port = port
        self.device = 0

class MODEL():
    def __init__(self, param, device):
        # 基础参数
        self.param = param
        self.domain_name = param.domain_name
        self.task_name = param.task_name
        self.image_size = param.image_size
        self.action_repeat = param.action_repeat
        self.frame_stack = param.frame_stack
        self.episode_length = param.episode_length
        self.eval_mode = param.eval_mode
        self.replay_buffer_capacity = param.replay_buffer_capacity
        self.algorithm = param.algorithm
        self.init_steps = param.init_steps
        self.train_steps = param.train_steps
        self.discount = param.discount
        self.batch_size = param.batch_size
        self.hidden_dim = param.hidden_dim
        self.actor_lr = param.actor_lr
        self.actor_beta = param.actor_beta
        self.actor_log_std_min = param.actor_log_std_min
        self.actor_log_std_max = param.actor_log_std_max
        self.actor_update_freq = param.actor_update_freq
        self.critic_lr = param.critic_lr
        self.critic_beta = param.critic_beta
        self.critic_tau = param.critic_tau
        self.critic_target_update_freq = param.critic_target_update_freq
        self.num_shared_layers = param.num_shared_layers
        self.num_head_layers = param.num_head_layers
        self.num_filters = param.num_filters
        self.encoder_feature_dim = param.projection_dim
        self.encoder_lr = param.encoder_lr
        self.encoder_tau = param.encoder_tau
        self.encoder_stride = param.encoder_stride
        self.decoder_type = param.decoder_type
        self.decoder_lr = param.decoder_lr
        self.decoder_update_freq = param.decoder_update_freq
        self.decoder_weight_lambda = param.decoder_weight_lambda
        self.init_temperature = param.init_temperature
        self.alpha_lr = param.alpha_lr
        self.alpha_beta = param.alpha_beta
        self.cfpred_lr = param.cfpred_lr
        self.rsd_nstep = param.rsd_nstep
        self.rsd_discount = param.rsd_discount
        self.cf_temp = param.cf_temp
        self.cf_output_dim = param.cf_output_dim
        self.elite_output_dim = param.elite_output_dim
        self.output_mode = param.output_mode
        self.omega_num_sample = param.omega_num_sample
        self.omega_output_mode = param.omega_output_mode
        self.soda_batch_size = param.soda_batch_size
        self.soda_tau = param.soda_tau
        self.svea_alpha = param.svea_alpha
        self.svea_beta = param.svea_beta
        self.save_freq = param.save_freq
        self.eval_freq = param.eval_freq
        self.eval_episodes = param.eval_episodes
        self.distracting_cs_intensity = param.distracting_cs_intensity
        self.seed = param.seed
        self.work_dir = os.path.join(param.work_dir, param.domain_name+'_'+param.task_name, param.algorithm, str(param.seed))
        self.save_tb = param.save_tb
        self.save_model = param.save_model
        self.save_buffer = param.save_buffer
        self.save_video = param.save_video
        self.render = param.render
        self.port = param.port
        self.device = device
        # 环境设置 
        self.train_env, self.test_env = self.set_env()
        # 配置保存地址
        utils.make_dir(self.work_dir)
        self.video_dir = utils.make_dir(os.path.join(self.work_dir, 'video'))
        self.model_dir = utils.make_dir(os.path.join(self.work_dir, 'model'))
        self.buffer_dir = utils.make_dir(os.path.join(self.work_dir, 'buffer'))
        self.video = VideoRecorder(self.video_dir if self.save_video else None, height=448, width=448)
        self.logger = Logger(self.work_dir)
        utils.write_info(param, os.path.join(self.work_dir, 'info.log'))
        self.param.device = self.device
        # replay buffer
        self.replay_buffer = utils.ReplayBuffer(
            obs_shape=self.train_env.observation_space.shape,
            action_shape=self.train_env.action_space.shape,
            capacity=self.train_steps,
            batch_size=self.batch_size,
            device=self.device
        )
        # 设置智能体
        self.agent = self.make_agent()
        # 训练测试奖励
        self.train_rewards = []
        self.test_rewards = []


    # 设置环境
    def set_env(self):
        train_env = make_env(
            domain_name=self.domain_name,
            task_name=self.task_name,
            seed=self.seed,
            episode_length=self.episode_length,
            action_repeat=self.action_repeat,
            image_size=self.image_size,
            mode='train'
        )
        test_env = make_env(
            domain_name=self.domain_name,
            task_name=self.task_name,
            seed=self.seed+42,
            episode_length=self.episode_length,
            action_repeat=self.action_repeat,
            image_size=self.image_size,
            mode=self.eval_mode,
            intensity=self.distracting_cs_intensity
        ) if self.eval_mode is not None else None
        return train_env, test_env

    def make_agent(self):
        cropped_obs_shape = (3*self.frame_stack, 84, 84)
        agent = make_agent(
            obs_shape=cropped_obs_shape,
            action_shape=self.train_env.action_space.shape,
            args=self.param
        )
        return agent

    def evaluate(self, env, step, eval_mode, test_env=False):
        episode_rewards = []
        for i in range(self.eval_episodes):
            print("episode:",i)
            obs = env.reset()
            self.video.init(enabled=(i==0))
            done = False
            episode_reward = 0
            print("reset")
            count = 0
            while not done:
                with utils.eval_mode(self.agent):
                    action = self.agent.select_action(obs)
                obs, reward, done, _ = env.step(action)
                self.video.record(env, eval_mode)
                episode_reward += reward
                count += 1
            if self.logger is not None:
                _test_env = '_test_env' if test_env else ''
                self.video.save(f'{step}{_test_env}.mp4')
                self.logger.log(f'eval/episode_reward{_test_env}', episode_reward, step)
            episode_rewards.append(episode_reward)

        average_episode_reward = np.mean(episode_rewards)
        self.test_rewards.append(average_episode_reward)

    def train(self):
        episode, episode_reward, done = 0, 0, True
        start_time = time.time()
        for step in range(self.train_steps):
            if done:
                if step > 0:
                    self.logger.log('train/duration', time.time() - start_time, step)
                    start_time = time.time()
                    self.logger.dump(step)
                # Evaluate agent periodically
                if episode % self.eval_freq == 0:
                    self.logger.log('eval/episode', episode, step)
                    # evaluate(env, agent, video, args.eval_episodes, L, step)
                    if self.test_env is not None:
                        self.evaluate(self.test_env, step, eval_mode=self.eval_mode, test_env=True)
                    # 保存记录
                    save_info = {"train_rewards": self.train_rewards , "test_rewards": self.test_rewards}
                    with open(os.path.join(self.model_dir, "record_" + str(step) + ".pkl"), "wb") as f:
                        pickle.dump(save_info, f, pickle.HIGHEST_PROTOCOL)   
                    self.logger.dump(step)
                # Save agent periodically
                if step % self.save_freq == 0:
                    if args.algorithm == "sgsac":
                        torch.save(self.agent.actor.state_dict(), os.path.join(self.model_dir, f"actor_{step}.pt"))
                        torch.save(self.agent.critic.state_dict(), os.path.join(self.model_dir, f"critic_{step}.pt"))
                        torch.save(self.agent.attribution_predictor.state_dict(), os.path.join(self.model_dir, f"attrib_predictor_{step}.pt"))
                    elif args.algorithm == "my_svea_ave_shift_trans_augconv" or args.algorithm == "my_svea_ave_shift_trans" :
                        torch.save(self.agent.actor.state_dict(), os.path.join(self.model_dir, f"actor_{step}.pt"))
                        torch.save(self.agent.critic.state_dict(), os.path.join(self.model_dir, f"critic_{step}.pt"))
                        torch.save(self.agent.transition_model.state_dict(), os.path.join(self.model_dir, f"transition_model_{step}.pt"))
                    else:
                        torch.save(self.agent.actor.state_dict(), os.path.join(self.model_dir, f"actor_{step}.pt"))
                        torch.save(self.agent.critic.state_dict(), os.path.join(self.model_dir, f"critic_{step}.pt"))
                # 记录训练信息
                self.logger.log('train/episode_reward', episode_reward, step)
                self.train_rewards.append(episode_reward)
                # 重启训练
                obs = self.train_env.reset()
                done = False
                episode_reward = 0
                episode_step = 0
                episode += 1
                reward = 0
                # 更新训练信息
                self.logger.log('train/episode', episode, step)


            # 最开始训练随机采样
            if step < self.init_steps:
                action = self.train_env.action_space.sample()
            else:
                with utils.eval_mode(self.agent):
                    action = self.agent.sample_action(obs)

            # 采样超过initsteps，更新训练
            if step >= self.init_steps:
                num_updates = self.init_steps if step == self.init_steps else 1
                # 如果刚收集满，则训练 init_steps 次，否则训练1次
                for _ in range(num_updates):
                    self.agent.update(self.replay_buffer, self.logger, step)

            # 记录奖励
            next_obs, reward, done, _ = self.train_env.step(action)
            # allow infinit bootstrap
            done_bool = 0 if episode_step + 1 == self.train_env._max_episode_steps else float(done)
            episode_reward += reward
            # 收集新的数据
            self.replay_buffer.add(obs, action,  reward, next_obs, done_bool)
            # 更新obs
            obs = next_obs
            episode_step += 1    

if __name__ == "__main__":
    args = parse_args()
    ##############################################
    # 环境变量
    domain_name = args.domain_name # 'walker'
    task_name = args.task_name     # 'walk'
    # image_size = 84
    action_repeat_dict = {"walker":2, "finger":2, "cartpole":8, "cheetah":4, "ball_in_cup":4}
    action_repeat = action_repeat_dict[domain_name]
    frame_stack = 3
    episode_length = 1000
    # 测试类型 {'train', 'color_easy', 'color_hard', 'video_easy', 'video_hard', 'distracting_cs', 'none'}
    eval_mode = args.eval_mode     # 'video_easy'
    ###############################################
    # 训练信息
    replay_buffer_capacity = 100000
    # 使用算法 {'sac', 'rad', 'curl', 'pad', 'soda', 'drq', 'svea'}
    algorithm = args.algorithm     # 'svea'
    if algorithm in {'rad', 'curl', 'pad', 'soda'}:
        image_size = 100
    else:
        image_size = 84
    init_steps = 1000
    train_steps = 500000
    discount = 0.99
    batch_size = 128
    hidden_dim = 1024
    ################################################
    # actor
    actor_lr = 1e-3
    actor_beta = 0.9
    actor_log_std_min = -10
    actor_log_std_max = 2
    actor_update_freq = 2
    ################################################
    # critic
    critic_lr = 1e-3
    critic_beta = 0.9
    critic_tau = 0.01
    critic_target_update_freq = 2
    ################################################
    # encoder
    num_shared_layers = 11
    num_head_layers = 0
    num_filters = 32
    encoder_feature_dim = 100
    encoder_lr = 1e-3
    encoder_tau = 0.05
    encoder_stride = 1
    ################################################
    # decoder
    # ['pixel', 'identity', 'contrastive', 'reward', 'inverse', 'reconstruction']
    decoder_type = 'identity'
    decoder_lr = 1e-3
    decoder_update_freq = 1
    decoder_weight_lambda = 1e-7
    ################################################
    # 最大熵
    init_temperature = 0.1
    alpha_lr = 1e-4
    alpha_beta = 0.5
    #################################################
    # 算法特有参数
    # crespt
    cfpred_lr=3e-4
    rsd_nstep=2
    rsd_discount=0.9
    cf_temp=0.01
    cf_output_dim=7
    elite_output_dim=5
    # choices=['min', 'max', 'random']
    output_mode='min'
    omega_num_sample=32
    # choices=[None, 'min_mu', 'min_all', 'sample']
    omega_output_mode='min_mu'
    # soda算法
    soda_batch_size = 256
    soda_tau = 0.005
    # svea算法
    svea_alpha = 0.5
    svea_beta = 0.5
    ######################################
    # 设置信息
    save_freq = 100000
    eval_freq = 50
    eval_episodes = 20 # 20
    # 干扰程度 {0., 0.025, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5}
    distracting_cs_intensity = 0
    seed = args.seed
    work_dir = "./" + args.workdir
    save_tb = True
    save_model = False
    save_buffer = False
    save_video = True
    render = False
    port = 2000
    ####################################################
    # 设置种子
    utils.set_seed_everywhere(seed)
    # 设备
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("device", device)
    ####################################################
    # 设置参数  
    param = Param(domain_name, task_name, image_size, action_repeat, frame_stack, episode_length, eval_mode, replay_buffer_capacity, algorithm, \
                 init_steps, train_steps, discount, batch_size, hidden_dim, actor_lr, actor_beta, actor_log_std_min, actor_log_std_max, actor_update_freq, \
                 critic_lr, critic_beta, critic_tau, critic_target_update_freq, num_shared_layers, num_head_layers, num_filters, encoder_feature_dim, \
                 encoder_lr, encoder_tau, encoder_stride, decoder_type, decoder_lr, decoder_update_freq, decoder_weight_lambda, init_temperature, alpha_lr, \
                 alpha_beta, cfpred_lr, rsd_nstep, rsd_discount, cf_temp, cf_output_dim, elite_output_dim, output_mode, omega_num_sample, omega_output_mode, \
                 soda_batch_size, soda_tau, svea_alpha, svea_beta, save_freq, eval_freq, eval_episodes, distracting_cs_intensity, seed, work_dir, save_tb, \
                 save_model, save_buffer, save_video, render, port, device)    
    my_model = MODEL(param, device)
    my_model.train()



