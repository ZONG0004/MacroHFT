import pathlib
import sys
import random
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import pickle
import os
from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.filterwarnings("ignore")

ROOT = str(pathlib.Path(__file__).resolve().parents[3])
sys.path.append(ROOT)
sys.path.insert(0, ".")

from MacroHFT.model.net import *
from MacroHFT.env.low_level_env import Testing_Env, Training_Env
from MacroHFT.RL.util.utili import get_ada, get_epsilon, LinearDecaySchedule
from MacroHFT.RL.util.replay_buffer import ReplayBuffer

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["F_ENABLE_ONEDNN_OPTS"] = "0"

parser = argparse.ArgumentParser()
parser.add_argument("--buffer_size",type=int,default=1000000,)
parser.add_argument("--dataset",type=str,default="ETHUSDT")
parser.add_argument("--q_value_memorize_freq",type=int, default=10,)
parser.add_argument("--batch_size",type=int,default=512)
parser.add_argument("--eval_update_freq",type=int,default=100)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--epsilon_start",type=float,default=0.5)
parser.add_argument("--epsilon_end",type=float,default=0.1)
parser.add_argument("--decay_length",type=int,default=5)
parser.add_argument("--update_times",type=int,default=10)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--tau", type=float, default=0.005)
parser.add_argument("--transcation_cost",type=float,default=2.0 / 10000)
parser.add_argument("--back_time_length",type=int,default=1)
parser.add_argument("--seed",type=int,default=12345)
parser.add_argument("--n_step",type=int,default=1)
parser.add_argument("--epoch_number",type=int,default=15)
parser.add_argument("--label",type=str,default="label_1")
parser.add_argument("--clf",type=str,default="slope")
parser.add_argument("--alpha",type=float,default="0")
parser.add_argument("--device",type=str,default="cuda:0")


def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def calculate_alpha(diff, k):
    alpha = 16 * (1 - torch.exp(-k * diff))
    return torch.clip(alpha, 0, 16)


class DQN(object):
    def __init__(self, args):  # 定义DQN的一系列属性
        self.seed = args.seed
        seed_torch(self.seed)
        if torch.cuda.is_available():
            self.device = torch.device(args.device)
        else:
            self.device = torch.device("cpu")
        self.result_path = os.path.join("./result/low_level", 
                                        '{}'.format(args.dataset), '{}'.format(args.clf), str(int(args.alpha)), args.label)
        self.label = int(args.label.split('_')[1])
        self.model_path = os.path.join(self.result_path,
                                       "seed_{}".format(self.seed))
        self.train_data_path = os.path.join(ROOT,
                                        "data", args.dataset, "train")
        self.val_data_path = os.path.join(ROOT,
                                        "data", args.dataset, "val")
        self.test_data_path = os.path.join(ROOT,
                                        "data", args.dataset, "test")
        if args.clf == 'slope':
            with open(os.path.join(self.train_data_path, 'slope_labels.pkl'), 'rb') as file:
                self.train_index = pickle.load(file)
            with open(os.path.join(self.val_data_path, 'slope_labels.pkl'), 'rb') as file:
                self.val_index = pickle.load(file)
            with open(os.path.join(self.test_data_path, 'slope_labels.pkl'), 'rb') as file:
                self.test_index = pickle.load(file)
        elif args.clf == 'vol':
            with open(os.path.join(self.train_data_path, 'vol_labels.pkl'), 'rb') as file:
                self.train_index = pickle.load(file)
            with open(os.path.join(self.val_data_path, 'vol_labels.pkl'), 'rb') as file:
                self.val_index = pickle.load(file)
            with open(os.path.join(self.test_data_path, 'vol_labels.pkl'), 'rb') as file:
                self.test_index = pickle.load(file)


        self.dataset=args.dataset
        self.clf = args.clf
        if "BTC" in self.dataset:
            self.max_holding_number=0.01
        elif "ETH" in self.dataset:
            self.max_holding_number=0.2
        elif "DOT" in self.dataset:
            self.max_holding_number=10
        elif "LTC" in self.dataset:
            self.max_holding_number=10
        else:
            raise Exception ("we do not support other dataset yet")
        self.epoch_number = args.epoch_number
        
        self.log_path = os.path.join(self.model_path, "log")
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        self.writer = SummaryWriter(self.log_path)
        self.update_counter = 0
        self.q_value_memorize_freq = args.q_value_memorize_freq

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        self.tech_indicator_list = np.load('./data/feature_list/single_features.npy', allow_pickle=True).tolist()
        self.tech_indicator_list_trend = np.load('./data/feature_list/trend_features.npy', allow_pickle=True).tolist()

        self.transcation_cost = args.transcation_cost
        self.back_time_length = args.back_time_length
        self.n_action = 2
        self.n_state_1 = len(self.tech_indicator_list)
        self.n_state_2 = len(self.tech_indicator_list_trend)
        self.eval_net, self.target_net = subagent(
            self.n_state_1, self.n_state_2, self.n_action, 64).to(self.device), subagent(
                self.n_state_1, self.n_state_2, self.n_action,
                64).to(self.device)
        self.hardupdate()
        self.update_times = args.update_times
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(),
                                          lr=args.lr)
        self.loss_func = nn.MSELoss()
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.tau = args.tau
        self.n_step = args.n_step
        self.eval_update_freq = args.eval_update_freq
        self.buffer_size = args.buffer_size
        self.epsilon_start = args.epsilon_start
        self.epsilon_end = args.epsilon_end
        self.decay_length = args.decay_length
        self.epsilon_scheduler = LinearDecaySchedule(start_epsilon=self.epsilon_start, end_epsilon=self.epsilon_end, decay_length=self.decay_length)
        self.epsilon = args.epsilon_start

    def update(self, replay_buffer):
        self.eval_net.train()
        batch, _, _ = replay_buffer.sample()
        batch = {k: v.to(self.device) for k, v in batch.items()}
        a_argmax = self.eval_net(batch['next_state'], batch['next_state_trend'], batch['next_previous_action']).argmax(dim=-1, keepdim=True)
        q_target = batch['reward'] + self.gamma * (1 - batch['terminal']) * self.target_net(batch['next_state'], batch['next_state_trend'], 
                                                    batch['next_previous_action']).gather(-1, a_argmax).squeeze(-1)

        q_distribution = self.eval_net(batch['state'], batch['state_trend'], batch['previous_action'])
        q_current = q_distribution.gather(-1, batch['action']).squeeze(-1)

        td_error = self.loss_func(q_current, q_target)

        demonstration = batch['demo_action']
        KL_loss = F.kl_div(
            (q_distribution.softmax(dim=-1) + 1e-8).log(),
            (demonstration.softmax(dim=-1) + 1e-8),
            reduction="batchmean",
        )

        alpha = args.alpha
        loss = td_error + alpha * KL_loss
        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.eval_net.parameters(), 1)
        self.optimizer.step()
        for param, target_param in zip(self.eval_net.parameters(), self.target_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        self.update_counter += 1
        self.eval_net.eval()
        return td_error.cpu(), KL_loss.cpu(), torch.mean(
            q_current.cpu()), torch.mean(q_target.cpu())

    def hardupdate(self):
        self.target_net.load_state_dict(self.eval_net.state_dict())

    def act(self, state, state_trend, info):
        x1 = torch.FloatTensor(state).to(self.device)
        x2 = torch.FloatTensor(state_trend).to(self.device)
        previous_action = torch.unsqueeze(
            torch.tensor(info["previous_action"]).long().to(self.device),
            0).to(self.device)
        if np.random.uniform() < (1-self.epsilon):
            actions_value = self.eval_net(x1, x2, previous_action)
            action = torch.max(actions_value, 1)[1].data.cpu().numpy()
            action = action[0]
        else:
            action_choice = [0,1]
            action = random.choice(action_choice)
        return action

    def act_test(self, state, state_trend, info):
        x1 = torch.FloatTensor(state).to(self.device)
        x2 = torch.FloatTensor(state_trend).to(self.device)
        previous_action = torch.unsqueeze(
            torch.tensor(info["previous_action"]).long(), 0).to(self.device)
        actions_value = self.eval_net(x1, x2, previous_action)
        action = torch.max(actions_value, 1)[1].data.cpu().numpy()
        action = action[0]
        return action

    def train(self):
        epoch_return_rate_train_list = []
        epoch_final_balance_train_list = []
        epoch_required_money_train_list = []
        epoch_reward_sum_train_list = []
        df_list = self.train_index[self.label]
        df_number=int(len(df_list))       
        step_counter = 0
        episode_counter = 0
        epoch_counter = 0        
        self.replay_buffer = ReplayBuffer(args, self.n_state_1, self.n_state_2, self.n_action)   
        best_return_rate = -float('inf')
        best_model = None
        for sample in range(self.epoch_number):
            print('epoch ', epoch_counter + 1)
            random_list = self.train_index[self.label]
            random.shuffle(random_list)
            random_position_list = random.choices(range(self.n_action), k=df_number)
            print(random_list)
            
            for i in range(df_number):
                df_index = random_list[i]
                print("training with df", df_index)
                self.df = pd.read_feather(
                    os.path.join(self.train_data_path, "df_{}.feather".format(df_index)))
                self.eval_net.eval()
                               
                train_env = Training_Env(
                        df=self.df,
                        tech_indicator_list=self.tech_indicator_list,
                        tech_indicator_list_trend=self.tech_indicator_list_trend,
                        transcation_cost=self.transcation_cost,
                        back_time_length=self.back_time_length,
                        max_holding_number=self.max_holding_number,
                        initial_action=random_position_list[i],
                        alpha = 0)
                s, s2, info = train_env.reset()
                episode_reward_sum = 0
                
                while True:
                    a = self.act(s, s2, info)
                    s_, s2_, r, done, info_ = train_env.step(a)
                    self.replay_buffer.store_transition(s, s2, info['previous_action'], info['q_value'], a, r, s_, s2_, info_['previous_action'],
                                    info_['q_value'], done)
                    episode_reward_sum += r

                    s, s2, info = s_, s2_, info_
                    step_counter += 1
                    if step_counter % self.eval_update_freq == 0 and step_counter > (
                            self.batch_size + self.n_step):
                        for i in range(self.update_times):
                            td_error, KL_loss, q_eval, q_target = self.update(self.replay_buffer)
                            if self.update_counter % self.q_value_memorize_freq == 1:
                                self.writer.add_scalar(
                                    tag="td_error",
                                    scalar_value=td_error,
                                    global_step=self.update_counter,
                                    walltime=None)
                                self.writer.add_scalar(
                                    tag="KL_loss",
                                    scalar_value=KL_loss,
                                    global_step=self.update_counter,
                                    walltime=None)
                                self.writer.add_scalar(
                                    tag="q_eval",
                                    scalar_value=q_eval,
                                    global_step=self.update_counter,
                                    walltime=None)
                                self.writer.add_scalar(
                                    tag="q_target",
                                    scalar_value=q_target,
                                    global_step=self.update_counter,
                                    walltime=None)
                    if done:
                        break
                episode_counter += 1
                final_balance, required_money = train_env.final_balance, train_env.required_money
                self.writer.add_scalar(tag="return_rate_train",
                                   scalar_value=final_balance / (required_money),
                                   global_step=episode_counter,
                                   walltime=None)
                self.writer.add_scalar(tag="final_balance_train",
                                    scalar_value=final_balance,
                                    global_step=episode_counter,
                                    walltime=None)
                self.writer.add_scalar(tag="required_money_train",
                                    scalar_value=required_money,
                                    global_step=episode_counter,
                                    walltime=None)
                self.writer.add_scalar(tag="reward_sum_train",
                                    scalar_value=episode_reward_sum,
                                    global_step=episode_counter,
                                    walltime=None)
                epoch_return_rate_train_list.append(final_balance / (required_money))
                epoch_final_balance_train_list.append(final_balance)
                epoch_required_money_train_list.append(required_money)
                epoch_reward_sum_train_list.append(episode_reward_sum)
                

            epoch_counter += 1
            self.epsilon = self.epsilon_scheduler.get_epsilon(epoch_counter)
            mean_return_rate_train = np.mean(epoch_return_rate_train_list)
            mean_final_balance_train = np.mean(epoch_final_balance_train_list)
            mean_required_money_train = np.mean(epoch_required_money_train_list)
            mean_reward_sum_train = np.mean(epoch_reward_sum_train_list)
            self.writer.add_scalar(
                    tag="epoch_return_rate_train",
                    scalar_value=mean_return_rate_train,
                    global_step=epoch_counter,
                    walltime=None,
                )
            self.writer.add_scalar(
                tag="epoch_final_balance_train",
                scalar_value=mean_final_balance_train,
                global_step=epoch_counter,
                walltime=None,
                )
            self.writer.add_scalar(
                tag="epoch_required_money_train",
                scalar_value=mean_required_money_train,
                global_step=epoch_counter,
                walltime=None,
                )
            self.writer.add_scalar(
                tag="epoch_reward_sum_train",
                scalar_value=mean_reward_sum_train,
                global_step=epoch_counter,
                walltime=None,
                )
            epoch_path = os.path.join(self.model_path,
                                        "epoch_{}".format(epoch_counter))
            if not os.path.exists(epoch_path):
                os.makedirs(epoch_path)
            torch.save(self.eval_net.state_dict(),
                        os.path.join(epoch_path, "trained_model.pkl"))
            val_path = os.path.join(epoch_path, "val")
            if not os.path.exists(val_path):
                    os.makedirs(val_path)
            return_rate_0 = self.val_cluster(epoch_path, val_path, 0)
            return_rate_1 = self.val_cluster(epoch_path, val_path, 1)
            return_rate_eval = (return_rate_0 + return_rate_1) / 2
            if return_rate_eval > best_return_rate:
                best_return_rate = return_rate_eval
                best_model = self.eval_net.state_dict()
            epoch_return_rate_train_list = []
            epoch_final_balance_train_list = []
            epoch_required_money_train_list = []
            epoch_reward_sum_train_list = []
        best_model_path = os.path.join("./result/low_level", 
                                        '{}'.format(self.dataset), '{}'.format(self.clf), self.label, 'best_model.pkl')
        torch.save(best_model.state_dict(), best_model_path)


    def val_cluster(self, epoch_path, save_path, initial_action):
        self.eval_net.load_state_dict(
            torch.load(os.path.join(epoch_path, "trained_model.pkl")))
        self.eval_net.eval()
        df_list = self.val_index[self.label]
        df_number=int(len(df_list)) 
        action_list = []
        reward_list = []
        final_balance_list = []
        required_money_list = []
        commission_fee_list = []
        for i in range(df_number):
            print("validating on df", df_list[i])
            self.df = pd.read_feather(
                os.path.join(self.val_data_path, "df_{}.feather".format(df_list[i])))
            
            val_env = Testing_Env(
                    df=self.df,
                    tech_indicator_list=self.tech_indicator_list,
                    tech_indicator_list_trend=self.tech_indicator_list_trend,
                    transcation_cost=self.transcation_cost,
                    back_time_length=self.back_time_length,
                    max_holding_number=self.max_holding_number,
                    initial_action=initial_action)
            s, s2, info = val_env.reset()
            done = False
            action_list_episode = []
            reward_list_episode = []
            while not done:
                a = self.act_test(s, s2, info)
                s_, s2_, r, done, info_ = val_env.step(a)
                reward_list_episode.append(r)
                s, s2, info = s_, s2_, info_
                action_list_episode.append(a)
            portfit_magine, final_balance, required_money, commission_fee = val_env.get_final_return_rate(
                slient=True)
            final_balance = val_env.final_balance
            required_money = val_env.required_money
            action_list.append(action_list_episode)
            reward_list.append(reward_list_episode)
            final_balance_list.append(final_balance)
            required_money_list.append(required_money)
            commission_fee_list.append(commission_fee)
        action_list = np.array(action_list)
        reward_list = np.array(reward_list)
        final_balance_list = np.array(final_balance_list)
        required_money_list = np.array(required_money_list)
        commission_fee_list = np.array(commission_fee_list)
        np.save(os.path.join(save_path, "action_val_{}.npy".format(initial_action)), action_list)
        np.save(os.path.join(save_path, "reward_val_{}.npy".format(initial_action)), reward_list)
        np.save(os.path.join(save_path, "final_balance_val_{}.npy".format(initial_action)),
            final_balance_list)
        np.save(os.path.join(save_path, "require_money_val_{}.npy".format(initial_action)),
                required_money_list)
        np.save(os.path.join(save_path, "commission_fee_history_val_{}.npy".format(initial_action)),
                commission_fee_list)
        return_rate_mean = np.nan_to_num(final_balance_list / required_money_list).mean()
        np.save(os.path.join(save_path, "return_rate_mean_val_{}.npy".format(initial_action)),
                return_rate_mean)
        return return_rate_mean
        
        
            


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    agent = DQN(args)
    agent.train()
