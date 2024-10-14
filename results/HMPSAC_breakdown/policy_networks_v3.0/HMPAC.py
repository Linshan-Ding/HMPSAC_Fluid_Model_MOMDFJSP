"""
actor+critic PPO算法作为上层控制策略网络
"""
import numpy as np
import torch, random, copy, sys
from torch import nn
import torch.nn.functional as F
from torch import optim
from agents.Base_Agent import Base_Agent
from utilities.Utility_Functions import create_actor_distribution
from utilities.data_structures.Config import Config
from environments.MO_DFJSP_breakdown import MO_DFJSP_Environment
from visdom import Visdom
from utilities.Utility_Class import AddData
from utilities.Utility_Class import FigGan, MyError, DataProcess
import random, time


# 构建工序策略网络类
class TaskPolicyNet(nn.Module):
    def __init__(self, input_size_1, hidden_size, hidden_layer_1, output_size_1):
        super(TaskPolicyNet, self).__init__()
        self.name = "task_policy"
        # 定义工序策略网络输入层
        self.layers_1 = nn.ModuleList([nn.Linear(input_size_1, hidden_size), nn.ReLU()])
        # 定义工序策略网络隐藏层
        for i in range(hidden_layer_1 - 1):
            self.layers_1.append(nn.Linear(hidden_size, hidden_size))
            self.layers_1.append(nn.ReLU())
        # 定义工序策略网络输出层
        self.layers_1.append(nn.Linear(hidden_size, output_size_1))

    def forward(self, x):
        for layer in self.layers_1:
            x = layer(x)
        x = F.softmax(x, dim=-1)
        return x


# 构建机器策略网络类
class MachinePolicyNet(nn.Module):
    def __init__(self, input_size_2, hidden_size, hidden_layer_2, output_size_2):
        super(MachinePolicyNet, self).__init__()
        self.name = "machine_policy"
        # 定义机器策略网络输入层
        self.layers_2 = nn.ModuleList([nn.Linear(input_size_2, hidden_size), nn.ReLU()])
        # 定义机器策略网络隐藏层
        for i in range(hidden_layer_2 - 1):
            self.layers_2.append(nn.Linear(hidden_size, hidden_size))
            self.layers_2.append(nn.ReLU())
        # 定义机器策略网络输出层
        self.layers_2.append(nn.Linear(hidden_size, output_size_2))

    def forward(self, x):
        for layer in self.layers_2:
            x = layer(x)
        x = F.softmax(x, dim=-1)
        return x


class ActorNet(nn.Module):
    """演员策略网络"""
    def __init__(self, input_size, hidden_size, hidden_layer, output_size):
        super(ActorNet, self).__init__()
        # 定义机器策略网络输入层
        self.layers = nn.ModuleList([nn.Linear(input_size, hidden_size), nn.ReLU()])
        # 定义机器策略网络隐藏层
        for i in range(hidden_layer - 1):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
            self.layers.append(nn.ReLU())
        # 定义机器策略网络输出层
        self.layers.append(nn.Linear(hidden_size, output_size))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = F.softmax(x, dim=-1)
        return x


class CriticNet(nn.Module):
    """评论家网络"""
    def __init__(self, input_size, hidden_size, hidden_layer, output_size):
        super(CriticNet, self).__init__()
        # 定义评论家网络输入层
        self.layers = nn.ModuleList([nn.Linear(input_size, hidden_size), nn.ReLU()])
        # 定义评论家网络隐藏层
        for i in range(hidden_layer - 1):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
            self.layers.append(nn.ReLU())
        # 定义评论家网络输出层
        self.layers.append(nn.Linear(hidden_size, output_size))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class PPO(Base_Agent, Config):
    """多策略近端策略优化算法"""
    def __init__(self):
        Base_Agent.__init__(self)
        Config.__init__(self)  # 继承算法超参数类
        self.config = Config()  # 算法控制参数
        self.agent = "MP_PPO"
        self.hyper_parameters = self.hyper_parameters[self.agent]  # 算法控制参数
        self.action_types = "DISCRETE"
        self.episode_states = []  # 状态
        self.episode_actions = []  # 动作
        self.episode_rewards = []  # 回报
        self.epsilon_decay_denominator = self.hyper_parameters["epsilon_decay_rate_denominator"]  # 探索衰变率分母
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 采用GPU训练
        self.actor_number = self.hyper_parameters["actor_number"]  # 策略网络数量[5/9/11]
        self.policy_tuple = ['task', 'machine']  # 策略网络编号元组
        self.policy_task = self.policy_tuple[0]  # 完工时间目标网络编号
        self.policy_machine = self.policy_tuple[-1]  # 总延迟时间目标网络编号
        # 超参数
        self.num_episodes_to_run = self.hyper_parameters["num_episodes_to_run"]  # 训练周期数
        self.learning_rate = self.hyper_parameters["learning_rate"]  # 学习率
        self.discount_rate = self.hyper_parameters["discount_rate"]  # 折扣率
        self.buffer_size = self.hyper_parameters["buffer_size"]  # 回放记忆缓存大小
        self.batch_size = self.hyper_parameters["batch_size"]  # 采样批量
        self.gradient_clipping_norm = self.hyper_parameters["gradient_clipping_norm"]  # 梯度裁剪
        self.learning_iterations_per_round_actor = self.hyper_parameters["learning_iterations_per_round_actor"]  # 每次采样更新，演员连续更新轮数
        self.learning_iterations_per_round_critic = self.hyper_parameters["learning_iterations_per_round_critic"]  # 每次采样更新，评论家连续更新轮数
        # 算法运行参数
        self.global_step_number = 0  # 运行总步数
        self.episode_number = 0  # 更新周期数
        # 环境状态维度参数
        self.state_size = 30
        self.action_size = 3
        self.actor_input_size = self.state_size
        self.critic_input_size = self.state_size
        # 存储相关值
        self.actor_old_log_prob = None  # 旧网络的动作概率
        self.discounted_returns = None  # 折扣回报
        self.critic_targets = None  # 评论家输出的目标V值
        self.advantages = None  # 优势函数值
        self.log_prob = None  # 动作概率
        # 初始化当前优化的策略网络对象
        self.actor_new = None  # 策略网络对象字典
        # 定义评论家网络+评论家优化器
        self.critic_local = None
        self.critic_optimizer = None
        self.memory = None
        self.load_policy_network()  # 加载上层网络
        # 导入三个训练好的目标策略网络
        self.objectives_policy = {'makespan': 0, 'tardiness': 1, 'energy': 2}
        self.action_size_dict = {'task': 12, 'machine': 10}
        self.policy_dict = {0: {}, 1: {}, 2: {}}
        self.load_policy_model()  # 加载下层三个目标网络
        self.reward_sum = float('-inf')  # 基准算例的回报

    def load_policy_model(self):
        """加载三目标策略网络"""
        for objective, policy in self.objectives_policy.items():
            file_path = 'D:/Python project/HMPSAC_Fluid_Model_MOMDFJSP/results/HMPSAC_breakdown/policy_networks_v5.' + str(policy + 1) + '/'
            actor_net_task = TaskPolicyNet(input_size_1=30, hidden_size=200, hidden_layer_1=3, output_size_1=12).to(self.device)
            actor_net_task.load_state_dict(torch.load(file_path + 'actor_task_model.path'))
            self.policy_dict[policy]['task'] = actor_net_task
            actor_net_machine = MachinePolicyNet(input_size_2=31, hidden_size=200, hidden_layer_2=3, output_size_2=10).to(self.device)
            actor_net_machine.load_state_dict(torch.load(file_path + 'actor_machine_model.path'))
            self.policy_dict[policy]['machine'] = actor_net_machine

    def load_policy_network(self):
        """加载上层策略网络"""
        high_control_network = 'D:/Python project/HMPSAC_Fluid_Model_MOMDFJSP/results/HMPSAC_breakdown/policy_networks_v3.0/control_policy_network.path'
        self.actor_new = ActorNet(input_size=self.actor_input_size, hidden_size=200, hidden_layer=3, output_size=self.action_size).to(self.device)
        self.actor_new.load_state_dict(torch.load(high_control_network))

    def run_one_episode(self, environment):
        """特定策略网络在特定环境下运行一个周期"""
        self.state = environment.reset()
        self.done = False
        while not self.done:
            self.action, _ = self.pick_action_and_log_prob(policy=self.actor_new, state=self.state, epsilon_exploration=0)
            action_task = self.pick_lower_action(policy=self.policy_dict[self.action]['task'], state=self.state, action_size=self.action_size_dict['task'])
            state_machine = np.append(self.state, action_task)  # 带选择的工序规则信息的状态
            action_machine = self.pick_lower_action(policy=self.policy_dict[self.action]['machine'], state=state_machine,
                                                    action_size=self.action_size_dict['machine'])
            action_task_machine = np.array([action_task, action_machine])
            self.next_state, self.reward, self.done = environment.step(action_task_machine,  reward_policy=3,  completion=517,
                                                                       tardiness=1330, energy_consumption=558604)
            self.state = self.next_state  # 更新当前状态

        # 输出
        return [environment.completion_time, environment.delay_time_sum, environment.energy_consumption]

    def pick_action_and_log_prob(self, policy, state, epsilon_exploration=None):
        """基于策略采样一个动作"""
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        actor_output = policy.forward(state)
        action_distribution = create_actor_distribution(self.action_types, actor_output, self.action_size)
        action = action_distribution.sample().cpu().numpy()
        action = int(action)
        if random.random() <= epsilon_exploration:
            action = random.randint(0, self.action_size - 1)
        else:
            action = action
        action_log_prob = self.calculate_log_action_probability(action, action_distribution)
        return action, action_log_prob

    def pick_lower_action(self, policy, state, action_size):
        """基于策略采样一个动作"""
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        actor_output = policy.forward(state)
        action_distribution = create_actor_distribution(self.action_types, actor_output, action_size)
        action = action_distribution.sample().cpu().numpy()
        action = int(action)
        return action

    def calculate_log_action_probability(self, actions, action_distribution):
        """计算所选动作的log概率"""
        policy_distribution_log_prob = action_distribution.log_prob(torch.Tensor([actions]).to(self.device))
        return policy_distribution_log_prob


if __name__ == '__main__':
    # 读取实例的位置
    data_process = DataProcess()  # 实例化数据处理类
    solutions_dict = 'solutions_dict'
    times_dict = 'times_dict'
    path_instance = 'D:/Python project/HMPSAC_Fluid_Model_MOMDFJSP/data/HMPSAC_breakdown'
    path_writer = 'D:/Python project/HMPSAC_Fluid_Model_MOMDFJSP/results/HMPSAC_breakdown/policy_networks_v3.0/'
    file_name_list \
        = ['DDT0.5_M10_S1', 'DDT0.5_M10_S3', 'DDT0.5_M10_S5', 'DDT0.5_M15_S1', 'DDT0.5_M15_S3', 'DDT0.5_M15_S5',
           'DDT0.5_M20_S1', 'DDT0.5_M20_S3', 'DDT0.5_M20_S5', 'DDT1.0_M10_S1', 'DDT1.0_M10_S3', 'DDT1.0_M10_S5',
           'DDT1.0_M15_S1', 'DDT1.0_M15_S3', 'DDT1.0_M15_S5', 'DDT1.0_M20_S1', 'DDT1.0_M20_S3', 'DDT1.0_M20_S5',
           'DDT1.5_M10_S1', 'DDT1.5_M10_S3', 'DDT1.5_M10_S5', 'DDT1.5_M15_S1', 'DDT1.5_M15_S3', 'DDT1.5_M15_S5',
           'DDT1.5_M20_S1', 'DDT1.5_M20_S3', 'DDT1.5_M20_S5']
    # file_name_list = ['DDT1.0_M15_S3']
    # 生成存储文件
    file_path_solutions = path_writer + solutions_dict
    file_path_times = path_writer + times_dict
    solutions_dict = data_process.pickle_read(file_path_solutions)  # 存储解集的字典
    times_dict = data_process.pickle_read(file_path_times)
    print(solutions_dict)
    print(times_dict)
    ppo_object = PPO()
    # 特定算例下循环固定次数
    epoch_number = 30  # 循环次数
    algorithm = 'HMPAC'
    for file_name in file_name_list:  # 文件循环
        env_object = MO_DFJSP_Environment(use_instance=False, path=path_instance, file_name=file_name)  # 定义环境对象
        objectives = []  # 初始化写入的数据结构
        times = []  # 耗时
        for n in range(epoch_number):  # 次数循环
            time_start = time.process_time()
            state = env_object.reset()  # 初始化状态
            ppo_object.run_one_episode(env_object)  # 环境运行一个周期
            # 保存数据
            objectives.append([env_object.completion_time, env_object.delay_time_sum, env_object.energy_consumption])  # 写入目标值
            times.append(time.process_time() - time_start)
        # 保存生成的多个解
        solutions_dict[(algorithm, file_name)] = objectives
        times_dict[(algorithm, file_name)] = times
        # 保存文件
        data_process.pickle_save(file_path_solutions, solutions_dict)
        data_process.pickle_save(file_path_times, times_dict)
        print("写入文件：", file_name)