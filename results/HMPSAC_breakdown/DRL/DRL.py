"""
调度规则对比算法
"""
import numpy as np
from environments.MO_DFJSP_breakdown import MO_DFJSP_Environment
from utilities.Utility_Class import FigGan, MyError, DataProcess
import random, time


class CompositeRulesEnv(MO_DFJSP_Environment):
    def __init__(self, use_instance=True, **kwargs):
        super().__init__(use_instance=use_instance, **kwargs)
        self.task_rules = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]  # 工序可选规则列表
        self.machine_rules = [8]  # 机器可选规则列表
        self.rule_dict = {'rule' + str(T) + '_' + str(M): [T, M] for T in self.task_rules for M in self.machine_rules}

    @property
    def rule_random(self):
        return [random.randint(0, 11), random.randint(0, 9)]


if __name__ == '__main__':
    # 读取实例的位置
    data_process = DataProcess()  # 实例化数据处理类
    solutions_dict = 'solutions_dict'
    times_dict = 'times_dict'
    path_instance = 'D:/Python project/HMPSAC_Fluid_Model_MOMDFJSP/data/HMPSAC_breakdown'
    path_writer = 'D:/Python project/HMPSAC_Fluid_Model_MOMDFJSP/results/HMPSAC_breakdown/DRL/'
    file_name_list \
        = ['DDT0.5_M10_S1', 'DDT0.5_M10_S3', 'DDT0.5_M10_S5', 'DDT0.5_M15_S1', 'DDT0.5_M15_S3', 'DDT0.5_M15_S5',
           'DDT0.5_M20_S1', 'DDT0.5_M20_S3', 'DDT0.5_M20_S5', 'DDT1.0_M10_S1', 'DDT1.0_M10_S3', 'DDT1.0_M10_S5',
           'DDT1.0_M15_S1', 'DDT1.0_M15_S3', 'DDT1.0_M15_S5', 'DDT1.0_M20_S1', 'DDT1.0_M20_S3', 'DDT1.0_M20_S5',
           'DDT1.5_M10_S1', 'DDT1.5_M10_S3', 'DDT1.5_M10_S5', 'DDT1.5_M15_S1', 'DDT1.5_M15_S3', 'DDT1.5_M15_S5',
           'DDT1.5_M20_S1', 'DDT1.5_M20_S3', 'DDT1.5_M20_S5']
    # file_name_list = ['DDT1.0_M15_R10']
    # 生成存储文件
    file_path_solutions = path_writer + solutions_dict
    file_path_times = path_writer + times_dict
    solutions_dict = data_process.pickle_read(file_path_solutions)  # 存储解集的字典
    times_dict = data_process.pickle_read(file_path_times)
    print(solutions_dict)
    print(times_dict)
    # 特定算例下循环固定次数
    epoch_number = 10  # 循环次数
    for file_name in file_name_list:  # 文件循环
        env_object = CompositeRulesEnv(use_instance=False, path=path_instance, file_name=file_name)  # 定义环境对象

        # 写入形成的复合调度规则
        for rule, action in env_object.rule_dict.items():  # 规则循环
            objectives = []  # 初始化写入的数据结构
            times = []  # 耗时
            for n in range(epoch_number):  # 次数循环
                time_start = time.process_time()
                state = env_object.reset()  # 初始化状态
                while not env_object.done:
                    next_state, reward, done = env_object.step(action, 0)
                # 保存数据
                objectives.append([env_object.completion_time, env_object.delay_time_sum, env_object.energy_consumption])  # 写入目标值
                times.append(time.process_time() - time_start)
            # 保存生成的多个解
            solutions_dict[(rule, file_name)] = objectives
            times_dict[(rule, file_name)] = times

        # 写入任意规则
        objectives = []  # 初始化写入的数据结构
        times = []  # 耗时
        rule = 'random'
        for n in range(epoch_number):  # 次数循环
            time_start = time.process_time()
            state = env_object.reset()  # 初始化状态
            while not env_object.done:
                action = env_object.rule_random
                next_state, reward, done = env_object.step(action, 0)
            # 保存数据
            objectives.append([env_object.completion_time, env_object.delay_time_sum, env_object.energy_consumption])  # 写入目标值
            times.append(time.process_time() - time_start)
        # 保存生成的多个解
        solutions_dict[(rule, file_name)] = objectives
        times_dict[(rule, file_name)] = times
        # 保存文件
        data_process.pickle_save(file_path_solutions, solutions_dict)
        data_process.pickle_save(file_path_times, times_dict)
        print("写入文件：", file_name)

    # 画甘特图
    # figure_object = FigGan(env_object)
    # figure_object.figure()
