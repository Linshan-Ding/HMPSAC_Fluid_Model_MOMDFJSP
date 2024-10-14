"""
继承环境，并重写调度规则
知名调度规则：
FIFO:最先到达的工件-First in First out
EDD:最小交期时间工件-Earliest due date
MRT:最多剩余处理时间-Most remaining processing time
SPT:下道工序加工时间最短的工件-Shortest processing time
CR:最小的紧迫系数-Critical ratio
复合调度规则：
CR_SPT
EDD_SPT
MRT_SPT
MRT_FIFO
"""
import numpy as np, os
from environments.MO_DFJSP_breakdown import MO_DFJSP_Environment
from utilities.Utility_Class import FigGan, MyError, DataProcess
import random, time


class RulesEnv(MO_DFJSP_Environment):
    def __init__(self, use_instance=True, **kwargs):
        super().__init__(use_instance=use_instance, **kwargs)
        self.task_rules = [0, 1, 2, 3, 4, 5, 6, 7, 8]  # 工序可选规则列表
        self.machine_rules = [0]  # 机器可选规则列表
        self.actions = tuple([(task_rule, machine_rule) for task_rule in range(9) for machine_rule in range(1)])
        self.rule_dict = {'FIFO': [0, 0], 'EDD': [1, 0], 'MRT': [2, 0], 'SPT': [3, 0], 'CR': [4, 0],
                          'CR_SPT': [5, 0], 'EDD_SPT': [6, 0], 'MRT_SPT': [7, 0], 'MRT_FIFO': [8, 0]}

    def task_select(self, task_rule):
        """工序选择规则"""
        # 5个知名规则
        if task_rule == 1:  # FIFO:最先到达的工件-First in First out
            rj = min(self.kind_task_available_list, key=lambda x: self.kind_task_dict[x].min_available_time)
        elif task_rule == 2:  # # EDD:最小交期时间工件-Earliest due date
            rj = min(self.kind_task_available_list, key=lambda x: self.kind_task_due_date[x])
        elif task_rule == 3:  # MRT:最多剩余处理时间-Most remaining processing time
            rj = max(self.kind_task_available_list, key=lambda x: self.kind_task_dict[x].time_remain)
        elif task_rule == 4:  # SPT:下道工序加工时间最短的工件-Shortest processing time
            rj = min(self.kind_task_available_list, key=lambda x: self.time_rj_dict[x])
        elif task_rule == 5:  # CR:最小的紧迫系数-Critical ratio
            rj = max(self.kind_task_available_list, key=lambda x: self.kind_task_delivery_urgency[x])
        # 4个组合规则
        elif task_rule == 6:  # CR_SPT
            rj_max = max(self.kind_task_available_list, key=lambda x: self.kind_task_delivery_urgency[x])
            max_rj_list = [rj for rj in self.kind_task_available_list if self.kind_task_delivery_urgency[rj]
                           == self.kind_task_delivery_urgency[rj_max]]
            rj = min(max_rj_list, key=lambda x: self.time_rj_dict[x])
        elif task_rule == 7:  # EDD_SPT
            rj_min = min(self.kind_task_available_list, key=lambda x: self.kind_task_due_date[x])
            min_rj_list = [rj for rj in self.kind_task_available_list if self.kind_task_due_date[rj]
                           == self.kind_task_due_date[rj_min]]
            rj = min(min_rj_list, key=lambda x: self.time_rj_dict[x])
        elif task_rule == 8:  # MRT_SPT
            rj_max = max(self.kind_task_available_list, key=lambda x: self.kind_task_dict[x].time_remain)
            max_rj_list = [rj for rj in self.kind_task_available_list if self.kind_task_dict[rj].time_remain
                           == self.kind_task_dict[rj_max].time_remain]
            rj = min(max_rj_list, key=lambda x: self.time_rj_dict[x])
        elif task_rule == 9:  # MRT_FIFO
            rj_max = max(self.kind_task_available_list, key=lambda x: self.kind_task_dict[x].time_remain)
            max_rj_list = [rj for rj in self.kind_task_available_list if self.kind_task_dict[rj].time_remain
                           == self.kind_task_dict[rj_max].time_remain]
            rj = min(max_rj_list, key=lambda x: self.kind_task_dict[x].min_available_time)
        else:
            raise MyError("报错：未定义该工序调度规则")
        return rj

    def machine_select(self, machine_rule, rj_selected):
        """机器分配规则"""
        machine_selectable_list = list(set(self.machine_idle_list)&set(self.kind_task_dict[rj_selected].machine_tuple))
        if machine_rule == 1:  # 随机选择一个可选机器
            m = random.choice(machine_selectable_list)
        else:
            raise MyError("报错：未定义该机器调度规则。")
        return m


if __name__ == '__main__':
    # 读取实例的位置
    data_process = DataProcess()  # 实例化数据处理类
    path_instance = 'D:/Python project/HMPSAC_Fluid_Model_MOMDFJSP/data/HMPSAC_breakdown'
    path_writer = 'D:/Python project/HMPSAC_Fluid_Model_MOMDFJSP/results/HMPSAC_breakdown/common_rules/'
    solutions_dict = 'solutions_dict'
    times_dict = 'times_dict'
    file_name_list \
        = ['DDT0.5_M10_S1', 'DDT0.5_M10_S3', 'DDT0.5_M10_S5', 'DDT0.5_M15_S1', 'DDT0.5_M15_S3', 'DDT0.5_M15_S5',
           'DDT0.5_M20_S1', 'DDT0.5_M20_S3', 'DDT0.5_M20_S5', 'DDT1.0_M10_S1', 'DDT1.0_M10_S3', 'DDT1.0_M10_S5',
           'DDT1.0_M15_S1', 'DDT1.0_M15_S3', 'DDT1.0_M15_S5', 'DDT1.0_M20_S1', 'DDT1.0_M20_S3', 'DDT1.0_M20_S5',
           'DDT1.5_M10_S1', 'DDT1.5_M10_S3', 'DDT1.5_M10_S5', 'DDT1.5_M15_S1', 'DDT1.5_M15_S3', 'DDT1.5_M15_S5',
           'DDT1.5_M20_S1', 'DDT1.5_M20_S3', 'DDT1.5_M20_S5']

    # file_name_list = ['DDT0.5_M10_R5', 'DDT1.0_M15_R10']
    # 生成存储文件
    file_path_solutions = path_writer + solutions_dict
    file_path_times = path_writer + times_dict
    solutions_dict = data_process.pickle_read(file_path_solutions)  # 存储解集的字典
    times_dict = data_process.pickle_read(file_path_times)
    print(solutions_dict)
    print(times_dict)
    # 特定算例下循环固定次数
    epoch_number = 30  # 循环次数
    for file_name in file_name_list:  # 文件循环
        env_object = RulesEnv(use_instance=False, path=path_instance, file_name=file_name)  # 定义环境对象
        # 写入形成的复合调度规则
        for rule, action in env_object.rule_dict.items():  # 规则循环
            objectives = []  # 初始化写入的数据结构
            times = []  # 耗时
            for n in range(epoch_number):  # 次数循环---输出最优值+平均值+标准差
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
        # 写入文件
        data_process.pickle_save(file_path_solutions, solutions_dict)
        data_process.pickle_save(file_path_times, times_dict)
        print("写入文件：", file_name)

    # 画甘特图
    # figure_object = FigGan(env_object)
    # figure_object.figure()
