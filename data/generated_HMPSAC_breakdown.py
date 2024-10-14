import os
import csv
import sys
import numpy as np
from numpy.random import poisson
from random import randint, uniform


class Instance:
    def __init__(self, DDT, M, S):
        self.DDT = DDT
        self.machine_count = M
        self.machine_tuple = tuple(range(M))
        self.order_count = S
        self.order_tuple = tuple(range(S))
        self.kind_count = randint(5, 15)  # 工件类型数
        self.kind_tuple = tuple(range(self.kind_count))
        self.file_name = f'DDT{DDT}_M{M}_S{S}'
        # ... Additional attributes for processes and operations
        self.task_r_dict, self.machine_rj_dict, self.kind_task_m_dict, self.time_rjm_dict, self.count_sr_dict, self.time_arrive_s_dict, \
        self.time_delivery_s_dict, self.kind_task_tuple, self.time_mrj_dict, self.power_mrj_dict, self.power_m_dict = self.process_information()

    """每种工件类型的工序数"""

    @property
    def J_r(self):
        return randint(5, 10)

    """工序在可选机器上的加工时间"""

    @property
    def t_rjm(self):
        return randint(1, 20)

    """每个订单中包含的每种工件类型的工件数量"""

    @property
    def N_sr(self):
        return randint(5, 10)

    """订单到达时间间隔"""

    @property
    def t_si(self):
        return uniform(100, 200)

    """工序在可选机器上的加工功率"""

    @property
    def p_rjm(self):
        return randint(10, 200)

    """机器的待机功率"""

    @property
    def p_m_idle(self):
        return randint(1, 9)

    def process_information(self):
        """生成实例文件数据，并存入csv文件"""
        # 机器、工件信息
        task_r_dict = {r: tuple(j for j in range(self.J_r)) for r in self.kind_tuple}  # [r]对应工序元组
        kind_task_tuple = tuple((r, j) for r in self.kind_tuple for j in task_r_dict[r])  # 工序类型元组
        machine_rj_dict = {r: {j: tuple(np.random.choice(self.machine_tuple, randint(1, self.machine_count), replace=False))
                               for j in task_r_dict[r]} for r in self.kind_tuple}  # [r][j]可选机器元组
        time_rjm_dict = {r: {j: {m: self.t_rjm for m in machine_rj_dict[r][j]}
                             for j in task_r_dict[r]} for r in self.kind_tuple}  # [r][j][m]加工时间
        kind_task_m_dict = {m: tuple((r, j) for r in self.kind_tuple for j in task_r_dict[r]
                                     if m in machine_rj_dict[r][j]) for m in self.machine_tuple}
        time_mrj_dict = {m: {rj: time_rjm_dict[rj[0]][rj[1]][m] for rj in kind_task_m_dict[m]} for m in self.machine_tuple}
        # 各工序加工时间均值
        time_rj_dict = {r: {j: sum([time_rjm_dict[r][j][m] for m in machine_rj_dict[r][j]]) / len(machine_rj_dict[r][j])
                            for j in task_r_dict[r]} for r in self.kind_tuple}
        # 订单信息
        count_sr_dict = {s: tuple(self.N_sr for r in range(len(task_r_dict))) for s in self.order_tuple}  # [s][r]工件类型的数量
        time_gap_s_dict = {
            s: sum([time_rj_dict[r][j] * count_sr_dict[s][r] for r in self.kind_tuple for j in task_r_dict[r]]) * self.DDT / (self.machine_count * 2)
            for s in self.order_tuple}  # 各订单交期-到达时间差值
        time_interval_list = [self.t_si for s in range(self.order_count - 1)]  # 各订单的间隔时间
        time_interval_list.insert(0, 0)
        time_arrive_s_dict = {s: int(sum(time_interval_list[:s + 1])) for s in self.order_tuple}  # 各订单的到达时间
        time_delivery_list = [time_arrive_s_dict[s] + time_gap_s_dict[s] for s in self.order_tuple]
        time_delivery_list.sort()
        time_delivery_s_dict = {s: int(time_delivery_list[s]) for s in self.order_tuple}  # 各订单的交期时间
        # 生成加工功率相关信息
        power_mrj_dict = {m: {rj: self.p_rjm for rj in kind_task_m_dict[m]} for m in self.machine_tuple}
        power_m_dict = {m: self.p_m_idle for m in self.machine_tuple}

        return task_r_dict, machine_rj_dict, kind_task_m_dict, time_rjm_dict, count_sr_dict, time_arrive_s_dict, \
               time_delivery_s_dict, kind_task_tuple, time_mrj_dict, power_mrj_dict, power_m_dict

    def generate_breakdowns(self, max_time=5000, mean_interval=190, mean_duration=10):
        breakdowns = {}
        for m in self.machine_tuple:
            breakdown_times = []
            total_time = 0
            while total_time < max_time:
                interval = int(np.random.exponential(mean_interval))
                duration = int(np.random.exponential(mean_duration))
                start_time = total_time + interval
                end_time = start_time + duration
                if end_time > max_time:
                    break  # Prevent exceeding the maximum time frame
                breakdown_times.append((start_time, end_time))
                total_time = end_time  # Move to the end of the current breakdown
            breakdowns[m] = breakdown_times
        return breakdowns

    def write_files(self):
        base_path = 'HMPSAC_breakdown'
        instance_folder = os.path.join(base_path, self.file_name)
        os.makedirs(instance_folder, exist_ok=True)

        # Write other data files as required...
        file_csv = {'based_data.csv': ['kind_count', 'machine_count', 'order_count', 'DDT'],
                    'process_data.csv': ['kind', 'task', 'machine_selectable', 'process_time', 'power'],
                    'order_data.csv': ['order', 'time_arrive', 'time_delivery', 'kind_number']}

        for csv_name, header in file_csv.items():
            data_file = os.path.join(instance_folder, csv_name)
            with open(data_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                rows = []  # 初始化写入数据
                if csv_name == 'based_data.csv':
                    rows.append([self.kind_count, self.machine_count, self.order_count, self.DDT])
                elif csv_name == 'process_data.csv':
                    for r in self.kind_tuple:
                        for j in self.task_r_dict[r]:
                            time_machine_tuple = tuple(self.time_rjm_dict[r][j][m] for m in self.machine_rj_dict[r][j])
                            power_machine_tuple = tuple(self.power_mrj_dict[m][(r, j)] for m in self.machine_rj_dict[r][j])
                            rows.append([r, j, self.machine_rj_dict[r][j], time_machine_tuple, power_machine_tuple])
                else:
                    for s in self.order_tuple:
                        rows.append([s, self.time_arrive_s_dict[s], self.time_delivery_s_dict[s], self.count_sr_dict[s]])
                writer.writerows(rows)

        # Writing machine breakdown data to CSV
        machine_data_path = os.path.join(instance_folder, 'machine_data.csv')
        with open(machine_data_path, 'w', newline='') as machine_file:
            writer = csv.writer(machine_file)
            writer.writerow(['machine', 'idle_power', 'breakdown_start', 'breakdown_end'])
            for m, breakdowns in self.generate_breakdowns().items():
                idle_power = np.random.randint(1, 10)
                for start, end in breakdowns:
                    writer.writerow([m, idle_power, start, end])


if __name__ == '__main__':
    # 程序需要确认是否覆盖原文件
    print("程序需要确认继续，请输入 y 继续(覆盖原生成文件，可导致数据丢失)或者 n 取消：")
    user_input = input()
    if user_input.lower() == 'y':
        print("继续执行程序...")
        # 初始化算例集参数
        DDT_list = [0.5, 1.0, 1.5]
        machine_count_list = [10, 15, 20]
        order_count_list = [1, 3, 5]
        file_name_list = []
        # 生成各实例文件
        for DDT in DDT_list:
            for M in machine_count_list:
                for S in order_count_list:
                    instance_object = Instance(DDT, M, S)
                    instance_object.write_files()
                    file_name_list.append(instance_object.file_name)
        print(file_name_list)
    else:
        print("取消执行程序。")
        sys.exit(0)
