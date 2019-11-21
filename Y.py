import numpy as np
# from Power_flow import Power_Flow
# 以下操作是基于已生成等值电路
# Node_data是节点数据
# 格式如下：列1：节点序号；列2：节点类型（短路节点标号'0',负荷节点或联络节点标号'1',电源节点标号'2'）(要求短路节点为1号节点)；
# 列3：电源节点电动势E';列4：电源节点的Xd'（正序）；列5：电源节点X2（负序）；列6：电源节点X0（零序）；
# 列3-6：不是电源节点的节点标'0'
# 本题中，节点0：地；节点1：A；节点2：B；节点3：D；节点4：E1；节点5：E2；节点6：E3.
# E1,E2,E3节点定义是在Xd'外


# 支路数据格式：列1：起始节点编号；列2：终止节点编号；列3：正常运行时支路上等价的电阻；列4：正常运行时支路上等价的电抗;
# 列5：切除故障后支路等价的电阻；列6：切除故障后支路等价的电抗; 列7：零序电路中的电抗；
# 若终止节点为地，则将其置零。
# 将短路节点定为节点1。
# 电源节点的自阻抗要加入Xd'，建议用终止节点编号为0的方式定义。



# 变压器数据格式：列1：变压器两端节点中靠近短路点的节点序号；列2：变压器两端节点中远离短路点的节点序号；
# 列3：变压器在列1对应一端的绕组接法
# 列4：变压器在列2对应一端的绕组接法（YN：'0',Y:'1',d:'2'）



def Y_generation(branch_data):  # 传入支路数据，生成节点导纳矩阵
    max_num_init = max(branch_data[:, 0])  # 为了确定节点数
    max_num_end = max(branch_data[:, 1])
    num_node = max(max_num_init, max_num_end)
    # Y初始化,乘以1j将float的array编程complex
    Y = 1j * np.zeros((int(num_node), int(num_node)))
    for i in range(branch_data.shape[0]):
        if branch_data[i][1] != 0:
            row = int(branch_data[i][0] - 1)  # 数据编号从1开始，程序编号从0开始
            col = int(branch_data[i][1] - 1)
            Y[row][col] += -1 / (branch_data[i][2] + 1j *
                                 branch_data[i][3])  # 互阻抗
            Y[col][row] = Y[row][col]
            Y[row][row] += 1 / (branch_data[i][2] + 1j *
                                branch_data[i][3])  # 自阻抗
            Y[col][col] += 1 / (branch_data[i][2] + 1j * branch_data[i][3])
        else:
            row = int(branch_data[i][0] - 1)  # 数据编号从1开始，程序编号从0开始
            Y[row][row] += 1 / (branch_data[i][2] + 1j *
                                branch_data[i][3])  # 自阻抗
    return Y


def Y_generation_ZS(branch_data):  # 传入支路数据，生成零序节点导纳矩阵
    max_num_init = max([i[0] for i in branch_data])  # 为了确定节点数
    max_num_end = max([i[1] for i in branch_data])
    num_node = int(max(max_num_init, max_num_end))
    Y = 1j * np.zeros((num_node, num_node))  # Y初始化,乘以1j将float的array编程complex
    for i in range(len(branch_data)):
        if branch_data[i][1] != 0:
            row = int(branch_data[i][0] - 1)  # 数据编号从1开始，程序编号从0开始
            col = int(branch_data[i][1] - 1)
            Y[row][col] += -1 / (branch_data[i][2] + 1j *
                                 branch_data[i][6])  # 互阻抗
            Y[col][row] = Y[row][col]
            Y[row][row] += 1 / (branch_data[i][2] + 1j *
                                branch_data[i][6])  # 自阻抗
            Y[col][col] += 1 / (branch_data[i][2] + 1j * branch_data[i][6])
        else:
            row = int(branch_data[i][0] - 1)  # 数据编号从1开始，程序编号从0开始
            Y[row][row] += 1 / (branch_data[i][2] + 1j *
                                branch_data[i][6])  # 自阻抗
    return Y

# Z=np.linalg.inv(Y_generation(branch_data))

# Bus_data = np.array(Bus_data).reshape(-1, 8)
#
# e,f,p,q=Power_Flow(Y_generation(branch_data),Bus_data)
# u=(e**2+f**2)**0.5
