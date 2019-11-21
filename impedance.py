import numpy as np
from Y import Y_generation, Y_generation_ZS
from math import atan,pi
from operator import itemgetter
import copy

def NSI(branch_data,Node_data): #Negative sequence impedance
    size=Node_data.shape[0]
    index_power_node = np.argwhere(Node_data[:, 1] == 2)[0][0]
    temp_branch=copy.deepcopy(branch_data)

    #生成负序的负荷等效电抗值，由于原branch_data中终止节点为0（大地）的只有带负荷节点
    load=[]
    for i in range(branch_data.shape[0]):
        if branch_data[branch_data.shape[0]-i-1][1]==0:
            load.append({'index':branch_data[branch_data.shape[0]-i-1][0],'R_load':branch_data[branch_data.shape[0]-i-1][2],'X_load':branch_data[branch_data.shape[0]-i-1][3]})
            temp_branch=np.delete(temp_branch,branch_data.shape[0]-i-1,0)

    for i in range(len(load)):
        X_load_negative=0.35*(load[i]['R_load']**2+load[i]['X_load']**2)**0.5 #负荷负序电抗计算公式：=0.35*\Z\
        temp_branch=np.insert(temp_branch,-1,[load[i]['index'],0,0,X_load_negative,0,0,0],0)

    #向branch_data中插入发电机负序电抗的相应值，相当于该节点向地接入负序电抗
    for i in range(index_power_node,size):
        temp_branch=np.insert(temp_branch,-1,[i+1,0,0,float(Node_data[i][4]),0,0,0],0)
    Z = np.linalg.inv(Y_generation(temp_branch))  #生成Z矩阵，Z[0][0]即为短路点的输入阻抗(负序网络)
    return Z[0][0]


def ZSI(branch_data,trans_data): #Zero sequence impedance

    temp_branch=copy.deepcopy(branch_data)#对于List里面还有list的要用深拷贝，deepcopy
    #删去零序中的负荷等效阻抗
    for i in range(branch_data.shape[0]):
        if branch_data[branch_data.shape[0]-i-1][1]==0:
            temp_branch=np.delete(temp_branch,branch_data.shape[0]-i-1,0)

    index_abort1=[];index_abort2=[] #改用变压器数据trans_data写
    for i in range(trans_data.shape[0]):
        if trans_data[i][2] != '0' or trans_data[i][3] != '0':
            index_abort1.append(int(trans_data[i][0]))
            index_abort2.append(int(trans_data[i][1]))
    #将变压器导致不能使零序电流流通的支路删去，即将变压器两端中远离短路点的一端置为接地
    for i in range(len(temp_branch)):
        if (temp_branch[i][0] in index_abort1 and temp_branch[i][1] in index_abort2):
            temp_branch[i][1]= 0
        if  (temp_branch[i][0] in index_abort2 and temp_branch[i][1] in index_abort1):
            temp_branch[i][0]=temp_branch[i][1]
            temp_branch[i][1]= 0

    Z = np.linalg.inv(Y_generation_ZS(temp_branch))  # 生成Z矩阵，Z[0][0]即为短路点的输入阻抗（零序网络）
    return Z[0][0]


#求附加阻抗，第四个输入量为短路类型，三相短路：1；两相短路接地：2；两相短路：3；单相短路：4:
def X_star(branch_data,Node_data,trans_data,SC_type=2):
    Z_NS = NSI(branch_data, Node_data)
    Z_ZS = ZSI(branch_data, trans_data)

    if SC_type==1:
        Z_star=0

    if SC_type==2:
        Z_star=(Z_NS*Z_ZS)/(Z_NS+Z_ZS)

    if SC_type==3:
        Z_star=Z_NS

    if SC_type==4:
        Z_star=Z_NS+Z_ZS

    return Z_star.imag

#从原始数据，生成求转移阻抗要用的branch_data(故障发生后)
def branch_data_for_imp_SC(branch_data,Node_data,trans_data,SC_type=2):
    temp_branch=copy.deepcopy(branch_data)
    size=Node_data.shape[0]
    index_power_node = np.argwhere(Node_data[:, 1] == 2)[0][0]
    # 向branch_data中插入发电机负序电抗的相应值，相当于该节点向地接入负序电抗
    for i in range(index_power_node, size):
        temp_branch = np.insert(temp_branch, -1, [i + 1, 0, 0, float(Node_data[i][3]),0,0,0], 0)
    X_star_value=X_star(branch_data,Node_data,trans_data,SC_type)
    temp_branch=np.insert(temp_branch,-1,[1, 0, 0, X_star_value,0,0,0],0)
    return temp_branch


#从原始数据，生成求转移阻抗要用的branch_data(故障切除后)
def branch_data_for_imp_afterCut(branch_data,Node_data,trans_data,SC_type=2):
    temp_branch=copy.deepcopy(branch_data)
    for i in range(branch_data.shape[0]):
        temp_branch[i][2]=temp_branch[i][4]
        temp_branch[i][3] = temp_branch[i][5]

    size=Node_data.shape[0]
    index_power_node = np.argwhere(Node_data[:, 1] == 2)[0][0]
    # 向branch_data中插入发电机负序电抗的相应值，相当于该节点向地接入负序电抗
    for i in range(index_power_node, size):
        temp_branch=np.insert(temp_branch, -1, [i + 1, 0, 0, float(Node_data[i][3]),0,0,0], 0)
    return temp_branch

#求各电源节点之间的转移阻抗和输入阻抗
#return Z矩阵，对角线元素为输入矩阵，非对角元为转移阻抗，节点顺序与Node_data一致
def Impedance(branch_data,Node_data):
    Y_transfer=Y_generation(branch_data)
    size=Y_transfer.shape[0]
    #通过查找Node_data数据，确定第一个电源节点的index
    index_power_node=np.argwhere(Node_data[:, 1] == 2)[0][0]
    Z=1j*np.zeros((size-index_power_node,size-index_power_node))
    # 得到仅有电源节点的Z矩阵，元素则为自阻抗和互阻抗
    # 这是用于计算转移阻抗的Z矩阵，将Xd'计及矩阵
    Z_transfer = np.linalg.inv(Y_transfer)[index_power_node:, index_power_node:]
    count=0
    Xd_dic = []
    for i in range(branch_data.shape[0]):
        if branch_data[branch_data.shape[0]-i-1][0]>index_power_node and branch_data[branch_data.shape[0]-i-1][1]==0:
            Xd_dic.append({'index':branch_data[branch_data.shape[0]-i-1][0],'value':branch_data[branch_data.shape[0]-i-1][3]})
            count+=1
        if count==size-index_power_node:
            break

    Z_input_list=[]
    for j in range(size-index_power_node):
        branch_data_exclude_Xd = copy.deepcopy(branch_data)
        for i in range(branch_data.shape[0]):
            if branch_data[branch_data.shape[0]-i-1][0]==index_power_node+j+1 and branch_data[branch_data.shape[0]-i-1][1]==0:
                branch_data_exclude_Xd=np.delete(branch_data_exclude_Xd,branch_data.shape[0]-i-1,0)
                break
        # 这是用于计算输入阻抗的Z矩阵，未将Xd'计及矩阵,计算发电机1时不考虑Xd'1，要考虑其它发电机的Xd'
        Y_input = Y_generation(branch_data_exclude_Xd)
        Z_input = np.linalg.inv(Y_input)[index_power_node:, index_power_node:]
        Z_input_list.append(Z_input[j][j])

    # 通过公式 Zij=zi*zj/Z_transfer[i][j]，求出电源节点i、j的转移阻抗
    # 通过公式 Zii=zi+Z_input[i][i]，求出电源节点i、j的转移阻抗

    Xd_value=[]
    Xd_dic_sorted = sorted(Xd_dic, key=itemgetter('index')) #将Xd按电源节点的顺序排好
    for i in range(len(Xd_dic_sorted)):
        Xd_value.append(Xd_dic_sorted[i]['value'])

    # print(Xd_dic)
    for i in range(size-index_power_node):
        for j in range(size-index_power_node):
            if i!=j:
                Z[i][j]=1j*Xd_value[i]*1j*Xd_value[j]/Z_transfer[i][j]
            else:
                Z[i][i]=1j*Xd_value[i]+Z_input_list[i]
    return Z

#求转移阻抗、输入阻抗的模，返回值为弧度制
def Z_value(branch_data,Node_data):
    Z = Impedance(branch_data, Node_data)
    size=Z.shape[0]
    Z_value_matrix=np.zeros((size,size))
    for i in range(size):
        for j in range(size):
            Z_value_matrix[i][j]=(Z[i][j].real**2+Z[i][j].imag**2)**0.5
    return Z_value_matrix


#求转移阻抗、输入阻抗角，返回值为弧度制
def alpha(branch_data,Node_data):
    Z=Impedance(branch_data, Node_data)

    size=Z.shape[0]
    alpha=np.zeros((size,size))
    for i in range(size):
        for j in range(i,size):
            if i!=j:
                alpha[i][j]=atan(Z[i][j].imag/Z[i][j].real)
                alpha[j][i]=alpha[i][j]
            else:
                alpha[i][i]=atan(Z[i][i].imag/Z[i][i].real)
            if alpha[i][j] <0:
                alpha[i][j]+=pi
                alpha[j][i] = alpha[i][j]
    alpha=pi/2-alpha
    return alpha


# print(branch_data_for_imp_SC(branch_data,Node_data,trans_data,2))
# print(Impedance(branch_data_for_imp_afterCut(branch_data,Node_data,trans_data,2),Node_data))
# print(NSI(branch_data,Node_data))
# print(ZSI(branch_data,trans_data))
# print(X_star(branch_data,Node_data,trans_data,2))
#print(Z_value(branch_data,Node_data))
# print(alpha(branch_data_for_imp_afterCut(branch_data,Node_data,trans_data,2),Node_data)*180/pi)
# print(Impedance(branch_data,Node_data))
