import numpy as np
import pandas as pd
import sympy as sy
import time
from scipy.linalg import solve
import matplotlib.pyplot as plt
from impedance import Z_value, alpha,branch_data_for_imp_SC,branch_data_for_imp_afterCut
import Y

def subs(func, value):
    '''将表达式用value字典替换'''
    if type(func) in (int,float):
        return func
    elif callable(func):
        return func(*value.values())
    else:
        return func.subs(value)
    
def newton_laphson(Functions, init_values, e=0.00001):
    '''
    牛顿拉夫逊法
    parameters
    ==========
    functions 多个形如f(x,y,z,...)=0的待迭代函数的迭代类型（向量）
    init_values 函数待求解变量的初始值,其中元素个数与所有函数的总变量个数相同
    h 步长
    e 收敛满足max(|F(n+1)-Fn|) < e
    '''
    #初始化雅可比矩阵
#     print('original functions -> ',Functions)
#     print('init_values -> ',init_values)
    iter_keys, iter_values = np.array(list(init_values.keys())),np.array(list(init_values.values()))
#     print('iter keys -> ',iter_keys)
    Jacobian = sy.diff(Functions, iter_keys).transpose()
#     print('Jacobian -> ',Jacobian)
    Functions = sy.lambdify(all_var, Functions,'numpy')
    nocoverage = True
    while(nocoverage):
        #前一次所有faunction的值构成的array
        Functions_values = np.array(Functions(*iter_values))
        Jacobian_values = Jacobian.subs(dict(zip(iter_keys,iter_values)))
        #将雅可比矩阵转成np.float32类型，否则无法求解
        Jacobian_values = np.array(Jacobian_values.tomatrix(),dtype=np.float32)
#         print(Jacobian_values)
#         print(iter_values)
#         print(solve(Jacobian_values,Functions_values))
        iter_values -= solve(Jacobian_values,Functions_values)
        Functions_values_again = np.array(Functions(*iter_values))
#         print(np.concatenate([Functions_values[:,None],Functions_values_again[:,None]],axis=1))
        nocoverage = False if max(abs(Functions_values_again-Functions_values)) < e else True
    return dict(zip(iter_keys, iter_values))
def hiding_trapezium(derivative, n=50, limit=(0,1), e=0.0001):
    step = (limit[1] - limit[0]) / n
    init_values = derivative.pop('init_values')
    #协同运算但不使用牛顿拉夫逊法的量
    co_vars,co_init_values = derivative.pop('co_vars'),derivative.pop('co_init_values')
    #生成定义域内的时间序列
    t = np.linspace(limit[0], limit[1], n+1)
    
    all_init_values = init_values+co_init_values
    #将所有初始值连接成一整个字典
    iter_values = dict(zip((*derivative,*co_vars), all_init_values))
    all_iter_values = np.full((n+1,len(all_init_values)), all_init_values, dtype=np.float32).T
    all_iter_values = dict(zip(iter_values.keys(),all_iter_values))
    for i in range(1, n+1):
        print(f'\r第[{i}]步',end='')
        new_derivative = {key:key-subs(key,iter_values)-step/2*(value+subs(value,iter_values)) \
                                                  for key,value in derivative.items()}
#         print('[+]---------',derivative)
        #将derivative剩下的元素与co_vars对应的值转换为array
        Functions = np.array(list(new_derivative.values()) + list(co_vars.values()))
#         print('[F]Functions ----> ',Functions)
#         print('[it]iter_values ---> ',iter_values)
#         print('[+]->',sy.diff(Functions, np.array(list(iter_values.keys()))).tomatrix().T)
        iter_values = newton_laphson(Functions, iter_values, e=e)
        for key,value in iter_values.items():
            all_iter_values[key][i] = value
            
    return t,all_iter_values

#读取节点分支数据
file_name = 'data/'
Node_data = np.load(f"{file_name}Node_data.npy")
branch_data = np.load(f"{file_name}branch_data.npy")
trans_data = np.load(f"{file_name}trans_data.npy")
index_power=np.argwhere(Node_data[:,1]==2)
E1=Node_data[index_power[0,0],2]
E2=Node_data[index_power[1,0],2]
E3=Node_data[index_power[2,0],2]
#对节点支路数据处理，获取短路后与故障切除后输入阻抗、转移阻抗
SC_branch = branch_data_for_imp_SC(branch_data,Node_data,trans_data)
after_branch = branch_data_for_imp_afterCut(branch_data,Node_data,trans_data)

z_value_SC = Z_value(SC_branch, Node_data)
z_value_after_cut = Z_value(after_branch, Node_data)

alpha_SC = alpha(SC_branch, Node_data)
alpha_after_cut = alpha(after_branch, Node_data)
#初始化参数与仿真
PM1, PM2, PM3 = 1.5, 1, 3
omegaB = 18000
x, y, t = sy.symbols('x,y,t')
dw1,dw2,dw3 = sy.symbols('\Delta\omega_1,\Delta\omega_2,\Delta\omega_3')
pe1,pe2,pe3 = sy.symbols('P_e1,P_e2,P_e3')
dt1,dt2,dt3= sy.symbols('\delta_1, \delta_2, \delta_3')
all_var = (dw1,dw2,dw3,dt1,dt2,dt3,pe1,pe2,pe3)
#稳态开始
deriv_dict_start= {dw1:0,dw2:0,dw3:0,dt1:omegaB*dw1,dt2:omegaB*dw2,dt3:omegaB*dw3,
                   'co_vars':{pe1:pe1-1.5,pe2:pe2-1,pe3:pe3-3},
                   'co_init_values':(1.5,1,3),
                   'init_values':(0,0,0,38.64149,29.44408,15.52411)}
x_stable, y_stable = hiding_trapezium(deriv_dict_start, limit=(0,0.05), n=1)
print(y_stable)
#短路开始
pe1_expr = E1**2/z_value_SC[0,0]*sy.sin(alpha_SC[0,0])+E1*E2/z_value_SC[0,1]*sy.sin((dt1-dt2)*sy.pi/180-alpha_SC[0,1])+E1*E3/z_value_SC[0,2]*sy.sin((dt1-dt3)*sy.pi/180-alpha_SC[0,2])
pe2_expr = E2**2/z_value_SC[1,1]*sy.sin(alpha_SC[1,1])+E1*E2/z_value_SC[1,0]*sy.sin((dt2-dt1)*sy.pi/180-alpha_SC[1,0])+E2*E3/z_value_SC[1,2]*sy.sin((dt2-dt3)*sy.pi/180-alpha_SC[1,2])
pe3_expr = E3**2/z_value_SC[2,2]*sy.sin(alpha_SC[2,2])+E1*E3/z_value_SC[2,0]*sy.sin((dt3-dt1)*sy.pi/180-alpha_SC[2,0])+E2*E3/z_value_SC[2,1]*sy.sin((dt3-dt2)*sy.pi/180-alpha_SC[2,1])

short_init_values = [i[-1] for i in y_stable.values()]
short_dict = dict(zip(all_var, short_init_values))
Pe_short = [float(subs(pe1_expr,short_dict)),
           float(subs(pe2_expr,short_dict)),
           float(subs(pe3_expr,short_dict))]
deriv_dict_short = {dw1:(PM1-pe1)/10,dw2:(PM2-pe2)/7,dw3:(PM3-pe3)/15,
                    dt1:omegaB*dw1,dt2:omegaB*dw2,dt3:omegaB*dw3,
                    "co_vars":{pe1:pe1-pe1_expr,pe2:pe2-pe2_expr,pe3:pe3-pe3_expr},
                    'co_init_values':Pe_short,
                    'init_values':short_init_values[:-3]}
print(deriv_dict_short)
x_short,y_short = hiding_trapezium(deriv_dict_short, limit=(0.05,0.051), n=1)
print(y_short)
#故障切除
pe1_expr_cut = E1**2/z_value_after_cut[0,0]*sy.sin(alpha_after_cut[0,0])+E1*E2/z_value_after_cut[0,1]*sy.sin((dt1-dt2)*sy.pi/180-alpha_after_cut[0,1])+E1*E3/z_value_after_cut[0,2]*sy.sin((dt1-dt3)*sy.pi/180-alpha_after_cut[0,2])
pe2_expr_cut = E2**2/z_value_after_cut[1,1]*sy.sin(alpha_after_cut[1,1])+E1*E2/z_value_after_cut[1,0]*sy.sin((dt2-dt1)*sy.pi/180-alpha_after_cut[1,0])+E2*E3/z_value_after_cut[1,2]*sy.sin((dt2-dt3)*sy.pi/180-alpha_after_cut[1,2])
pe3_expr_cut = E3**2/z_value_after_cut[2,2]*sy.sin(alpha_after_cut[2,2])+E1*E3/z_value_after_cut[2,0]*sy.sin((dt3-dt1)*sy.pi/180-alpha_after_cut[2,0])+E2*E3/z_value_after_cut[2,1]*sy.sin((dt3-dt2)*sy.pi/180-alpha_after_cut[2,1])

cut_init_values = [i[-1] for i in y_short.values()]
cut_dict = dict(zip(all_var, cut_init_values))
Pe_cut = [float(subs(pe1_expr_cut,cut_dict)),
           float(subs(pe2_expr_cut,cut_dict)),
           float(subs(pe3_expr_cut,cut_dict))]
deriv_dict_short = {dw1:(PM1-pe1)/10,dw2:(PM2-pe2)/7,dw3:(PM3-pe3)/15,
                    dt1:omegaB*dw1,dt2:omegaB*dw2,dt3:omegaB*dw3,
                    "co_vars":{pe1:pe1-pe1_expr_cut,pe2:pe2-pe2_expr_cut,pe3:pe3-pe3_expr_cut},
                    'co_init_values':Pe_cut,
                    'init_values':cut_init_values[:-3]}
x_cut,y_cut = hiding_trapezium(deriv_dict_short, limit=(0.051,1.15), n=20)
fig, axes = plt.subplots(1,2,figsize=(16,10))
Delta, Graphs = (dt1,dt2,dt3), ((x_stable,y_stable), (x_short,y_short), (x_cut,y_cut))
for graph in Graphs:
    for delta in Delta:
        axes[0].plot(graph[0], graph[1][delta], label='$'+str(delta)+'$')
    axes[1].plot(graph[0], graph[1][Delta[0]]-graph[1][Delta[1]],label='$\delta_{12}$')
    axes[1].plot(graph[0], graph[1][Delta[0]]-graph[1][Delta[2]],label='$\delta_{13}$')
    axes[1].plot(graph[0], graph[1][Delta[1]]-graph[1][Delta[2]],label='$\delta_{23}$')

plt.grid()
fig.legend()