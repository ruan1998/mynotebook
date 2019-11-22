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

def Pe(E=None, Delta=None, Z_value=None, Alpha=None):
    '''三机系统Pe方程'''
    Pe_list = []
    for i in range(3):
        j,k = {0,1,2} - {i}
        pe = E[i]**2/Z_value[i,i]*sy.sin(Alpha[i,i])+E[i]*E[j]/Z_value[i,j]*sy.sin((Delta[i]-Delta[j])*sy.pi/180-Alpha[i,j])+E[i]*E[k]/Z_value[i,k]*sy.sin((Delta[i]-Delta[k])*sy.pi/180-Alpha[i,k])
        Pe_list.extend(pe)
    return Pe_list

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
    iter_keys, iter_values = np.array(list(init_values.keys())),np.array(list(init_values.values()))
    Jacobian = sy.diff(Functions, iter_keys).transpose()
    Functions = sy.lambdify(all_var, Functions,'numpy')
    nocoverage = True
    while(nocoverage):
        #前一次所有faunction的值构成的array
        Functions_values = np.array(Functions(*iter_values))
        Jacobian_values = Jacobian.subs(dict(zip(iter_keys,iter_values)))
        #将雅可比矩阵转成np.float32类型，否则无法求解
        Jacobian_values = np.array(Jacobian_values.tomatrix(),dtype=np.float32)
        iter_values -= solve(Jacobian_values,Functions_values)
        Functions_values_again = np.array(Functions(*iter_values))
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
        #将derivative剩下的元素与co_vars对应的值转换为array
        Functions = np.array(list(new_derivative.values()) + list(co_vars.values()))
        iter_values = newton_laphson(Functions, iter_values, e=e)
        for key,value in iter_values.items():
            all_iter_values[key][i] = value
            
    return t,all_iter_values

#读取节点分支数据
file_name = 'data/'
Node_data = np.load(f"{file_name}Node_data.npy")
branch_data = np.load(f"{file_name}branch_data.npy")
trans_data = np.load(f"{file_name}trans_data.npy")
E = Node_data[np.argwhere(Node_data[:,1]==2),2]
#初始化参数与仿真
PM1, PM2, PM3 = 1.5, 1, 3
omegaB = 18000
x, y, t = sy.symbols('x,y,t')
dw1,dw2,dw3 = sy.symbols('\Delta\omega_1,\Delta\omega_2,\Delta\omega_3')
pe1,pe2,pe3 = sy.symbols('P_e1,P_e2,P_e3')
dt1,dt2,dt3= sy.symbols('\delta_1, \delta_2, \delta_3')
all_var = (dw1,dw2,dw3,dt1,dt2,dt3,pe1,pe2,pe3)
Delta = (dt1,dt2,dt3)

#对节点支路数据处理，获取短路后与故障切除后输入阻抗、转移阻抗
SC_branch = branch_data_for_imp_SC(branch_data,Node_data,trans_data)
after_branch = branch_data_for_imp_afterCut(branch_data,Node_data,trans_data)

z_value_SC = Z_value(SC_branch, Node_data)
z_value_after_cut = Z_value(after_branch, Node_data)

alpha_SC = alpha(SC_branch, Node_data)
alpha_after_cut = alpha(after_branch, Node_data)

#稳态开始
deriv_dict_start= {dw1:0,dw2:0,dw3:0,dt1:omegaB*dw1,dt2:omegaB*dw2,dt3:omegaB*dw3,
                   'co_vars':{pe1:pe1-1.5,pe2:pe2-1,pe3:pe3-3},
                   'co_init_values':(1.5,1,3),
                   'init_values':(0,0,0,38.64149,29.44408,15.52411)}
x_stable, y_stable = hiding_trapezium(deriv_dict_start, limit=(0,0.05), n=1)
print(y_stable)
#短路开始
pe1_expr, pe2_expr, pe3_expr = Pe(E,Delta, z_value_SC, alpha_SC)
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
x_short,y_short = hiding_trapezium(deriv_dict_short, limit=(0.05,0.15), n=5)
print(y_short)
#故障切除
pe1_expr_cut, pe2_expr_cut, pe3_expr_cut = Pe(E,Delta, z_value_after_cut, alpha_after_cut)
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
x_cut,y_cut = hiding_trapezium(deriv_dict_short, limit=(0.15,3), n=100)

x = np.concatenate((x_stable,x_short,x_cut))
y = {i:np.concatenate((y_stable[i],y_short[i],y_cut[i])) for i in (dt1,dt2,dt3,dw1,dw2,dw3)}

fig, axes = plt.subplots(1,3,figsize=(16,8))
Delta, graph = ((dt1,dw1),(dt2,dw2),(dt3,dw3)), (x, y)
for delta, omega in Delta:
    axes[1].plot(graph[0], graph[1][delta], label='$'+str(delta)+'$')
    axes[0].plot(graph[0], graph[1][omega], label='$'+str(omega)+'$')
    axes[1].legend(loc='best', prop={'size':16})
    axes[0].legend(loc='best', prop={'size':16})
    
axes[2].plot(graph[0], graph[1][dt1]-graph[1][dt2],label='$\delta_{12}$')
axes[2].plot(graph[0], graph[1][dt1]-graph[1][dt3],label='$\delta_{13}$')
axes[2].plot(graph[0], graph[1][dt2]-graph[1][dt3],label='$\delta_{23}$')
axes[2].legend(loc='best', prop={'size':16})  

axes[0].grid();axes[1].grid();axes[2].grid()
axes[0].set_xlim(0, 3)
axes[1].set_xlim(0, 3)
axes[2].set_xlim(0, 3)
plt.tight_layout()
plt.show()
# plt.savefig('stablity_3s.png',bbox_inches = 'tight')