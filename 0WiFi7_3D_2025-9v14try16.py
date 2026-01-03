import time
# import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use('Agg')
from itertools import combinations
import numpy
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import torch, argparse
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import math
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
from numpy import random
from copy import deepcopy
# torch.autograd.set_detect_anomaly(True)
from scipy.spatial import Delaunay

from joblib import Parallel, delayed, cpu_count
from tqdm import tqdm
####Method

Throughput_model_verify=0

For4s=0


# 4StageMethod
FourStage=For4s# when forstage==1, Num_of_interferences should be set as 0
# Change=0  #增加对比方法，但是不改变原来方法
start_greedy=For4s
start_uniform=0 #
start_random=0
FirstStageOnly=0 #i.e., Greedy

RandomMethod=0


#nloptMethod
NloptMethod=0
# if NloptMethod==1:
#     Change=1  #增加对比方法，会改变原来方法个别语句

global LocalOptimizerMethod

LocalOptimizerMethod=0

#DeepRL

#plotfigure
plotFigure=0

global BestThroughput

BestThroughput=0 #Initialized to 0 存放所有迭代中至今最优的值，每次新的迭代要比它好才能选取。

CannotBeCovered=1 #定义一个全局变量。在函数内部要用它时，要用global【在主函数中其实可不用】

Exhaustive_search=0#FourStage=1 and Exhaustive_search=1 for comparison
ES=0  #for operation phase
ES_method_for_oper=0 #run Func_ES()

run=1
rhoH=6#Mbsp(4,5,6)
P_InterferenceTop_dBm=-64   #in dBm(-60,-70,-80)
#power level of interference sources
#P_interference=[-20, -17, -14,-11,-8,-5,-2,0,3,6,8]
GenerateStaticStas=0
GenerateStaticStas_unUniform=0

global NumOfSTAs, Num_Of_Static_STAs
if Exhaustive_search==1:
    NumOfSTAs = 400
    Num_Of_Static_STAs=100
else:
    NumOfSTAs = 2800
    Num_Of_Static_STAs=200
    
    # NumOfSTAs = 800
    # Num_Of_Static_STAs=200
    
if ES==1:
    NumOfSTAs = 400
    Num_Of_Static_STAs=100

#For exhaustive search
if Exhaustive_search==1:
    RegionLength = 50
    RegionWidth = 50
else:
    # RegionLength = 80
    # RegionWidth = 60
    RegionLength = 120
    RegionWidth = 100

if ES==1: #exhaustive_search, ES, what are the difference?
    RegionLength = 50
    RegionWidth = 50

P_interference=3#in dBm (-8,-2,4)(0,3,6)??

#++++++++++++++++++
Num_of_interferences=20

GridLength=10 #in m
CandidateCites=[]
KK=0 # Fault tolerance degree

#the ceiling height of the 3D venue
global height
height=9
#the height of a station
global h_sta
h_sta=1.5
#the actual height of a station
global H_sta
H_sta=height-h_sta


#Find generation of stations
#Gen_Loc_of_STAs




plot_layout_of_RPs=0 #画静态参考点
PlotLayoutOfInts=0 #画干扰源
plotReward=1

# Safety margin test, For Table IX , By using "###P_InterferenceTop_dBm" to search the setting for 
# P_interferenceTop_W
SafetyMargin=0

global MaxThroughput
MaxThroughput=0
global MinThroughput
MinThroughput=0

global APsthreshold
APsthreshold=(RegionLength/GridLength)*(RegionWidth/GridLength)

global APLoc48
APLoc48=0

Num_of_antennas=4 #考虑空分复用,MIMO中，可以多个数据流

labda24G=12.5 # in cm 各个频段的波长
labda5G1=6 #in cm
labda5G2=6
labda6G=5



def define_beam_variables():
    import numpy as np
    global dBi, dBi_Rx
    global G_arr_AP, G_arr_sta
    global G_beam24G_AP, G_beam5G1_AP, G_beam5G2_AP, G_beam6G_AP
    global G_beam24G_sta, G_beam5G1_sta, G_beam5G2_sta, G_beam6G_sta
    
    Num_of_antennas_beam=4

    dBi = 0  # Tx
    dBi_Rx = 0  # Rx

    G_arr_AP = 10 * np.log10(Num_of_antennas_beam)
    G_arr_sta = 10 * np.log10(2)

    # AP 波束增益
    G_beam24G_AP = 10*np.log10((4*3.14*Num_of_antennas_beam*6.25*6.25)/(12.5*12.5))
    G_beam5G1_AP = 10*np.log10((4*3.14*Num_of_antennas_beam*3*3)/(6*6))
    G_beam5G2_AP = 10*np.log10((4*3.14*Num_of_antennas_beam*3*3)/(6*6))
    G_beam6G_AP = 10*np.log10((4*3.14*Num_of_antennas_beam*2.5*2.5)/(5*5))

    # STA 波束增益
    G_beam24G_sta = 10*np.log10((4*3.14*2*6.25*6.25)/(12.5*12.5))
    G_beam5G1_sta = 10*np.log10((4*3.14*2*3*3)/(6*6))
    G_beam5G2_sta = 10*np.log10((4*3.14*2*3*3)/(6*6))
    G_beam6G_sta = 10*np.log10((4*3.14*2*2.5*2.5)/(5*5))








Num_of_antennas_beam=4 #考虑4个波束

global dBi,dBi_Rx,G_arr_AP,G_beam24G_AP,G_beam5G1_AP,G_beam5G2_AP,G_beam6G_AP,G_arr_sta
global G_beam24G_sta,G_beam5G1_sta,G_beam5G2_sta,G_beam6G_sta
dBi=0 # dBi for Tx
dBi_Rx=0 #dBi for RX
G_arr_AP=10*np.log10(Num_of_antennas_beam)
G_arr_sta=10*np.log10(2)

G_beam24G_AP=10*np.log10((4*3.14*Num_of_antennas_beam*6.25*6.25)/(12.5*12.5))  #四个天线，2*2排列,加上边界，面积为(2*d)*(2*d)
G_beam5G1_AP=10*np.log10((4*3.14*Num_of_antennas_beam*3*3)/(6*6))
G_beam5G2_AP=10*np.log10((4*3.14*Num_of_antennas_beam*3*3)/(6*6))
G_beam6G_AP=10*np.log10((4*3.14*Num_of_antennas_beam*2.5*2.5)/(5*5))


G_beam24G_sta=10*np.log10((4*3.14*2*6.25*6.25)/(12.5*12.5))  #2个天线，1*2排列,加上边界，面积为(2*d)*(1*d)
G_beam5G1_sta=10*np.log10((4*3.14*2*3*3)/(6*6))
G_beam5G2_sta=10*np.log10((4*3.14*2*3*3)/(6*6))
G_beam6G_sta=10*np.log10((4*3.14*2*2.5*2.5)/(5*5))

# print(G_arr_AP,G_beam24G_AP,G_beam5G1_AP,G_beam5G2_AP,G_beam6G_AP)
# input()

# print(G_arr_sta,G_beam24G_sta,G_beam5G1_sta,G_beam5G2_sta,G_beam6G_sta)
# input()

# global episode_timesteps
# episode_timesteps=50
# episode_timesteps # define the number of steps in each episode
Ver='Uni_v3-test'


# Generate_c=1#for interference sources
#++++++++++++++++++
if FourStage==1 or start_greedy==1 or start_uniform==1 or start_random==1 or FirstStageOnly==1 or RandomMethod==1 or LocalOptimizerMethod==1:
    Num_of_interferences=0




#++++++++++++++++++For fixed method
FixedResource_f=0
Uniformly_Select=0 # uniformly selected channels and poler levels and then fixed
C_P_from_planneingPhase=0 #power levels and channels are from planning phase

Uniformly_Select_C_P=0
if Uniformly_Select_C_P==1:
    ap=np.loadtxt('ap_info.csv')
    Power_result = np.array([[0, 0, 0, 0]])
    Channel_result = np.array([[0, 0, 0, 0]])
    for i in range(len(ap)):
    # print('r_24G test')
    # print(AP[i].r_24G)
    # input()   
    
        Power_result = np.append(Power_result, [[random.randint(24, 28), random.randint(25, 29), random.randint(25, 29), random.randint(27, 31)]], axis=0)
        Channel_result = np.append(Channel_result, [[random.randint(1,5), random.randint(5, 20), random.randint(20, 28), random.randint(28, 59)]], axis=0)
                
    Power_temp = np.delete(Power_result, 0, axis=0)
    Channel_temp = np.delete(Channel_result, 0, axis=0)
    
    # print(ap)    
    
    np.savetxt('Power_info_ui.csv', Power_temp) #uniformly selected
    np.savetxt('Channel_info_ui.csv', Channel_temp)
    
    print("The channels and power levels for fixed method have been generated!")
    input()



MinDisFromINtToNode=3




time_start=time.time()
# ============================================
#Running settings
GetMovingPattern=0
MovingSteps=100
# samplingPeriod=30 #in s
samplingPeriod=5 #in s
StatisticalPeriod=3600*4 #in s, i.e., 4h * 3600s/h
TotalSamples=int(StatisticalPeriod/samplingPeriod)
# ============================================

Gamma=0








if FixedResource_f==1 and C_P_from_planneingPhase==1:  #the channel, power of APs, the locations of ints. are loaded from files.
    Ver='_Fixed_V1'
    C_fix=np.loadtxt('Channel_info.csv')
    P_fix=np.loadtxt('Power_info.csv')
    X_interference=np.loadtxt('X_ints.csv')
    Y_interference=np.loadtxt('y_ints.csv')
    
if FixedResource_f==1 and Uniformly_Select==1:  #the channel, power of APs, the locations of ints. are loaded from files.
    Ver='_Fixed_V1'
    C_fix=np.loadtxt('Channel_info_ui.csv')
    P_fix=np.loadtxt('Power_info_ui.csv')
    X_interference=np.loadtxt('X_ints.csv')
    Y_interference=np.loadtxt('y_ints.csv')
#
UniIns=1
if UniIns==1:
    X_interference1 = random.uniform(0, RegionLength, size=(1, Num_of_interferences))
    Y_interference1 = random.uniform(0, RegionWidth, size=(1, Num_of_interferences))
    Z_interference1 = random.uniform(0, height, size=(1, Num_of_interferences))
    X_interference=X_interference1[0]
    Y_interference=Y_interference1[0]
    Z_interference=Z_interference1[0]
    
    np.savetxt('X_ints.csv',X_interference)
    np.savetxt('y_ints.csv',Y_interference)
    np.savetxt('z_ints.csv',Z_interference)
    
    print("The locations of interference sources have been generated!")
    # input()
    
else:
# print(X_interference1[0])
# print(Y_interference1[0])
# input()
    aaa=1
    # if Num_of_interferences==20:
    #     # X_interference = random.uniform(0, RegionLength, size=(1, Num_of_interferences))
    #     # Y_interference = random.uniform(0, RegionWidth, size=(1, Num_of_interferences))
    #     # X_interference=[]
    #     # Y_interference=[]
    #     # for Int in range(Num_of_interferences):
    #     # # X_interference=[20,40,20,40,60,30,50,30,50,60,20,40,20,40,30]
    #     # # Y_interference=[20,20,40,40,30,30,30,50,50,40,30,30,50,50,40]
    #     #     X_interference.append()
    #     #     IS[i].c = random.randint(1, len(C) + 1)
    #     # X_interference=[20,60,20,40,40,30,50,30,50,60,20,30,40,50,60,10,40,70,40,40]
    #     # Y_interference=[20,20,40,12,24,10,10,50,50,40,30,30,36,30,30,30,0,30,60,48]
    #     X_interference=[10,30,50,70,10,30,50,70,10,30,50,70,10,30,50,70,10,30,50,70]
    #     Y_interference=[10,10,10,10,20,20,20,20,30,30,30,30,40,40,40,40,50,50,50,50]

    # if Num_of_interferences==15:
    #     # X_interference=[20,40,20,40,60,30,50,30,50,60,20,40,20,40,30]
    #     # Y_interference=[20,20,40,40,30,30,30,50,50,40,30,30,50,50,40]
    #     X_interference=[20,60,20,40,40,30,50,30,50,60,20,30,40,50,60]
    #     Y_interference=[20,20,40,40,20,10,10,50,50,40,30,30,30,30,30]

    # # if Num_of_interferences==10:
    # #     X_interference=[20,60,20,40,40,30,50,30,50,60]
    # #     Y_interference=[20,20,40,40,20,10,10,50,50,40]

    # if Num_of_interferences==5:
        
        
    #     X_interference=[20,60,20,60,40]
    #     Y_interference=[20,20,40,40,30]

# np.savetxt('X_ints.csv',X_interference)
# np.savetxt('y_ints.csv',Y_interference)

if Num_of_interferences!=0:
    X_interference=np.loadtxt('X_ints.csv') # the locations of ints. are from the files.
    Y_interference=np.loadtxt('y_ints.csv')
    Z_interference=np.loadtxt('z_ints.csv')

def sort_descending_with_index(lst):
    return sorted(enumerate(lst), key=lambda x: x[1], reverse=True)

# print(X_interference,Y_interference)
#
# input()

def Func_GenerateGrid():
    a_temp=[]
    TempLength=np.arange(GridLength/2,RegionLength,GridLength)
    TempWidth=np.arange(GridLength/2,RegionWidth,GridLength)
    for j in range(len(TempWidth)):
        for i in range(len(TempLength)):
            a_temp.append(TempLength[i])
            a_temp.append(TempWidth[j])
            CandidateCites.append(a_temp)
            # print(a_temp)
            # print(CandidateCites)
            a_temp = []

            # input()

Func_GenerateGrid()
# print(len(CandidateCites))
# print(CandidateCites)
# print(CandidateCites[1][0])
# print(CandidateCites[1][1])
# print(CandidateCites[2])
# print(CandidateCites[17])
# input()






# Coordinate_x = random.uniform(0, RegionLength, size=(1, NumOfAPs))
# Coordinate_y = random.uniform(0, RegionWidth, size=(1, NumOfAPs))

# Set GenerateStaticStas=1 to generate locations of stations. Then set GenerateStaticStas=0 for comparison
# with the same locations of stations

if GenerateStaticStas==1:

    X_StaticSTAs=random.uniform(0, RegionLength, size=(1, Num_Of_Static_STAs))
    Y_StaticSTAs=random.uniform(0, RegionWidth, size=(1, Num_Of_Static_STAs))
    Z_StaticSTAs=random.uniform(0, height, size=(1, Num_Of_Static_STAs))
# 合并为一张表：每行 [x, y, z]
    data = np.vstack((X_StaticSTAs, Y_StaticSTAs, Z_StaticSTAs)).T
    # 保存到单个文件
    np.savetxt(f'Stas_run{run}.csv', data, delimiter=',', header='X,Y,Z', comments='')
    # np.savetxt(f'x_Sstas_run{run}.csv',X_StaticSTAs)
    # np.savetxt(f'y_Sstas_run{run}.csv',Y_StaticSTAs)
    # np.savetxt(f'z_Sstas_run{run}.csv',Z_StaticSTAs)
    
    print("The locations of stations have been generated!")

    # input()
if GenerateStaticStas_unUniform==1: 
     
    import numpy as np 
    import matplotlib.pyplot as plt 
    from mpl_toolkits.mplot3d import Axes3D 
 
 
    # 计算两半的节点数量（处理奇数情况） 
    part=1
    half_num = Num_Of_Static_STAs // part
    remaining_num = Num_Of_Static_STAs - half_num 
     
    # 第一半：空间均匀分布的节点 
    X_uniform = np.random.uniform(0, RegionLength, size=half_num) 
    Y_uniform = np.random.uniform(0, RegionWidth, size=half_num) 
    Z_uniform = np.random.uniform(0, height, size=half_num) 
    uniform_points = np.column_stack((X_uniform, Y_uniform, Z_uniform)) 
     
    # 第二半：分布在四个竖直平面上的节点 
    # 为四个平面平均分配节点（处理不能整除的情况） 
    points_per_plane = [remaining_num // 4 + 1] * (remaining_num % 4) + [remaining_num // 4] * (4 - remaining_num % 4) 
     
    plane_points = [] 
    # 1. x=0 竖直平面（y-z平面） 
    n1 = points_per_plane[0] 
    y1 = np.random.uniform(0, RegionWidth, size=n1) 
    z1 = np.random.uniform(1e-9, height - 1e-9, size=n1)  # 严格排除上下平面边界 
    plane_points.append(np.column_stack((np.zeros(n1), y1, z1))) 
     
    # 2. x=RegionLength 竖直平面 
    n2 = points_per_plane[1] 
    y2 = np.random.uniform(0, RegionWidth, size=n2) 
    z2 = np.random.uniform(1e-9, height - 1e-9, size=n2) 
    plane_points.append(np.column_stack((np.full(n2, RegionLength), y2, z2))) 
     
    # 3. y=0 竖直平面（x-z平面） 
    n3 = points_per_plane[2] 
    x3 = np.random.uniform(0, RegionLength, size=n3) 
    z3 = np.random.uniform(1e-9, height - 1e-9, size=n3) 
    plane_points.append(np.column_stack((x3, np.zeros(n3), z3))) 
     
    # 4. y=RegionWidth 竖直平面 
    n4 = points_per_plane[3] 
    x4 = np.random.uniform(0, RegionLength, size=n4) 
    z4 = np.random.uniform(1e-9, height - 1e-9, size=n4) 
    plane_points.append(np.column_stack((x4, np.full(n4, RegionWidth), z4))) 
     
    # 合并四个平面的节点 
    plane_points = np.vstack(plane_points) 
     
    # 合并所有节点并打乱顺序 
    all_points = np.vstack((uniform_points, plane_points)) 
    np.random.shuffle(all_points)  # 打乱节点顺序，使两种分布混合 
     
    # 保存到文件 
    np.savetxt(f'Stas_run{run}.csv', all_points, delimiter=',', header='X,Y,Z', comments='') 
     
    # 3D可视化 
    fig = plt.figure(figsize=(10, 8)) 
    ax = fig.add_subplot(111, projection='3d') 
     
    # 绘制均匀分布的点（蓝色） 
    ax.scatter(uniform_points[:, 0], uniform_points[:, 1], uniform_points[:, 2],  
               c='blue', s=10, alpha=0.6, label='Distributed in 3D venue') 
     
    # 绘制四个平面上的点（用不同颜色区分四个平面） 
    # 从plane_points列表中提取四个平面的点（保持原始顺序） 
    if part==2:
        p1, p2, p3, p4 = plane_points[:n1], plane_points[n1:n1+n2], plane_points[n1+n2:n1+n2+n3], plane_points[n1+n2+n3:] 
        ax.scatter(p1[:, 0], p1[:, 1], p1[:, 2], c='red', s=10, alpha=0.6, label='Distributed in plane 1') 
        ax.scatter(p2[:, 0], p2[:, 1], p2[:, 2], c='green', s=10, alpha=0.6, label='Distributed in plane 2') 
        ax.scatter(p3[:, 0], p3[:, 1], p3[:, 2], c='orange', s=10, alpha=0.6, label='Distributed in plane 3') 
        ax.scatter(p4[:, 0], p4[:, 1], p4[:, 2], c='purple', s=10, alpha=0.6, label='Distributed in plane 4') 
     
    # 设置坐标轴标签和标题 
    ax.set_xlabel('X') 
    ax.set_ylabel('Y') 
    ax.set_zlabel('Z') 
    # ax.set_title(f'静态节点分布 (总数: {Num_Of_Static_STAs})') 
    ax.legend() 
     
    # 设置坐标轴范围与区域大小一致 
    ax.set_xlim(0, RegionLength) 
    ax.set_ylim(0, RegionWidth) 
    ax.set_zlim(0, height) 
     
    plt.tight_layout() 
    if part==1:
        plt.savefig('Uniform_stations.png',bbox_inches='tight',dpi=500,pad_inches=0.0)
    if part==2:
        plt.savefig('UN_Uniform_stations.png',bbox_inches='tight',dpi=500,pad_inches=0.0)
    plt.show()

# X_StaticSTAs = np.loadtxt(f'x_Sstas_run{run}.csv')
# Y_StaticSTAs = np.loadtxt(f'y_Sstas_run{run}.csv')
# Z_StaticSTAs = np.loadtxt(f'z_Sstas_run{run}.csv')

# input()

loaded_data = np.loadtxt(f'Stas_run1.csv', delimiter=',', skiprows=1)
X_StaticSTAs = loaded_data[:, 0]
Y_StaticSTAs = loaded_data[:, 1]
Z_StaticSTAs = loaded_data[:, 2]



if Exhaustive_search==1:
    Num_Of_Static_STAs=100 # there are other 200 stations (200 grid points)

    X_StaticSTAs=random.uniform(0, RegionLength, size=(1, Num_Of_Static_STAs))
    Y_StaticSTAs=random.uniform(0, RegionWidth, size=(1, Num_Of_Static_STAs))
    Z_StaticSTAs=random.uniform(0, height, size=(1, Num_Of_Static_STAs))

    np.savetxt('x_Sstas_ES.csv',X_StaticSTAs)
    np.savetxt('y_Sstas_ES.csv',Y_StaticSTAs)
    np.savetxt('z_Sstas_ES.csv',Z_StaticSTAs)

    X_StaticSTAs = np.loadtxt('x_Sstas_ES.csv')
    Y_StaticSTAs = np.loadtxt('y_Sstas_ES.csv')
    Z_StaticSTAs = np.loadtxt('z_Sstas_ES.csv')
# print(X_StaticSTAs)
# print(Y_StaticSTAs)
# print(Z_StaticSTAs)
# print('Generate Static Stas.')
# input()

# X_interference=random.uniform(0, RegionLength, size=(1, Num_of_interferences))
# Y_interference=random.uniform(0, RegionWidth, size=(1, Num_of_interferences))

if PlotLayoutOfInts==1:#最后画出干扰源的位置，3D的
    # plt.plot(X_interference, Y_interference, 'gp',markersize=8.)
    # # plt.plot(X_StaticSTAs,Y_StaticSTAs,'ko')
    # plt.scatter(X_interference, Y_interference,s=20,c='k')

    # plt.xlim(0, 80)
    # plt.ylim(0, 60)

    # plt.xticks(fontsize=13)
    # plt.yticks(fontsize=13)
    # plt.xlabel('The length of the region (m)', fontsize=13)
    # plt.ylabel('The width of the region (m)', fontsize=13)
    # plt.grid()
    
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 绘制3D点
    ax.scatter(X_interference, Y_interference,Z_interference,s=2,c='g', marker='o')  # c='r' 表示红色，marker='o' 表示圆形标记
    # ax.scatter(SSTA_XX, SSTA_YY,SSTA_ZZ,s=2,c='k', marker='o')  # c='r' 表示红色，marker='o' 表示圆形标记

    # 设置坐标轴标签
    ax.set_xlabel('Length')
    ax.set_ylabel('Width')
    ax.set_zlabel('Height') #左侧的height字看不到，看如何处理
    
    ax.set_box_aspect([1.7, 1.5, 0.8])#设置长宽高比例

    # 显示图形
    plt.show()
    
    
    
    
    
    if Num_of_interferences == 5:
        plt.savefig('Figure_5ins.png', bbox_inches='tight', dpi=500, pad_inches=0.0)
    if Num_of_interferences == 10:
        plt.savefig('Figure_10ins.png', bbox_inches='tight', dpi=500, pad_inches=0.0)
    if Num_of_interferences == 15:
        plt.savefig('Figure_15ins.png', bbox_inches='tight', dpi=500, pad_inches=0.0)
    if Num_of_interferences == 20:
        plt.savefig('Figure_20ins.png', bbox_inches='tight', dpi=500, pad_inches=0.0)




    plt.show()


#Path loss model settings
#Channel model F
d_BP=30 #in m
SNR=[3,6,8,11,15,20,21,26,28,31,33,38,40] #33或以下是802.11ax 38是802.11be
Bits_per_Hz=[0.5,1,1.5,2,3,4.5,5,6,6.67,7.5,8.33,9,9.6]#8.33或以下是802.11ax 9是802.11be,12bits*3/4。coding rate 3/4。

def Get_pbs_per_Hz(SINR):
    if SINR<3:
        return 0
    if 3<=SINR<6:
        return 0.5
    if 6<=SINR<8:
        return 1
    if 8<=SINR<11:
        return 1.5
    if 11<=SINR<15:
        return 2
    if 15<=SINR<20:
        return 3
    if 20<=SINR<21:
        return 4.5
    if 21<=SINR<26:
        return 5
    if 26<=SINR<28:
        return 6
    if 28<=SINR<31:
        return 6.67
    if 31<=SINR<33:
        return 7.5
    if 33<=SINR<38:
        return 8.33
    if 38<=SINR<40:
        return 9
    if SINR>=40:
        return 9.6

# a=Get_pbs_per_Hz(31)
# print(a)
# input()
Band=['24G','5G1','5G2','6G']
fc_2dot4G=2437000000 #in Hz
fc_5GI=5250000000 #in Hz
fc_5GII=5787500000 #in Hz
fc_6G=6185000000 #in Hz
Wsc=78125 #Hz
def Sigma(d):#Shadow fading standard deviation (dB)
    if d<=d_BP:
        SD=3
    else:
        SD=6
    return SD

# x=abs(np.random.normal(0,Sigma(50)))
#
#
# print(Sigma(50)/((2*pi)**(0.5)))
# input()

def PL(d,f):#f的单位，是GHz，还是MHZ,还是Hz?要验证一下。文献是用Hz。
    if d<=d_BP:
        # PathLoss=20*np.log10(d)+20*np.log10(f)-147.5+np.random.normal(0,Sigma(d))
        if d<1:
            d=MinDisFromINtToNode
            # print('ddd=',d)
            # print('log1',20 * np.log10(d))
            # print('log2',20 * np.log10(f))
            # print('f',f)
            # print(20 * np.log10(d)+20 * np.log10(f)+Sigma(d)/((2*pi)**(0.5)))
            # PathLoss = 20 * np.log10(d) + 20 * np.log10(f) - 147.5 + Sigma(d) / ((2 * pi) ** (0.5))
            # print('pathloss=',PathLoss)
            # input()

        PathLoss = 20 * np.log10(d) + 20 * np.log10(f) - 147.5+Sigma(d)/((2*np.pi)**(0.5))
        # print('pathloss=',PathLoss)
        # input()
    else:
        # PathLoss=20*np.log10(d_BP)+20*np.log10(f)-147.5+np.random.normal(0,Sigma(d))+35*np.log10((d)/(d_BP))
        PathLoss = 20 * np.log10(d_BP) + 20 * np.log10(f) - 147.5+35*np.log10((d)/(d_BP))+Sigma(d)/((2*np.pi)**(0.5))
    return PathLoss

def Range(PT,PR,B):
     #in dBi for directional antenna
    # if B=='24G':
    #     d=d_BP*10**((PT-PR-20*np.log10(d_BP)-20*log10(fc_2dot4G)+147.5-6/((2*pi)**(0.5)))/35)
    # if B=='5G1':
    #     d=d_BP*10**((PT-PR-20*np.log10(d_BP)-20*log10(fc_5GI)+147.5-6/((2*pi)**(0.5)))/35)
    # if B=='5G2':
    #     d=d_BP*10**((PT-PR-20*np.log10(d_BP)-20*log10(fc_5GII)+147.5-6/((2*pi)**(0.5)))/35)
    # if B=='6G':
    #     d=d_BP*10**((PT-PR-20*np.log10(d_BP)-20*log10(fc_6G)+147.5-6/((2*pi)**(0.5)))/35)
    if B=='24G':     
       
        
        d=d_BP*10**((PT+dBi-PR-20*np.log10(d_BP)-20*np.log10(fc_2dot4G)+147.5-6/((2*np.pi)**(0.5)))/35)
        # d=(d**2-H_sta**2)**(0.5)

        # dd=2
    if B=='5G1':
        
        
        d=d_BP*10**((PT+dBi-PR-20*np.log10(d_BP)-20*np.log10(fc_5GI)+147.5-6/((2*np.pi)**(0.5)))/35)
        # d = (d ** 2 - H_sta**2) ** (0.5)
    if B=='5G2':
        
        
        d=d_BP*10**((PT+dBi-PR-20*np.log10(d_BP)-20*np.log10(fc_5GII)+147.5-6/((2*np.pi)**(0.5)))/35)
        # d = (d ** 2 - H_sta**2) ** (0.5)
    if B=='6G':
        d=d_BP*10**((PT+dBi-PR-20*np.log10(d_BP)-20*np.log10(fc_6G)+147.5-6/((2*np.pi)**(0.5)))/35)
        # d = (d ** 2 - H_sta**2) ** (0.5)
    # print("H_sta:",H_sta)
    # print('range tset')
    # print(d)
    # input()
    return d

# x=PL(61,fc_2dot4G)
# b=23-x
# print(x,b)
# input()
#


# b2=PL(40,6425000000)
# print(b,b2)
# print('=====')
# a1=np.log10(2402000000)
# a2=np.log10(2472000000)
# b1=np.log10(5150000000)
# b2=np.log10(5350000000)
# c1=np.log10(5725000000)
# c2=np.log10(5850000000)
# d1=np.log10(5925000000)
# d2=np.log10(6425000000)
# print(a1,a2)
# print(b1,b2)
# print(c1,c2)
# print(d1,d2)
# input()

#power level
P_24G=[24,25,26,27]
P_5G1=[25,26,27,28]
P_5G2=[25,26,27,28]
P_6G=[27,28,29,30]

P=[26]

P_noise_dBM=-90 #in dBm
P_noise_W=10**((P_noise_dBM)/10)/1000 #in W

global P_interferenceTop_W

P_interferenceTop_W=10**((P_InterferenceTop_dBm)/10)/1000


if P_InterferenceTop_dBm==0:
    P_interferenceTop_W=0
print("P_ITop=",P_InterferenceTop_dBm)
print("P_interferenceTop_W",P_interferenceTop_W)
print('P_noise_W=',P_noise_W)

P_target_W=100*(P_noise_W+P_interferenceTop_W) # At least 20 dB, i.e., 10^2 = 100 times
P_target_dBm=10*np.log10(P_target_W*1000) # Change W to mW
print('P_target_W',P_target_W)
print('P_target_dBm',P_target_dBm)

# input()

d1=Range(max(P_24G),P_target_dBm,'24G')
d2=Range(max(P_5G1),P_target_dBm,'5G1')
d3=Range(max(P_5G2),P_target_dBm,'5G2')
d4=Range(max(P_6G),P_target_dBm,'6G')



# print("d1:", d1, type(d1))
# print("d2:", d2, type(d2))
# print("d3:", d3, type(d3))
# print("d4:", d4, type(d4))



d1_24G = float(np.asarray(d1).item())
d2_5G1 = float(np.asarray(d2).item())
d3_5G2 = float(np.asarray(d3).item())
d4_6G = float(np.asarray(d4).item())

# D = min(d1, d2, d3, d4)

D = np.min([d1, d2, d3, d4])  #coverage

#The range of side lobes of stations
IntRange_sta24G=Range(max(P_24G)-13,P_target_dBm,'24G')
IntRange_sta5G1=Range(max(P_5G1)-13,P_target_dBm,'5G1')
IntRange_sta5G2=Range(max(P_5G1)-13,P_target_dBm,'5G2')
IntRange_sta6G=Range(max(P_6G)-13,P_target_dBm,'6G')

Dij_24G=d1_24G+d1_24G+IntRange_sta24G
Dij_5G1=d2_5G1+d2_5G1+IntRange_sta5G1
Dij_5G2=d3_5G2+d3_5G2+IntRange_sta5G2
Dij_6G=d4_6G+d4_6G+IntRange_sta6G

# print(IntRange_sta24G,IntRange_sta5G1,IntRange_sta5G2,IntRange_sta6G)

# print("a=",IntRange_sta24G+d1)
# input()


# print(type(d1))
# <class 'numpy.float64'>
# input()
# D = min(float(d1), float(d2), float(d3), float(d4))
# D=min(d1,d2,d3,d4)
print('radio range:',d1,d2,d3,d4,D)
# print("press enter...")
# input()

def Radio_vector_settings():
    global Radio_vector_24G,Radio_vector_5G1,Radio_vector_5G2,Radio_vector_6G
    d1_24G = Range(P_24G[0], P_target_dBm, '24G')
    d1_5G1 = Range(P_5G1[0], P_target_dBm, '5G1')
    d1_5G2 = Range(P_5G2[0], P_target_dBm, '5G2')
    d1_6G = Range(P_6G[0], P_target_dBm, '6G')

    d2_24G = Range(P_24G[1], P_target_dBm, '24G')
    d2_5G1 = Range(P_5G1[1], P_target_dBm, '5G1')
    d2_5G2 = Range(P_5G2[1], P_target_dBm, '5G2')
    d2_6G = Range(P_6G[1], P_target_dBm, '6G')

    d3_24G = Range(P_24G[2], P_target_dBm, '24G')
    d3_5G1 = Range(P_5G1[2], P_target_dBm, '5G1')
    d3_5G2 = Range(P_5G2[2], P_target_dBm, '5G2')
    d3_6G = Range(P_6G[2], P_target_dBm, '6G')

    d4_24G = Range(P_24G[3], P_target_dBm, '24G')
    d4_5G1 = Range(P_5G1[3], P_target_dBm, '5G1')
    d4_5G2 = Range(P_5G2[3], P_target_dBm, '5G2')
    d4_6G = Range(P_6G[3], P_target_dBm, '6G')

    Radio_vector_24G=[d1_24G,d2_24G,d3_24G,d4_24G]
    Radio_vector_5G1=[d1_5G1,d2_5G1,d3_5G1,d4_5G1]
    Radio_vector_5G2=[d1_5G2,d2_5G2,d3_5G2,d4_5G2]
    Radio_vector_6G=[d1_6G,d2_6G,d3_6G,d4_6G]

    # print(Radio_vector_24G)
    # print(Radio_vector_5G1)
    # print(Radio_vector_5G2)
    # print(Radio_vector_6G)
    # input()
Radio_vector_settings()
# C=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]#2.4G and 5G band-I
# C=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
C=np.arange(1,59) #for interference sources

Generate_c=1
if Generate_c==1:
    C_for_IS=[]
    for ins in range(Num_of_interferences):
        
        if ES==1:
            C_for_IS.append(random.randint(1,5))
        else:
            C_for_IS.append(random.randint(1,len(C)+1))
    np.savetxt('C_for_ISs.csv',C_for_IS)
    # IS[i].c=random.randint(1,len(C)+1)
    # IS[i].c = C_for_IS[i]
    
    print("The channels for the ins. have been generated!")
    print(C_for_IS)
    # input()

if FourStage==0:
    C_for_IS=np.loadtxt('C_for_ISs.csv')



#Channels in different band
C_2dot4G=[1,2,3,4]
C_5GI=np.arange(5,20) # channel 5 to channel 19
C_5GII=np.arange(20,28)
C_6G=np.arange(28,59)


if ES==1: #for es method in a small case, i.e., the size of the set of channels is small
    #Channels in different band
    C_2dot4G=[1,2,3,4]
    C_5GI=np.arange(5,6) # channel 5 to channel 19
    C_5GII=np.arange(20,21)
    C_6G=np.arange(28,29)



def define_frequencies():
    """
    幂等地在当前进程的全局命名空间定义 Frequence... 和相关 C_ocs 列表。
    可在主进程或并行子进程内调用（安全）。
    调用后，脚本中原先通过 globals()[f'Frequence{F}_{B}'] 访问的变量会存在。
    """
    import numpy as np
    G = globals()

    # 如果已经定义了某些关键变量，就直接返回（避免重复赋值）
    # 这里用 Frequence20_24G 作为“是否已定义”的判断依据
    if 'Frequence20_24G' in G and 'Frequence20_5G1' in G and 'Frequence20_6G' in G:
        return

    # ---------- 2.4 GHz ----------
    G['Frequence20_24G'] = [1, 2, 3]
    G['Frequence40_24G'] = [4]

    # 通用 20/40/80 (若你需要这些全局名)
    G['Frequence20'] = [1,2,3,5,6,7,8,9,10,11,12]
    G['Frequence40'] = [4,13,14,15,16]
    G['Frequence80'] = [17,18]

    # OCS of 2.4G band
    G['C_ocs1'] = [2,4]
    G['C_ocs2'] = [3,4]

    # ---------- 5G - band I ----------
    G['Frequence20_5G1'] = np.arange(5,13)
    G['Frequence40_5G1'] = np.arange(13,17)
    G['Frequence80_5G1'] = [17,18]
    G['Frequence160_5G1'] = [19]

    G['C_ocs3'] = [5,13,17,19]
    G['C_ocs4'] = [6,13,17,19]
    G['C_ocs5'] = [7,14,17,19]
    G['C_ocs6'] = [8,14,17,19]
    G['C_ocs7'] = [9,15,18,19]
    G['C_ocs8'] = [10,15,18,19]
    G['C_ocs9'] = [11,16,18,19]
    G['C_ocs10'] = [12,16,18,19]

    # ---------- 5G - band II ----------
    G['Frequence20_5G2'] = [20,21,22,23,24]
    G['Frequence40_5G2'] = [25,26]
    G['Frequence80_5G2'] = [27]

    G['C_ocs11'] = [20,25,27]
    G['C_ocs12'] = [21,25,27]
    G['C_ocs13'] = [22,26,27]
    G['C_ocs14'] = [23,26,27]

    # ---------- 6G ----------
    G['Frequence20_6G'] = np.arange(28,44)
    G['Frequence40_6G'] = np.arange(44,52)
    G['Frequence80_6G'] = np.arange(52,56)
    G['Frequence160_6G'] = np.arange(56,58)
    G['Frequence320_6G'] = [58]

    # 若需要，也可把一个集中字典 Frequencies 写入全局（可选）
    # G['Frequencies'] = {
    #     '24G': {20: G['Frequence20_24G'], 40: G['Frequence40_24G']},
    #     '5G1': {20: G['Frequence20_5G1'], 40: G['Frequence40_5G1'], 80: G['Frequence80_5G1'], 160: G['Frequence160_5G1']},
    #     '5G2': {20: G['Frequence20_5G2'], 40: G['Frequence40_5G2'], 80: G['Frequence80_5G2']},
    #     '6G': {20: G['Frequence20_6G'], 40: G['Frequence40_6G'], 80: G['Frequence80_6G'], 160: G['Frequence160_6G'], 320: G['Frequence320_6G']}
    # }
    return








#############################2.4G
Frequence20_24G=[1,2,3]
Frequence40_24G=[4]

#20 MHz channels
Frequence20=[1,2,3,5,6,7,8,9,10,11,12]
Frequence40=[4,13,14,15,16]
Frequence80=[17,18]
# Frequence160=[19] #暂时先不考虑160MHz

# OCS of 2.4G band
C_ocs1=[2,4]
C_ocs2=[3,4]

#############################5GI
Frequence20_5G1=np.arange(5,13)
Frequence40_5G1=np.arange(13,17)
Frequence80_5G1=[17,18]
Frequence160_5G1=[19]

#OCS of 5G band-I
C_ocs3=[5,13,17,19]
C_ocs4=[6,13,17,19]
C_ocs5=[7,14,17,19]
C_ocs6=[8,14,17,19]
C_ocs7=[9,15,18,19]
C_ocs8=[10,15,18,19]
C_ocs9=[11,16,18,19]
C_ocs10=[12,16,18,19]

########################5GII
Frequence20_5G2=[20,21,22,23,24]
Frequence40_5G2=[25,26]
Frequence80_5G2=[27]

#OCS of 5G band-II
C_ocs11=[20,25,27]
C_ocs12=[21,25,27]
C_ocs13=[22,26,27]
C_ocs14=[23,26,27]

###########################6G
Frequence20_6G=np.arange(28,44)
Frequence40_6G=np.arange(44,52)
Frequence80_6G=np.arange(52,56)
Frequence160_6G=np.arange(56,58)
Frequence320_6G=[58]

#OCS of 6G band
C_ocs15=[28,44,52,56,58]
C_ocs16=[29,44,52,56,58]
C_ocs17=[30,45,52,56,58]
C_ocs18=[31,45,52,56,58]
C_ocs19=[32,46,53,56,58]
C_ocs20=[33,46,53,56,58]
C_ocs21=[34,47,53,56,58]
C_ocs22=[35,47,53,56,58]
C_ocs23=[36,48,54,57,58]
C_ocs24=[37,48,54,57,58]
C_ocs25=[38,49,54,57,58]
C_ocs26=[39,49,54,57,58]
C_ocs27=[40,50,55,57,58]
C_ocs28=[41,50,55,57,58]
C_ocs29=[42,51,55,57,58]
C_ocs30=[43,51,55,57,58]



#RU sets for 20 MHz
Total_bandwidth_20M=242*Wsc

RU20_1 =[242]
RU20_2=[106+16,106]

#RU sets for 40 MHz
Total_bandwidth_40M=484*Wsc
f_40M=2437000000 #should be updated later
RU40_1=[484]
RU40_2=[242,242]
RU40_3=[106+26+26,26+26+106,26+52+52+26]
RU40_4 =[106+26,106+26,106,106]

#RU sets for 80 MHz:there are 37 RUs in total
Total_bandwidth_80M=996*Wsc
f_80M=2437000000 #should be updated later
RU80_1=[996]
RU80_2=[484+26,484]
RU80_3=[106*3+26,106+26+106+52+26,4*26+106*2]
RU80_4=[242+26,242,242,242]
RU80_5=[106+26+52+26,106+26+52+26,106+26+52,26+52+106,26+26+106+26]
RU80_6=[106+26+52,106+52,106+52,106+52,106+52,52+52+26+26]
RU80_7=[106+26+26,26+52+52+26,52+26*3,52+26*3,52+26*3,52+52+26,106+26]
RU80_8=[106+26,106+26,106+26,106+26,106+26,106,106,106]

Total_bandwidth_160M=2*996*Wsc
def Func_Settings_160MRUs():
    global RU160_1,RU160_2,RU160_3,RU160_4,RU160_5,RU160_6,RU160_7,RU160_8
    global RU160_9,RU160_10,RU160_11,RU160_12,RU160_13,RU160_14,RU160_15,RU160_16
    RU160_1=[996*2]
    RU160_2=[996,996]
    RU160_3=[242*3+16,242*3+16,242*2]
    RU160_4=[484+26,484,484+26,484]
    RU160_5=[484,3*106+26+26,3*106+26+26,3*106+26+26,3*106+26+26,]
    RU160_6=[106*3+26,106+26+106+52+26,4*26+106*2,106*3+26,106+26+106+52+26,4*26+106*2]
    RU160_7=[484,16+242,242,242,242+16,242+16,242]
    RU160_8=[242+26,242,242,242,242+26,242,242,242]
    RU160_9=[242+26,242,242,242,242+26,242,242,106+26,106]
    RU160_10=[106+26+52+26,106+26+52+26,106+26+52,26+52+106,26+26+106+26,106+26+52+26,106+26+52+26,106+26+52,26+52+106,26+26+106+26]
    RU160_11=[106+26+52,106+52+26,106+52,26+52+106,26+106+26,106+52+26,106+52+26,106+26+52,26+52+106,26+26+106+26,26*6]
    RU160_12=[106+26+52,106+52,106+52,106+52,106+52,52+52+26+26,106+26+52,106+52,106+52,106+52,106+52,52+52+26+26]
    RU160_13=[5*26+26,5*26+26,5*26+26,5*26+26,5*26+26,5*26+26,5*26+26,5*26+26,5*26+26,5*26,5*26,5*26,5*26]
    RU160_14=[106+26+26,26+52+52+26,52+26*3,52+26*3,52+26*3,52+52+26,106+26,106+26+26,26+52+52+26,52+26*3,52+26*3,52+26*3,52+52+26,106+26]
    RU160_15=[5*26,5*26,5*26,5*26,5*26,5*26,5*26,5*26,5*26,5*26,5*26,5*26,5*26,5*26,4*26]
    RU160_16=[106+26,106+26,106+26,106+26,106+26,106,106,106,106+26,106+26,106+26,106+26,106+26,106,106,106]
Func_Settings_160MRUs()
# print(RU160_10)
# input()

Total_bandwidth_320M=4*996*Wsc

def Func_Settings_320MRUs():
    global RU320_1,RU320_2,RU320_3,RU320_4,RU320_5,RU320_6,RU320_7,RU320_8
    global RU320_9,RU320_10,RU320_11,RU320_12,RU320_13,RU320_14,RU320_15,RU320_16
    global RU320_17,RU320_18,RU320_19,RU320_20,RU320_21,RU320_22,RU320_23,RU320_24
    global RU320_25,RU320_26,RU320_27,RU320_28,RU320_29,RU320_30,RU320_31,RU320_32

    RU320_1=[4*996]
    RU320_2=[2*996,2*996]
    RU320_3=[50*26,49*26,49*26]
    RU320_4=[37*26,37*26,37*26,37*26]
    RU320_5=[30*26,30*26,30*26,29*26,29*26]
    RU320_6=[25*26,25*26,25*26,25*26,24*26,24*26]
    RU320_7=[22*26,21*26,21*26,21*26,21*26,21*26,21*26]
    RU320_8=[19*26,19*26,19*26,19*26,18*26,18*26,18*26,18*26]
    RU320_9=[17*26,17*26,17*26,17*26,16*26,16*26,16*26,16*26,16*26]
    RU320_10=[15*26,15*26,15*26,15*26,15*26,15*26,15*26,15*26,14*26,14*26]
    RU320_11=[14*26,14*26,14*26,14*26,14*26,13*26,13*26,13*26,13*26,13*26,13*26]
    RU320_12=[13*26,13*26,13*26,13*26,12*26,12*26,12*26,12*26,12*26,12*26,12*26,12*26]
    RU320_13=[12*26,12*26,12*26,12*26,12*26,11*26,11*26,11*26,11*26,11*26,11*26,11*26,11*26]
    RU320_14=[11*26,11*26,11*26,11*26,11*26,11*26,11*26,11*26,10*26,10*26,10*26,10*26,10*26,10*26]
    RU320_15=[10*26,10*26,10*26,10*26,10*26,10*26,10*26,10*26,10*26,10*26,10*26,10*26,10*26,9*26,9*26]
    RU320_16=[10*26,10*26,10*26,10*26,9*26,9*26,9*26,9*26,9*26,9*26,9*26,9*26,9*26,9*26,9*26,9*26]
    RU320_17=[9*26,9*26,9*26,9*26,9*26,9*26,9*26,9*26,9*26,9*26,9*26,12*26,8*26,8*26,8*26,8*26,8*26]
    RU320_18=[9*26,9*26,9*26,9*26,8*26,8*26,8*26,8*26,8*26,8*26,8*26,8*26,8*26,8*26,8*26,8*26,8*26,8*26]
    RU320_19=[8*26,8*26,8*26,8*26,8*26,8*26,8*26,8*26,8*26,8*26,8*26,8*26,8*26,8*26,8*26,7*26,7*26,7*26,7*26]
    RU320_20=[8*26,8*26,8*26,8*26,8*26,8*26,8*26,8*26,7*26,7*26,7*26,7*26,7*26,7*26,7*26,7*26,7*26,7*26,7*26,7*26]
    RU320_21=[8*26,7*26,7*26,7*26,7*26,7*26,7*26,7*26,7*26,7*26,7*26,7*26,7*26,7*26,7*26,7*26,7*26,7*26,7*26,7*26,7*26]
    RU320_22=[7*26,7*26,7*26,7*26,7*26,7*26,7*26,7*26,7*26,7*26,7*26,7*26,7*26,7*26,7*26,7*26,6*26,6*26,6*26,6*26,6*26,6*26]
    RU320_23=[7*26,7*26,7*26,7*26,7*26,7*26,7*26,7*26,7*26,7*26,6*26,6*26,6*26,6*26,6*26,6*26,6*26,6*26,6*26,6*26,6*26,6*26,6*26]
    RU320_24=[7*26,7*26,7*26,7*26,6*26,6*26,6*26,6*26,6*26,6*26,6*26,6*26,6*26,6*26,6*26,6*26,6*26,6*26,6*26,6*26,6*26,6*26,6*26,6*26]
    RU320_25=[6*26,6*26,6*26,6*26,6*26,6*26,6*26,6*26,6*26,6*26,6*26,6*26,6*26,6*26,6*26,6*26,6*26,6*26,6*26,6*26,6*26,6*26,6*26,5*26,5*26]
    RU320_26=[6*26,6*26,6*26,6*26,6*26,6*26,6*26,6*26,6*26,6*26,6*26,6*26,6*26,6*26,6*26,6*26,6*26,6*26,5*26,5*26,5*26,5*26,5*26,5*26,5*26,5*26]
    RU320_27=[6*26,6*26,6*26,6*26,6*26,6*26,6*26,6*26,6*26,6*26,6*26,6*26,6*26,5*26,5*26,5*26,5*26,5*26,5*26,5*26,5*26,5*26,5*26,5*26,5*26,5*26,5*26]
    RU320_28=[6*26,6*26,6*26,6*26,6*26,6*26,6*26,6*26,5*26,5*26,5*26,5*26,5*26,5*26,5*26,5*26,5*26,5*26,5*26,5*26,5*26,5*26,5*26,5*26,5*26,5*26,5*26,5*26]
    RU320_29=[6*26,6*26,6*26,5*26,5*26,5*26,5*26,5*26,5*26,5*26,5*26,5*26,5*26,5*26,5*26,5*26,5*26,5*26,5*26,5*26,5*26,5*26,5*26,5*26,5*26,5*26,5*26,5*26,5*26]
    RU320_30=[5*26,5*26,5*26,5*26,5*26,5*26,5*26,5*26,5*26,5*26,5*26,5*26,5*26,5*26,5*26,5*26,5*26,5*26,5*26,5*26,5*26,5*26,5*26,5*26,5*26,5*26,5*26,5*26,4*26,4*26]
    RU320_31=[5*26,5*26,5*26,5*26,5*26,5*26,5*26,5*26,5*26,5*26,5*26,5*26,5*26,5*26,5*26,5*26,5*26,5*26,5*26,5*26,5*26,5*26,5*26,5*26,4*26,4*26,4*26,4*26,4*26,4*26,4*26]
    RU320_32=[5*26,5*26,5*26,5*26,5*26,5*26,5*26,5*26,5*26,5*26,5*26,5*26,5*26,5*26,5*26,5*26,5*26,5*26,5*26,5*26,4*26,4*26,4*26,4*26,4*26,4*26,4*26,4*26,4*26,4*26,4*26,4*26]
Func_Settings_320MRUs()
# print(RU320_13)
# input()

# For RU assignment in AP coordination
a1=np.arange(1,4)
a2=np.arange(5,13)
a3=np.arange(20,25)
a4=np.arange(28,44)
C20M=np.hstack((a1,a2,a3,a4)) #20 MHz channels

a1=np.arange(4,5)
a2=np.arange(13,17)
a3=np.arange(25,27)
a4=np.arange(44,52)
C40M=np.hstack((a1,a2,a3,a4)) #40 MHz channels
# print(C40M)
# input()

a2=np.arange(17,19)
a3=np.arange(27,28)
a4=np.arange(52,56)
C80M=np.hstack((a2,a3,a4)) #80 MHz channels
# print(C80M)
# input()


a4=np.arange(19,20)
a2=np.arange(56,58)#160 MHz channels
C160M=np.hstack((a2,a4))

# print(C160M)
# input()

C320M=np.arange(58,59)
# print(C320M)
# input()

T_ul=3*10**(-3)
T_dl=T_ul*2
T_DIFS=34 * 10**(-6) # will be updated later
T_backoff=0 #will be updated later
T_SIFS=10*10**(-6)
T_OFDMA=32*8/9500000
T_TF=68*8/9500000
T_MBA=118*8/950000

# T_SIFS=10*10**(-6)
# T_DL=T_DIFS+T_backoff+T_dl+T_SIFS+T_OFDMA
# T_UL=T_DIFS+T_backoff+T_TF+2*T_SIFS+T_ul+T_MBA

# class Static_STA:
#     def __init__(self):
#         self.x=0
#         self.y=0

class Interference_Source:
    def __init__(self):
        self.x=0
        self.y=0
        self.z=0
        self.p=0  #power level
        self.c=0  #channel

IS = [] #Interference sources
for i in range(Num_of_interferences):
    IS.append(Interference_Source())
    # IS[i].x = X_interference[0][i]
    # IS[i].y = Y_interference[0][i]
    IS[i].x = X_interference[i]
    IS[i].y = Y_interference[i]
    IS[i].p=P_interference
    # IS[i].c=random.randint(1,len(C)+1)

    IS[i].c = C_for_IS[i]

class AccessPoint:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.z = height
        self.ID_Of_STAs=[] #The ID of STAs that associate with this AP

        self.ID_Of_STAs_24G=[]
        self.ID_Of_STAs_5G1 = []
        self.ID_Of_STAs_5G2 = []
        self.ID_Of_STAs_6G = []

        self.Num_Of_STAs=0 #The number of STAs that have been associated with this AP

        self.Dis_Between_AP_and_STAs=[] #The distance between AP and STAs that associated with this AP
        self.Dis_Between_AP_and_STAs_24G = []
        self.Dis_Between_AP_and_STAs_5G1 = []
        self.Dis_Between_AP_and_STAs_5G2 = []
        self.Dis_Between_AP_and_STAs_6G = []


        self.P=0 #power level
        self.P_24G=27
        self.P_5G1=28
        self.P_5G2=28
        self.P_6G=30
        # self.P_24G=0
        # self.P_5G1=0
        # self.P_5G2=0
        # self.P_6G=0

        self.C=0 #channel no.
        self.C_24G=0
        self.C_5G1=0
        self.C_5G2=0
        self.C_6G=0

        self.NumRUs_24G=0
        self.NumRUs_5G1=0
        self.NumRUs_5G2=0
        self.NumRUs_6G=0

        self.RUperUser_24G=0
        self.RUperUser_5G1=0
        self.RUperUser_5G2=0
        self.RUperUser_6G=0

        self.CandidateChannelsList_24G=[]
        self.CandidateChannelsList_5G1 = []
        self.CandidateChannelsList_5G2 = []
        self.CandidateChannelsList_6G = []

        self.NumOfInterference_24G=0
        self.NumOfInterference_5G1 = 0
        self.NumOfInterference_5G2 = 0
        self.NumOfInterference_6G = 0

        self.r_24G=0
        self.r_5G1 = 0
        self.r_5G2 = 0
        self.r_6G = 0



        self.Total_Interference=0
        self.Total_Interference_24G = 0
        self.Total_Interference_5G1 = 0
        self.Total_Interference_5G2 = 0
        self.Total_Interference_6G = 0


        self.SINR=[]
        self.SINR_24G = []
        self.SINR_5G1 = []
        self.SINR_5G2 = []
        self.SINR_6G = []

        self.Rx_power=0
        self.Rx_power_24G = 0
        self.Rx_power_5G1 = 0
        self.Rx_power_5G2 = 0
        self.Rx_power_6G = 0

        self.Rx_power_dBm=0
        self.Rx_power_dBm_24G = 0
        self.Rx_power_dBm_5G1 = 0
        self.Rx_power_dBm_5G2 = 0
        self.Rx_power_dBm_6G = 0

        self.Rx_power_W=0
        self.Rx_power_W_24G = 0
        self.Rx_power_W_5G1 = 0
        self.Rx_power_W_5G2 = 0
        self.Rx_power_W_6G = 0

        self.Group=[] #Maximum transmission rounds
        self.Group_24G = []
        self.Group_5G1 = []
        self.Group_5G2 = []
        self.Group_6G = []

        self.NeighborAPList=[]
        self.NeighborAPList_24G = []
        self.NeighborAPList_5G1 = []
        self.NeighborAPList_5G2 = []
        self.NeighborAPList_6G = []

        self.NeiInt_24G=[]
        self.NeiInt_5G1 = []
        self.NeiInt_5G2 = []
        self.NeiInt_6G = []

        self.Round=0
        self.Round_24G = 0
        self.Round_5G1 = 0
        self.Round_5G2 = 0
        self.Round_6G = 0

        self.Total_time=0
        self.Total_time_24G = 0
        self.Total_time_5G1 = 0
        self.Total_time_5G2 = 0
        self.Total_time_6G = 0

        self.Dis_to_IS=[]
        # self.Round=[] #Maximum transmission rounds
        # self.DataRate_UL=0 #Uplink data rate

AP = []
APNode=[]
# for i in range(NumOfAPs):
#     AP.append(AccessPoint())
#     AP[i].x = Coordinate_x[0][i]
#     AP[i].y = Coordinate_y[0][i]
#     # print(AP[i].x,AP[i].y)
#     APNode.append(AccessPoint())

class Station:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.z = 0
        self.IDOfAP=[]

        self.NumOfRU=0
        self.NumOfRU_24G=0
        self.NumOfRU_5G1=0
        self.NumOfRU_5G2=0
        self.NumOfRU_6G=0


        self.DataRate_DL=0 #downlink data rate
        self.DataRate_DL_24G = 0
        self.DataRate_DL_5G1 = 0
        self.DataRate_DL_5G2 = 0
        self.DataRate_DL_6G = 0


        self.DataRate_UL=0 #uplink data rate
        self.DataRate_UL_24G = 0
        self.DataRate_UL_5G1 = 0
        self.DataRate_UL_5G2 = 0
        self.DataRate_UL_6G = 0

        self.Total_Interference=0
        self.Total_Interference_24G=0
        self.Total_Interference_5G1 = 0
        self.Total_Interference_5G2 = 0
        self.Total_Interference_6G = 0


        self.SINR=0 # the SINR of downlink
        self.SINR_24G = 0
        self.SINR_5G1 = 0
        self.SINR_5G2 = 0
        self.SINR_6G = 0

        self.Rx_power_W=0
        self.Rx_power_W_24G=0
        self.Rx_power_W_5G1=0
        self.Rx_power_W_5G2=0
        self.Rx_power_W_6G=0


        self.Rx_power_dBm=0
        self.Rx_power_dBm_24G = 0
        self.Rx_power_dBm_5G1 = 0
        self.Rx_power_dBm_5G2 = 0
        self.Rx_power_dBm_6G = 0

        self.Dis_to_AP=0

        self.Throughput=0
        self.Throughput_24G = 0
        self.Throughput_5G1 = 0
        self.Throughput_5G2 = 0
        self.Throughput_6G = 0

        self.SINR_UP=0
        self.SINR_UP_24G=0
        self.SINR_UP_5G1=0
        self.SINR_UP_5G2=0
        self.SINR_UP_6G=0


        self.Dis_to_IS=[]
        #For mobile case
        self.pauseTime=0
        self.speed=0

        ###For Greedy
        self.NumOfAPs=0
        self.IDOfAPs=[]
        self.DistancesFromAPs=[]

#test
# x=PL(61,fc_5GI)
# print(x)
# input

STA = []
for i in range(NumOfSTAs):
    STA.append(Station())

###step
def Gen_Loc_of_STAs_40by30():
    CandidateCites_STA = []
    # for i in range(NumOfSTAs):
    #     STA[i].x = random.uniform(0, RegionLength)
    #     STA[i].y = random.uniform(0, RegionWidth)
        # print(STA[i].x, STA[i].y)
    a_temp = []
    TempLength = np.arange(0, RegionLength+1, RegionLength/39)
    TempWidth = np.arange(0, RegionWidth+1, RegionWidth/29)

    # print('len',len(TempLength))
    # print('width',len(TempWidth))
    # print(TempLength)
    # print(TempWidth)
    # input()

    for j in range(len(TempWidth)):
        for i in range(len(TempLength)):
            a_temp.append(TempLength[i])
            a_temp.append(TempWidth[j])
            CandidateCites_STA.append(a_temp)
            # print(a_temp)
            # print(CandidateCites)
            a_temp = []
    # print(len(CandidateCites_STA))
    # print(CandidateCites_STA)
    # input()

    for i in range(NumOfSTAs):
        STA[i].x=CandidateCites_STA[i][0]
        STA[i].y=CandidateCites_STA[i][1]

    for i in range(Num_Of_Static_STAs):
        # print(X_StaticSTAs)

        STA.append(Station())
        STA[NumOfSTAs + i].x = X_StaticSTAs[0][i]
        STA[NumOfSTAs + i].y = Y_StaticSTAs[0][i]

def Gen_Loc_of_STAs_32by25():
    CandidateCites_STA = []
    # for i in range(NumOfSTAs):
    #     STA[i].x = random.uniform(0, RegionLength)
    #     STA[i].y = random.uniform(0, RegionWidth)
        # print(STA[i].x, STA[i].y)
    a_temp = []
    TempLength = np.arange(0, RegionLength+1, RegionLength/31)
    TempWidth = np.arange(0, RegionWidth+1, RegionWidth/24)

    # print('len',len(TempLength))
    # print('width',len(TempWidth))
    # print(TempLength)
    # print(TempWidth)
    # input()

    for j in range(len(TempWidth)):
        for i in range(len(TempLength)):
            a_temp.append(float(TempLength[i]))
            a_temp.append(float(TempWidth[j]))
            CandidateCites_STA.append(a_temp)
            # print(a_temp)
            # print(CandidateCites)
            a_temp = []
    # print(len(CandidateCites_STA))
    # print(CandidateCites_STA)
    # input()

    for i in range(NumOfSTAs):
        STA[i].x=CandidateCites_STA[i][0]
        
        # print(CandidateCites_STA[20][1])
        
        # input()
        
        STA[i].y=CandidateCites_STA[i][1]
        STA[i].z=h_sta  #1.5 m 网格点高度为1.5m
        
        
        # print(CandidateCites_STA[i][0],CandidateCites_STA[i][1],STA[i].z)
        
        # input()

    # print(len(STA))
    # input()

    for i in range(Num_Of_Static_STAs):
        STA.append(Station())
        # STA[NumOfSTAs + i].x = X_StaticSTAs[0][i]
        # STA[NumOfSTAs + i].y = Y_StaticSTAs[0][i]
        STA[NumOfSTAs + i].x = X_StaticSTAs[i]
        STA[NumOfSTAs + i].y = Y_StaticSTAs[i]
        STA[NumOfSTAs + i].z=Z_StaticSTAs[i]
        
    
    # print(CandidateCites_STA)
    
    # input()
    return CandidateCites_STA

def Gen_Loc_of_STAs_64by50():
    CandidateCites_STA = []
    # for i in range(NumOfSTAs):
    #     STA[i].x = random.uniform(0, RegionLength)
    #     STA[i].y = random.uniform(0, RegionWidth)
        # print(STA[i].x, STA[i].y)
    a_temp = []
    TempLength = np.arange(0, RegionLength+1, RegionLength/63)
    TempWidth = np.arange(0, RegionWidth+1, RegionWidth/49)

    # print('len',len(TempLength))
    # print('width',len(TempWidth))
    # print(TempLength)
    # print(TempWidth)
    # input()

    for j in range(len(TempWidth)):
        for i in range(len(TempLength)):
            a_temp.append(float(TempLength[i]))
            a_temp.append(float(TempWidth[j]))
            CandidateCites_STA.append(a_temp)
            # print(a_temp)
            # print(CandidateCites)
            a_temp = []
    # print(len(CandidateCites_STA))
    # print(CandidateCites_STA)
    # input()

    for i in range(NumOfSTAs):
        STA[i].x=CandidateCites_STA[i][0]
        
        # print(CandidateCites_STA[20][1])
        
        # input()
        
        STA[i].y=CandidateCites_STA[i][1]
        STA[i].z=h_sta  #1.5 m 网格点高度为1.5m
        
        
        # print(CandidateCites_STA[i][0],CandidateCites_STA[i][1],STA[i].z)
        
        # input()

    # print(len(STA))
    # input()

    for i in range(Num_Of_Static_STAs):
        STA.append(Station())
        # STA[NumOfSTAs + i].x = X_StaticSTAs[0][i]
        # STA[NumOfSTAs + i].y = Y_StaticSTAs[0][i]
        STA[NumOfSTAs + i].x = X_StaticSTAs[i]
        STA[NumOfSTAs + i].y = Y_StaticSTAs[i]
        STA[NumOfSTAs + i].z=Z_StaticSTAs[i]
        
    
    # print(CandidateCites_STA)
    
    # input()
    return CandidateCites_STA

def Gen_Loc_of_STAs_128by100():
    CandidateCites_STA = []
    # for i in range(NumOfSTAs):
    #     STA[i].x = random.uniform(0, RegionLength)
    #     STA[i].y = random.uniform(0, RegionWidth)
        # print(STA[i].x, STA[i].y)
    a_temp = []
    TempLength = np.arange(0, RegionLength+1, RegionLength/127)
    TempWidth = np.arange(0, RegionWidth+1, RegionWidth/99)

    # print('len',len(TempLength))
    # print('width',len(TempWidth))
    # print(TempLength)
    # print(TempWidth)
    # input()

    for j in range(len(TempWidth)):
        for i in range(len(TempLength)):
            a_temp.append(float(TempLength[i]))
            a_temp.append(float(TempWidth[j]))
            CandidateCites_STA.append(a_temp)
            # print(a_temp)
            # print(CandidateCites)
            a_temp = []
    # print(len(CandidateCites_STA))
    # print(CandidateCites_STA)
    # input()

    for i in range(NumOfSTAs):
        STA[i].x=CandidateCites_STA[i][0]
        
        # print(CandidateCites_STA[20][1])
        
        # input()
        
        STA[i].y=CandidateCites_STA[i][1]
        STA[i].z=h_sta  #1.5 m 网格点高度为1.5m
        
        
        # print(CandidateCites_STA[i][0],CandidateCites_STA[i][1],STA[i].z)
        
        # input()

    # print(len(STA))
    # input()

    for i in range(Num_Of_Static_STAs):
        STA.append(Station())
        # STA[NumOfSTAs + i].x = X_StaticSTAs[0][i]
        # STA[NumOfSTAs + i].y = Y_StaticSTAs[0][i]
        STA[NumOfSTAs + i].x = X_StaticSTAs[i]
        STA[NumOfSTAs + i].y = Y_StaticSTAs[i]
        STA[NumOfSTAs + i].z=Z_StaticSTAs[i]
        
    
    # print(CandidateCites_STA)
    
    # input()
    return CandidateCites_STA

def Gen_Loc_of_STAs_20by20():#for 50 m by 50 m in a small case
    CandidateCites_STA = []
    # for i in range(NumOfSTAs):
    #     STA[i].x = random.uniform(0, RegionLength)
    #     STA[i].y = random.uniform(0, RegionWidth)
        # print(STA[i].x, STA[i].y)
    a_temp = []
    TempLength = np.arange(0, RegionLength+1, RegionLength/19)
    TempWidth = np.arange(0, RegionWidth+1, RegionWidth/19)

    # print('len',len(TempLength))
    # print('width',len(TempWidth))
    # print(TempLength)
    # print(TempWidth)
    # input()

    for j in range(len(TempWidth)):
        for i in range(len(TempLength)):
            a_temp.append(TempLength[i])
            a_temp.append(TempWidth[j])
            CandidateCites_STA.append(a_temp)
            # print(a_temp)
            # print(CandidateCites)
            a_temp = []
    # print(len(CandidateCites_STA))
    # print(CandidateCites_STA)
    # input()

    for i in range(NumOfSTAs):
        STA[i].x=CandidateCites_STA[i][0]
        STA[i].y=CandidateCites_STA[i][1]
        STA[i].z=h_sta

    # print(len(STA))
    # input()

    for i in range(Num_Of_Static_STAs):
        STA.append(Station())
        # STA[NumOfSTAs + i].x = X_StaticSTAs[0][i]
        # STA[NumOfSTAs + i].y = Y_StaticSTAs[0][i]
        STA[NumOfSTAs + i].x = X_StaticSTAs[i]
        STA[NumOfSTAs + i].y = Y_StaticSTAs[i]
        
        STA[NumOfSTAs + i].z = Z_StaticSTAs[i]
        
def Gen_Loc_of_STAs(x_width,y_length):#for 50 m by 50 m in a small case
    CandidateCites_STA = []
    # for i in range(NumOfSTAs):
    #     STA[i].x = random.uniform(0, RegionLength)
    #     STA[i].y = random.uniform(0, RegionWidth)
        # print(STA[i].x, STA[i].y)
    a_temp = []
    TempLength = np.arange(0, RegionLength+1, RegionLength/(y_length-1))
    TempWidth = np.arange(0, RegionWidth+1, RegionWidth/(x_width-1))

    # print('len',len(TempLength))
    # print('width',len(TempWidth))
    # print(TempLength)
    # print(TempWidth)
    # input()

    for j in range(len(TempWidth)):
        for i in range(len(TempLength)):
            a_temp.append(TempLength[i])
            a_temp.append(TempWidth[j])
            CandidateCites_STA.append(a_temp)
            # print(a_temp)
            # print(CandidateCites)
            a_temp = []
    # print(len(CandidateCites_STA))
    # print(CandidateCites_STA)
    # input()

    for i in range(NumOfSTAs):
        STA[i].x=CandidateCites_STA[i][0]
        STA[i].y=CandidateCites_STA[i][1]
        STA[i].z=h_sta

    # print(len(STA))
    # input()

    for i in range(Num_Of_Static_STAs):
        STA.append(Station())
        # STA[NumOfSTAs + i].x = X_StaticSTAs[0][i]
        # STA[NumOfSTAs + i].y = Y_StaticSTAs[0][i]
        STA[NumOfSTAs + i].x = X_StaticSTAs[i]
        STA[NumOfSTAs + i].y = Y_StaticSTAs[i]
        
        STA[NumOfSTAs + i].z = Z_StaticSTAs[i]
        
    return CandidateCites_STA
        

def find_closest_factors(a, b, n):
    """
    找到n的因数对(x,y)使得x/y最接近a/b
    参数:
        a (int): 目标比例的分子
        b (int): 目标比例的分母
        n (int): 需要分解的整数
    返回:
        tuple: 最佳匹配的因数对(x,y)
    """
    if n <= 0:
        raise ValueError("n必须是正整数")
    if b == 0:
        raise ValueError("比例分母b不能为0")
    
    target_ratio = a / b
    best_diff = float('inf')
    best_pair = (1, n)
    
    max_factor = int(math.isqrt(n)) + 1
    
    for x in range(1, max_factor):
        if n % x == 0:
            y = n // x
            current_ratio = x / y
            diff = abs(current_ratio - target_ratio)
            
            if diff < best_diff or (diff == best_diff and x > best_pair[0]):
                best_diff = diff
                best_pair = (x, y)
                
    return best_pair










# if __name__ == "__main__":
#     print("整数比例分解工具")
#     print("输入格式: a b n (用空格分隔)")
#     try:
#         a, b, n = map(int, input().split())
#         x, y = find_closest_factors(a, b, n)
#         print(f"\n分解结果: {x} × {y} = {n}")
#         print(f"实际比例: {x/y:.6f}")
#         print(f"目标比例: {a/b:.6f}")
#         print(f"比例差值: {abs((x/y)-(a/b)):.6f}")
#     except ValueError as e:
#         print(f"输入错误: {e}")
        
    

#For ES method
def Gen_Loc_of_STAs_random():

    # if Exhaustive_search==1:
    #     NumOfSTAs=200


    for i in range(NumOfSTAs):
        STA[i].x = random.uniform(0, RegionLength)
        STA[i].y = random.uniform(0, RegionWidth)
    for i in range(Num_Of_Static_STAs):
        # print(X_StaticSTAs)

        STA.append(Station())
        STA[NumOfSTAs + i].x = X_StaticSTAs[i]
        STA[NumOfSTAs + i].y = Y_StaticSTAs[i]

def Gen_Loc_of_STAs_Init_for_RWM():
    for i in range(NumOfSTAs):
        STA[i].x = random.uniform(0, RegionLength)
        STA[i].y = random.uniform(0, RegionWidth)

if NumOfSTAs== 1200 and Num_Of_Static_STAs==300: #for planning phase
    # NumOfSTAs = 1200
    # Num_Of_Static_STAs = 300
    Gen_Loc_of_STAs_40by30() #Obtain the locations of STAs

if NumOfSTAs== 800 and Num_Of_Static_STAs==200:
    CandidateCites_STA=Gen_Loc_of_STAs_32by25()
    # Gen_Loc_of_STAs_random()
    
if NumOfSTAs== 3200 and Num_Of_Static_STAs==800:
    CandidateCites_STA=Gen_Loc_of_STAs_64by50()
    # Gen_Loc_of_STAs_random()    
    # print(CandidateCites_STA)
    # # print()
if NumOfSTAs== 3200*4 and Num_Of_Static_STAs==800*4:
    CandidateCites_STA=Gen_Loc_of_STAs_128by100()
    
    # input()
    
#Gen_Loc_of_STAs
x_width, y_length = find_closest_factors(RegionWidth, RegionLength, NumOfSTAs)
CandidateCites_STA=Gen_Loc_of_STAs(x_width, y_length)

# print(len(CandidateCites_STA))

# # print(x_width,y_length)
# input()

# print(len(STA))
# input()
    
    

if Exhaustive_search==1 or ES==1:
    # Gen_Loc_of_STAs_random()
    Gen_Loc_of_STAs_20by20()
    
    

Locs_STAs=[]
STA_XX=[]
STA_YY=[]
STA_ZZ=[]
SSTA_XX=[]
SSTA_YY=[]
SSTA_ZZ=[]
for i in range(len(STA)):
    temp_sta_locs=[]
    temp_sta_locs.append(STA[i].x)
    # STA_XX.append(STA[i].x)
    temp_sta_locs.append(STA[i].y)
    temp_sta_locs.append(STA[i].z)
    # STA_YY.append(STA[i].y)
    Locs_STAs.append(temp_sta_locs)

for i in range(NumOfSTAs):
    STA_XX.append(STA[i].x)
    STA_YY.append(STA[i].y)
    STA_ZZ.append(STA[i].z)
    
for i in range(Num_Of_Static_STAs):
    
    # print("Num_Of_Static_STAs",Num_Of_Static_STAs)
    
    # print("Length STA:",len(STA))
    
    
    SSTA_XX.append(STA[NumOfSTAs+i].x)
    SSTA_YY.append(STA[NumOfSTAs + i].y)
    SSTA_ZZ.append(STA[NumOfSTAs + i].z)

if plot_layout_of_RPs==1:
    # plt.plot(STA_XX, STA_YY, 'gp')
    # plt.plot(X_StaticSTAs,Y_StaticSTAs,'ko')
    
    # 创建一个3D图形对象
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 绘制3D点
    ax.scatter(STA_XX, STA_YY,STA_ZZ,s=2,c='g', marker='o')  # c='r' 表示红色，marker='o' 表示圆形标记
    ax.scatter(SSTA_XX, SSTA_YY,SSTA_ZZ,s=2,c='k', marker='o')  # c='r' 表示红色，marker='o' 表示圆形标记

    print(STA_ZZ,SSTA_ZZ)

    # 设置坐标轴标签
    ax.set_xlabel('Length')
    ax.set_ylabel('Width')
    ax.set_zlabel('Height') #左侧的height字看不到，看如何处理
    
    ax.set_box_aspect([1.7, 1.5, 0.8])#设置长宽高比例

    # 显示图形
    plt.show()
    
    
    # plt.scatter(STA_XX, STA_YY,s=2,c='g')
    # plt.scatter(SSTA_XX, SSTA_YY,s=2,c='k')

    # if Exhaustive_search==1 or ES==1:
    #     plt.xlim(-1, 51)
    #     plt.ylim(-1, 51)
    # else:
    #     plt.xlim(-1, 81)
    #     plt.ylim(-1, 61)


    # plt.xticks(fontsize=13)
    # plt.yticks(fontsize=13)
    # plt.xlabel('The length of the region (m)', fontsize=13)
    # plt.ylabel('The width of the region (m)', fontsize=13)
    # plt.grid()
    # plt.savefig('Layout.eps', bbox_inches='tight', dpi=500, pad_inches=0.0)
    # plt.show()
    input()


# print(Locs_STAs)
# print(len(Locs_STAs))


# Locs_sta_arr=np.array(Locs_STAs)
# print(Locs_sta_arr)
# np.savetxt('STAs_Locs.csv',Locs_sta_arr)
# STALocs=np.loadtxt('STAs_Locs.csv')

# plt.plot(STALocs)
# plt.show()
#
# input()

def Reset_STAsetting():
    for i in range(len(STA)):
        STA[i].IDOfAP = []
        STA[i].NumOfRU = 0
        STA[i].NumOfRU_24G=0
        STA[i].NumOfRU_5G1=0
        STA[i].NumOfRU_5G2=0
        STA[i].NumOfRU_6G=0

        STA[i].DataRate_DL = 0  # downlink data rate
        STA[i].DataRate_DL_24G = 0
        STA[i].DataRate_DL_5G1 = 0
        STA[i].DataRate_DL_5G2 = 0
        STA[i].DataRate_DL_6G = 0


        STA[i].DataRate_UL = 0  # uplink data rate
        STA[i].DataRate_UL_24G = 0
        STA[i].DataRate_UL_5G1 = 0
        STA[i].DataRate_UL_5G2 = 0
        STA[i].DataRate_UL_6G = 0

        STA[i].Total_Interference = 0
        STA[i].Total_Interference_24G=0
        STA[i].Total_Interference_5G1 = 0
        STA[i].Total_Interference_5G2 = 0
        STA[i].Total_Interference_6G = 0

        STA[i].SINR = 0  # the SINR of downlink
        STA[i].SINR_24G = 0
        STA[i].SINR_5G1 = 0
        STA[i].SINR_5G2 = 0
        STA[i].SINR_6G = 0

        STA[i].Rx_power_W = 0
        STA[i].Rx_power_W_24G=0
        STA[i].Rx_power_W_5G1=0
        STA[i].Rx_power_W_5G2=0
        STA[i].Rx_power_W_6G=0


        STA[i].Rx_power_dBm = 0
        STA[i].Rx_power_dBm_24G = 0
        STA[i].Rx_power_dBm_5G1 = 0
        STA[i].Rx_power_dBm_5G2 = 0
        STA[i].Rx_power_dBm_6G = 0

        STA[i].Dis_to_AP = 0

        STA[i].Throughput = 0
        STA[i].Throughput_24G = 0
        STA[i].Throughput_5G1 = 0
        STA[i].Throughput_5G2 = 0
        STA[i].Throughput_6G = 0

        STA[i].SINR_UP = 0
        STA[i].SINR_UP_24G=0
        STA[i].SINR_UP_5G1=0
        STA[i].SINR_UP_5G2=0
        STA[i].SINR_UP_6G=0


        STA[i].Dis_to_IS = []
        # For mobile case
        STA[i].pauseTime = 0
        STA[i].speed = 0

        ###For Greedy
        STA[i].NumOfAPs = 0
        STA[i].IDOfAPs = []
        STA[i].DistancesFromAPs = []

#For static STAs, their numbers are numbered from len(NumOfSTAs) + i


# pip install pyqtgraph
# import pyqtgraph as pg
# app=pg.mkQApp()
# x=np.linspace(0,6*np.pi,100)
# y=np.sin(x)
# p=pg.plot(x,y,title=u'dd',left='amplit/v',bottom='t/s')
# p.setLabels(title='yyy')
# app.exec_()

#Save moving pattern
RWM_Pattern = np.zeros([NumOfSTAs, MovingSteps*2, 6])
def Func_RWM():
    Gen_Loc_of_STAs_Init_for_RWM()  # Initial locations of stations
    for i in range(NumOfSTAs):
        source_x=STA[i].x
        source_y=STA[i].y
        TotalTime=0
        TotalTime_temp=0
        j=-1
        temp_step=0
        while(1):
            temp_step=temp_step+1
            des_x = random.uniform(0, RegionLength)
            des_y = random.uniform(0, RegionWidth)
            pauseTime_i=random.uniform(60, 600)#Pause for 60s to 600s
            speed_i=random.uniform(0.5,1.5)    #Speed interval, in m/s
            MovingDistance=((source_x-des_x)**2+(source_y-des_y)**2)**(1/2)
            MovingTime=MovingDistance/speed_i
            TotalTimeForOneStep = pauseTime_i + MovingTime
            TotalTime = TotalTime + TotalTimeForOneStep

            #Moving info
            j=j+1
            TotalTime_temp=TotalTime_temp+MovingTime
            # RWM_Pattern[i,j,0]=RWM_Pattern[i,j,0]+MovingTime
            RWM_Pattern[i, j, 0] = TotalTime_temp
            RWM_Pattern[i, j, 1]=source_x
            RWM_Pattern[i, j, 2]=source_y
            RWM_Pattern[i, j, 3]=des_x
            RWM_Pattern[i, j, 4]=des_y
            RWM_Pattern[i, j, 5]=speed_i
            # print('total=',RWM_Pattern[i,j,0])
            # print('MovingTime=',MovingTime)

            #Pause info
            j=j+1
            TotalTime_temp = TotalTime_temp +pauseTime_i
            # RWM_Pattern[i,j,0]=RWM_Pattern[i,j-1,0]+pauseTime_i
            RWM_Pattern[i, j, 0] =TotalTime_temp
            RWM_Pattern[i, j, 1]=source_x
            RWM_Pattern[i, j, 2]=source_y
            RWM_Pattern[i, j, 3]=des_x
            RWM_Pattern[i, j, 4]=des_y
            RWM_Pattern[i, j, 5]=0

            # print('pauseTime_i=',pauseTime_i)
            # print('total=',RWM_Pattern[i,j,0])
            # print('j=',j)
            # input()

            source_x=des_x
            source_y=des_y
            if temp_step==MovingSteps:
                break
        # print('total=',RWM_Pattern[i,j,0])

def Func_LocationOnTheLine(T, x1, y1, x2, y2, speed):
    c_move = T * speed
    distance = ((x1 - x2)**2 + (y1 - y2)**2)**(0.5)
    a = abs(x1 - x2)
    b = abs(y1 - y2)
    x_leg = (c_move * a) / distance
    y_leg = (c_move * b) / distance

     # case    1
    if x1 < x2 and y1 < y2:
        x = x1 + x_leg
        y = y1 + y_leg


    # case    2
    if x1 > x2 and y1 < y2:
        x = x1 - x_leg
        y = y1 + y_leg

    # % case    3
    if x1 > x2 and y1 > y2:
        x = x1 - x_leg
        y = y1 - y_leg


    # % case    4
    if x1 < x2 and y1 > y2:
        x = x1 + x_leg
        y = y1 - y_leg

    # % case    5
    if x1 == x2 and y1 > y2:
        x = x1
        y = y1 - y_leg

    # % case    6
    if x1 == x2 and y1 < y2:
        x = x1
        y = y1 + y_leg

    # % case    7
    if x1 > x2 and y1 == y2:
        x = x1 + x_leg
        y = y1

    # % case    8
    if x1 < x2 and y1 == y2:
        x = x1 - x_leg
        y = y1

    return x,y

STA_realLoc=np.zeros([NumOfSTAs, TotalSamples, 2])
if GetMovingPattern==1:
    Func_RWM()
    a = np.arange(samplingPeriod, StatisticalPeriod+1, samplingPeriod) #Plus 1 means it can reach the value of StatisticalPeriod
    # print(a)
    for k in range(NumOfSTAs):
        for i in range(TotalSamples):
            for j in range(MovingSteps*2):
                # print(RWM_Pattern[k,j,0])
                # print(a[i])
                # print(j)
                # input()
                if j==0 and RWM_Pattern[k,j,0]>a[i]:

                    T = a[i]
                    [x, y] = Func_LocationOnTheLine(T, RWM_Pattern[k,j, 1], RWM_Pattern[k,j, 2], RWM_Pattern[k,j, 3],
                                                    RWM_Pattern[k,j, 4], RWM_Pattern[k,j, 5])
                    STA_realLoc[k,i, 0] = x
                    STA_realLoc[k,i, 1] = y
                    break
                if RWM_Pattern[k,j, 0] == a[i]:
                    STA_realLoc[k,i, 0] = RWM_Pattern[k,j, 3]
                    STA_realLoc[k,i, 1] = RWM_Pattern[k,j, 4]
                    break

                if RWM_Pattern[k,j, 0] > a[i]:
                    if np.mod(j, 2) == 1:
                        STA_realLoc[k,i, 0] = RWM_Pattern[k,j, 3]
                        STA_realLoc[k,i, 1] = RWM_Pattern[k,j, 4]
                    else:
                        T = a[i] - RWM_Pattern[k,j - 1, 0]
                        [x, y] = Func_LocationOnTheLine(T, RWM_Pattern[k,j, 1], RWM_Pattern[k,j, 2],
                                                        RWM_Pattern[k,j, 3], RWM_Pattern[k,j, 4],
                                                        RWM_Pattern[k,j, 5])
                        STA_realLoc[k,i, 0] = x
                        STA_realLoc[k,i, 1] = y

                        # print(x,y)
                    break

            # print('a[i]=',a[i])
            #
            # print('i=',i)
            # print('RWM[0]_totaltime=',RWM_Pattern[k,j,0])
            # # print(a[i])
            # print('STA_realLoc[k,i, 0]=',STA_realLoc[k,i, 0])
            # print('STA_realLoc[k,i, 1]=',STA_realLoc[k,i, 1])
            # # print(j)
            # print('test')
            # input()
    # ssss=STA_realLoc.shape
    # print('test')
    # print(ssss)
    # print(STA_realLoc[3,4, 0],STA_realLoc[3,4, 1])
    # print(STA_realLoc[30, 40, 0], STA_realLoc[30, 40, 1])
    with open('STA_realLoc.csv','w') as outfile:
        for i in STA_realLoc:
            np.savetxt(outfile,i,fmt='%f',delimiter=',')

    STA_realLoc=np.loadtxt('STA_realLoc.csv',delimiter=',').reshape((NumOfSTAs,TotalSamples,2))
    # print(STA_realLoc[3,4, 0],STA_realLoc[3,4, 1])
    # print(STA_realLoc[30, 40, 0], STA_realLoc[30, 40, 1])
    
    print("testtttt")
    print('The STAs\' locations have been saved!')
    os._exit(1)

    # input()


# print(RWM_Pattern.shape)
# print('rwm')
# print(RWM_Pattern[0])
# print(STA_realLoc.shape)
# print('reall')
# print(STA_realLoc[0])
# input()


def Plot_fig():
    Xsta=[]
    Ysta=[]
    # print('ttttt')
    # plt.figure()
    figure, axes = plt.subplots()
    # for i in range(NumOfSTAs+Num_Of_Static_STAs):
    #     Xsta.append(STA[i].x)
    #     Ysta.append(STA[i].y)
    #     if i<NumOfSTAs:
    #         plt.plot(STA[i].x,STA[i].y,'go',markersize=2.)
    #         # plt.text(STA[i].x+1,STA[i].y,str(i))
    #     else:
    #         plt.plot(STA[i].x,STA[i].y,'ko',markersize=2.) #black circles denote static STAs
    #         # plt.text(STA[i].x+1,STA[i].y,str(i))


        # plt.pause(1)

    # plt.figure()
    # plt.plot(Xsta,Ysta,'bo')
    # plt.show()
    # input()
    Xap=[]
    Yap=[]
    for i in range(len(AP)):
        Xap.append(AP[i].x)
        Yap.append(AP[i].y)
        plt.plot(AP[i].x,AP[i].y,'r^',markersize=8.)
        # plt.text(AP[i].x+1,AP[i].y,str(i))
        # plt.pause(1)
    # plt.plot(Xap,Yap,'r^')

    # draw_circle = plt.Circle((0.5, 0.5), 0.3,fill=False)
    #
    # axes.set_aspect(1)
    # axes.add_artist(draw_circle)
    # plt.title('Circle')
    # plt.show()
    # cc = 0
    # ccc = 0
    XY_list=[]
    for i in range(len(AP)):
        tempp=[]
        tempp.append(AP[i].x)
        tempp.append(AP[i].y)
        XY_list.append(tempp)
    # print(XY_list)
    # input()
    XY_list_temp=deepcopy(XY_list)

    for List_x in XY_list[:]:
        if XY_list.count(List_x)>1:
            XY_list.remove(List_x)

    for item in range(len(XY_list)):
        numAPs=XY_list_temp.count(XY_list[item])
        XY_list[item].append(numAPs)

    for i in range(len(XY_list)):
        plt.text(XY_list[i][0] + 1,XY_list[i][1], str(XY_list[i][2]),fontsize=13)

    # print("skdk")
    # input()

    # for i in range(len(AP)):
    #     for B in Band:
    #         if B=='24G':
    #             r = Range(AP[i].P_24G, P_target_dBm, '24G')
    #             # print(f"Range test: {Range(AP[0].P_24G, P_target_dBm, '24G')}")
    #             # print(AP[i].P_24G,P_target_dBm)
    #             # r=D
    #             # print(r)
    #             # input()
    #             if r is None or np.isnan(r) or r < 0 or r > 1e6:
    #                 print(f"⚠️ Skip invalid circle: i={i}, band={B}, r={r}")
    #                 continue
    #             print(f"Drawing circle {i}, {B}, radius={r}")
    #             draw_circle = plt.Circle((AP[i].x, AP[i].y), r, fill=False)
    #             axes.add_artist(draw_circle)
    #         if B=='5G1':
    #             r = Range(AP[i].P_5G1, P_target_dBm, '5G1')
    #             if r is None or np.isnan(r) or r < 0 or r > 1e6:
    #                 print(f"⚠️ Skip invalid circle: i={i}, band={B}, r={r}")
    #                 continue
    #             print(f"Drawing circle {i}, {B}, radius={r}")
    #             draw_circle = plt.Circle((AP[i].x, AP[i].y), r, fill=False)
    #             axes.add_artist(draw_circle)
    #             # r = D
    #             # print('r test')
    #             # print(r)
    #             # input()
    #         if B=='5G2':
    #             r = Range(AP[i].P_5G2, P_target_dBm, '5G2')
    #             if r is None or np.isnan(r) or r < 0 or r > 1e6:
    #                 print(f"⚠️ Skip invalid circle: i={i}, band={B}, r={r}")
    #                 continue
    #             print(f"Drawing circle {i}, {B}, radius={r}")
    #             draw_circle = plt.Circle((AP[i].x, AP[i].y), r, fill=False)
    #             axes.add_artist(draw_circle)
    #             # print(r)
    #             # input()
    #         if B=='6G':
    #             r = Range(AP[i].P_6G, P_target_dBm, '6G')
    #             # r = D
    #             if r is None or np.isnan(r) or r < 0 or r > 1e6:
    #                 print(f"⚠️ Skip invalid circle: i={i}, band={B}, r={r}")
    #                 continue
    #             print(f"Drawing circle {i}, {B}, radius={r}")
    #             draw_circle = plt.Circle((AP[i].x, AP[i].y), r, fill=False)
    #             axes.add_artist(draw_circle)
                # input()
            # x=AP[i].x

            # y=AP[i].y

            # print('r=',r)

            # draw_circle = plt.Circle((x, y), r,fill=False)

            # axes.set_aspect(1)
            # axes.add_artist(draw_circle)
            # plt.title('Circle')
            # plt.show()
            
            # input()

            # a=np.arange(x-r,x+r,0.01)
            # tempp=np.power(r,2)-np.power((a-x),2)
            # print(tempp)
            # input()
            # test=np.power(r, 2) - np.power((a - x), 2)
            # for kk in test:
            #     if kk<0:
            #         print(kk)
            # print(test)
            # #
            # input()

            # b=np.sqrt(np.abs(np.power(r,2)-np.power((a-x),2)))+y

            # c=np.sqrt(np.power(r,2)-np.power((a-x),2))-y

            # plt.plot(a,b,color='r',linestyle='--')
            # cc=cc+1
            # print('plo{}',format(cc))

            # plt.plot(a, 2*y-b, color='r', linestyle='--')
            # ccc=ccc+1
            # print('plo2{}', format(ccc))
            # plt.plot(a,c, color='r', linestyle='-')
            # plt.scatter(0,0,c='b',marker='o')





    # plt.plot(X_interference,Y_interference,'gp')
    # # plt.plot(X_StaticSTAs,Y_StaticSTAs,'ko')
    
    
    
    if Exhaustive_search==1 or ES==1:
        plt.xlim(-1, 51)
        plt.ylim(-1, 51)
    else:
        plt.xlim(-1,RegionLength+1)
        plt.ylim(-1,RegionWidth+1)
    
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.xlabel('The length of the region (m)', fontsize=13)
    plt.ylabel('The width of the region (m)', fontsize=13)
    plt.grid()
    # print("kdkdk333")
    # plt.savefig('Plot_result.png', dpi=150, bbox_inches='tight')
    plt.pause(2)
    plt.show()
    plt.close()
    print("✅ Finished Plot_fig() without crash.")
    # plt.show()
    # # # plt.pause(0.1)
    # # # plt.savefig('test.pdf')
    # plt.pause(2)
    # print("kdkdk444")

    # if RandomMethod==1:
    #     print("Before savefig--ran")
    #     plt.savefig('AP_Random.png', bbox_inches='tight', dpi=500, pad_inches=0.0)
    #     print("After savefig")
    # if start_greedy==1:
    #     print("Before savefig-greedy")
    #     plt.savefig('AP_Greedy2.png', bbox_inches='tight', dpi=150, pad_inches=0.1)
    #     print("After savefig")
    # if FourStage==1 and FirstStageOnly==0:
    #     print("Before savefig-4s")
    #     plt.savefig('AP_4s.png', bbox_inches='tight', dpi=500, pad_inches=0.0)
    #     print("After savefig")
    # if LocalOptimizerMethod==1:
    #     print("Before savefig-loc")
    #     plt.savefig('AP_LocalOptimizer.png', bbox_inches='tight', dpi=500, pad_inches=0.0)
    #     print("After savefig")

    # # plt.show()
    # print("kdkdk555")
    # plt.close()
    # print(">>> Finished Plot_fig()")
    
    


Temp_d=[]
def Association_singleLink():

    Ratio_2dot4G=3/32
    Ratio_5GI=8/32
    Ratio_5II=5/32
    Ratio_6G=16/32

    distance=np.zeros([NumOfSTAs+Num_Of_Static_STAs,len(AP)])

    # print(len(AP))
    # # input()
    # print(AP[1].x,AP[1].y)
    # input()
    #for dynamic STAs
    for i in range(NumOfSTAs+Num_Of_Static_STAs):
        for j in range(len(AP)):
            distance[i,j]=math.sqrt(math.pow(STA[i].x-AP[j].x,2)+math.pow(STA[i].y-AP[j].y,2))
    Distance=np.array(distance)
    # print(Distance)

    # Plot_fig()


    # print(Distance)
    # print(len(Distance))
    # print('tessssssss')
    # input()
    Temp_d = Distance.min(axis=1) # Should be randomly choose one when there are more than one AP
    # print(Temp_d)
    # input()

    # print(Temp_d)
    # input()
    # Temp_dd=np.array(Temp_d)
    # Temp_dd.T
    # print(Temp_dd)
    # print('teee')
    # input()
    Temp_AP_index=[]
    # print(NumOfSTAs+Num_Of_Static_STAs)
    # print(len(STA))
    # print('test')
    # input()

    for i in range(len(STA)):
        temp_min_arr=np.where(Distance[i,:]==Temp_d[i])

        Temp_min_arr=temp_min_arr[0]
        # if len(Temp_min_arr)>1:
        #     print(Temp_min_arr)
        #     input()
        TTemp_min_arr=np.random.choice(Temp_min_arr,1)
        minID=TTemp_min_arr[0]
        # if len(Temp_min_arr) > 1:
        #     print(minID)
        #     input()
        Temp_AP_index.append(minID)


    # Temp_np_d=np.where(Distance==Temp_d)
    # print(Temp_np_d)
    # input()
    #
    # Temp_AP_index1=np.array(Temp_AP_index)
    # Temp_AP_index2 = Distance.argmin(axis=1)
    # input()
    #
    #
    # print(Temp_AP_index)
    # input()
    #print(Temp_AP_index)
    #input()
    for i in range(NumOfSTAs+Num_Of_Static_STAs):
        AP[Temp_AP_index[i]].ID_Of_STAs.append(i)
        AP[Temp_AP_index[i]].Dis_Between_AP_and_STAs.append(Temp_d[i])
        STA[i].Dis_to_AP=Temp_d[i]
        STA[i].IDOfAP=Temp_AP_index[i]
        # print(STA[i].IDOfAP)
        # input()
    # for i in range(NumOfAPs):
    #     AP[i].Num_Of_STAs = len(AP[i].ID_Of_STAs)

    # #for static STAs
    # distance_static = np.zeros([Num_Of_Static_STAs, NumOfAPs])
    # for i in range(Num_Of_Static_STAs):
    #     for j in range(NumOfAPs):
    #         distance_static[i,j]=math.sqrt(math.pow(S_STA[i].x-AP[j].x,2)+math.pow(S_STA[i].y-AP[j].y,2))
    # Temp_d_static = distance_static.min(axis=1)
    # Temp_AP_index_static = distance_static.argmin(axis=1)
    # for i in range(Num_Of_Static_STAs):
    #     AP[Temp_AP_index_static[i]].ID_Of_STAs.append('s'+str(i))#'s' denotes 'static'
    #     AP[Temp_AP_index_static[i]].Dis_Between_AP_and_STAs.append(Temp_d_static[i])

    for i in range(len(AP)):
        AP[i].Num_Of_STAs = len(AP[i].ID_Of_STAs)

    for i in range(len(AP)):
        # AP[i].Num_Of_STAs
        # print('Assign STAs to 2.4G band')
        ID_Of_STAs=np.array(AP[i].ID_Of_STAs)
        Dis_Between_AP_and_STAs=np.array(AP[i].Dis_Between_AP_and_STAs)

        num1=int(np.rint(AP[i].Num_Of_STAs*Ratio_2dot4G))
        if num1==0:
            print('Nums of stas for 2.4G is 0!')
            num1=1
            print('Nums of stas for 2.4G is assigned 1!')
        # print(num1,AP[i].Num_Of_STAs)
        Index=np.arange(0,len(ID_Of_STAs))
        # print(Index)
        Index_i = np.random.choice(Index, num1, replace=False)
        # print(Index_i)
        # AP[i].ID_Of_STAs=np.array(AP[i].ID_Of_STAs)
        # print(ID_Of_STAs)
        AP[i].ID_Of_STAs_24G =ID_Of_STAs[Index_i]
        AP[i].Dis_Between_AP_and_STAs_24G=Dis_Between_AP_and_STAs[Index_i]
        # print(AP[i].ID_Of_STAs_24G)
        ID_Of_STAs=np.delete(ID_Of_STAs,Index_i)
        Dis_Between_AP_and_STAs=np.delete(Dis_Between_AP_and_STAs,Index_i)
        # print(ID_Of_STAs)

        # input()

        # print('Assign STAs to 5GI band')
        num2 = int(np.rint(AP[i].Num_Of_STAs * Ratio_5GI))
        if num2==0:
            print('Nums of stas for 5G1 is 0!')
            num2=4
            print('Nums of stas for 5G1 is assign 4!')
        # print(num2, AP[i].Num_Of_STAs)
        Index = np.arange(0, len(ID_Of_STAs))
        # print(Index)
        Index_i = np.random.choice(Index, num2, replace=False)
        # print(Index_i)
        # AP[i].ID_Of_STAs=np.array(AP[i].ID_Of_STAs)
        # print(ID_Of_STAs)
        AP[i].ID_Of_STAs_5G1 = ID_Of_STAs[Index_i]
        AP[i].Dis_Between_AP_and_STAs_5G1 = Dis_Between_AP_and_STAs[Index_i]
        # print(AP[i].ID_Of_STAs_5G1)
        ID_Of_STAs = np.delete(ID_Of_STAs, Index_i)
        Dis_Between_AP_and_STAs = np.delete(Dis_Between_AP_and_STAs, Index_i)
        # print(ID_Of_STAs)
        # input()

        # print('Assign STAs to 5GII band')
        num3 = int(np.rint(AP[i].Num_Of_STAs * Ratio_5II))
        if num3==0:
            print('Nums of stas for 5G2 is 0!')
            num3=2
            print('Nums of stas for 5G2 is assign 2!')
        # print(num3, AP[i].Num_Of_STAs)
        Index = np.arange(0, len(ID_Of_STAs))
        # print(Index)
        Index_i = np.random.choice(Index, num3, replace=False)
        # print(Index_i)
        # AP[i].ID_Of_STAs=np.array(AP[i].ID_Of_STAs)
        # print(ID_Of_STAs)
        AP[i].ID_Of_STAs_5G2 = ID_Of_STAs[Index_i]
        AP[i].Dis_Between_AP_and_STAs_5G2 = Dis_Between_AP_and_STAs[Index_i]
        # print(AP[i].ID_Of_STAs_5G2)
        ID_Of_STAs = np.delete(ID_Of_STAs, Index_i)
        Dis_Between_AP_and_STAs = np.delete(Dis_Between_AP_and_STAs, Index_i)
        # print(ID_Of_STAs)
        # input()

        # print('Assign STAs to 6G band')
        AP[i].ID_Of_STAs_6G = ID_Of_STAs
        AP[i].Dis_Between_AP_and_STAs_6G=Dis_Between_AP_and_STAs
        # print(AP[i].ID_Of_STAs_6G)
        # print('---')
        # num = AP[i].Num_Of_STAs-num1-num2-num3
        # print(num, AP[i].Num_Of_STAs)
        # Index = np.arange(0, len(ID_Of_STAs))
        # print(Index)
        # Index_i = np.random.choice(Index, num, replace=False)
        # print(Index_i)
        # # AP[i].ID_Of_STAs=np.array(AP[i].ID_Of_STAs)
        # print(ID_Of_STAs)
        # AP[i].ID_Of_STAs_6G = ID_Of_STAs[Index_i]
        # print(AP[i].ID_Of_STAs_6G)
        # ID_Of_STAs = np.delete(ID_Of_STAs, Index_i)
        # print(ID_Of_STAs)
        # input()


        #
        # numOfSTAs=a=int(np.rint(3.2))
        # AP[i].ID_Of_STAs_24G=

    return


#multi-link
def Association():
    # global Change
    # if Change==1:
    global AP  #revisedII
    # print("Im here!")
    
    # if len(AP)>14:
    #     appp = []
    #     apppx=[]
    #     apppy=[]
    #     for tempii in range(len(AP)):
    #         appp.append(len(AP[tempii].ID_Of_STAs))
    #         apppx.append(AP[tempii].x)
    #         apppy.append(AP[tempii].y)
    #
    #     print('number of stations of ap', appp)
    #     print('number of stations of ap', apppx)
    #     print('number of stations of ap', apppy)
    #     input()
    
    # print("len",len(STA))
    # print("lenAP",len(AP))

    for j in range(len(STA)):
        
        # print("aa")
        DIS_for_Ass=[]
        for i in range(len(AP)):
            # if len(AP)>12:
            #     print('AP ID',i)
            #     input()
            # print(AP[i].x,AP[i].y)
            # if STA[j].z==h_sta: #如果是网格点，则距离是平面上的点与AP的距离
            #     Disx=((STA[j].x-AP[i].x)**2+(STA[j].y-AP[i].y)**2)**(0.5)
            #     # print("Disx:",Disx)
            #     #3D distance
            #     Disx=(Disx**2+H_sta**2)**(0.5)
            #     # print("Disx:", Disx)
            #     # input()
            #     DIS_for_Ass.append(Disx)
            # else: #如果是IoT设备，则在3D空间中任意位置
            Disx=((STA[j].x-AP[i].x)**2+(STA[j].y-AP[i].y)**2+(STA[j].z-AP[i].z)**2)**(0.5)
            DIS_for_Ass.append(Disx)
        alist = np.array(DIS_for_Ass)
        # print(alist)
        b = np.where(alist == alist.min())
        
        # print("Im here!")
        
        
        # a=np.argmin(alist)

        temp_ap=np.inf
        ap_i = []
        for i in range(len(b[0])):
            # print('len(b[0])',len(b[0]))

            if len(AP[b[0][i]].ID_Of_STAs)<temp_ap:
                # if len(b[0])>=2:
                #     appp=[]
                #     for tempii in range(len(AP)):
                #         appp.append(len(AP[tempii].ID_Of_STAs))
                #
                #     print('number of stations of ap',appp)
                #     print('len_b=',len(b[0]))
                #     print('i=',i)
                #     print('temp_ap',temp_ap)
                #     print('number of STAs',len(AP[b[0][i]].ID_Of_STAs))
                #
                #     input()

                temp_ap=len(AP[b[0][i]].ID_Of_STAs)
                ap_i=i
                # input()
        # if len(b[0]) >= 2:
        #     print('ap_i',ap_i)
        #     print('numAPP=', b[0][ap_i])
        #     input()

        AP[b[0][ap_i]].ID_Of_STAs.append(j)
        AP[b[0][ap_i]].Dis_Between_AP_and_STAs.append(alist[b[0][ap_i]])
        STA[j].IDOfAP =b[0][ap_i]

    #     print("AP list:",STA[j].IDOfAPs)
    #     print("ID of AP:",STA[j].IDOfAP)
    # input()



    #
    # input()
    if LocalOptimizerMethod==0: #除了LocalOptimizerMethod之外，我们的方法要调整关联数量
        test=0
        count = 0
        dectect=[]
        flage = 0
        while(1):
            test=test+1
    
            for i in range(len(AP)):
                AP[i].Num_Of_STAs = len(AP[i].ID_Of_STAs)
                # print("AP[i].Num_Of_STAs", i, AP[i].Num_Of_STAs)
            temp_stations=[]
            for i in range(len(AP)):
                AP[i].Num_Of_STAs = len(AP[i].ID_Of_STAs)
                temp_stations.append(AP[i].Num_Of_STAs)
    
            maximumNum=max(temp_stations)
            minimumNum=min(temp_stations)
            difff=maximumNum-minimumNum
            # print("The AP load difference========:",maximumNum-minimumNum)
    
            if len(dectect)>4 and dectect[-1]==dectect[-2] and dectect[-2]==dectect[-3] and dectect[-3]==dectect[-4]\
                    and dectect[-4]==dectect[-5] or len(dectect)>100:
                flage=1
    
            if maximumNum-minimumNum<4 or flage==1:
                # print("The AP load difference========:",maximumNum-minimumNum)
                flage =0
                dectect = []
                break
            else:
                # print("test:",test)
                # my_list = [10, 25, 30, 40]
                # max_value = max(my_list)  # 获取列表中的最大值
                #find all the APs with the maximum number of stations
                indices = [i for i, x in enumerate(temp_stations) if x == maximumNum]
                # print("All the APs with big load:",indices)  # 输出: [3, 5]
    
                if test==102:
                    break
    
                for max_index in indices:
    
                    # max_index1 = temp_stations.index(maximumNum)  # 获取最大值在列表中的索引 the index of the AP with maximum load
    
                # print("最大值为", maximumNum)
                #     print("最大值对应的 AP 索引为", max_index1)
                #     print("max_index",max_index)
                # input()
    
                # print(temp_stations,maximumNum,minimumNum)
                # input()
                # print("max load AP:",max_index)
                # print("original AP stations:", AP[max_index].ID_Of_STAs)
                # print("Num of original stations:", len(AP[max_index].ID_Of_STAs))
                # print("original dis",AP[max_index].Dis_Between_AP_and_STAs)
                # print("dis info:",AP[max_index].Dis_Between_AP_and_STAs)
                # input()
                    temp_dis=[]
                    temp_STA_list=AP[max_index].ID_Of_STAs[:] #the station list of the maximum-load AP
                    # print("temp_STA_list",temp_STA_list)
                    for i in range(maximumNum):
                        
                        # if STA[temp_STA_list[i]].z==h_sta:
                        #     dis=((STA[temp_STA_list[i]].x-AP[max_index].x)**2+(STA[temp_STA_list[i]].y-AP[max_index].y)**2)**(0.5)
                        #     # print(dis)
                        #     #3D distance
                        #     dis=(dis**2+H_sta**2)**(0.5)
                        # else:
                        dis=((STA[temp_STA_list[i]].x-AP[max_index].x)**2+\
                                 (STA[temp_STA_list[i]].y-AP[max_index].y)**2+(STA[temp_STA_list[i]].z-AP[max_index].z)**2)**(0.5)
                        # print(dis)
                        # input()
    
                        temp_dis.append(dis)
                    #     print("Total number of STAs:",maximumNum)
                    #     print("Its AP:",STA[temp_STA_list[i]].IDOfAP)
                    # input()
                    my_list = temp_dis #Distance list [descending order]
                    sorted_with_index = sort_descending_with_index(my_list)
                    # print(sorted_with_index)
                    # print(sorted_with_index[0][0])
                    # for index, value in sorted_with_index:
                    #     print(f"Index: {index}, Value: {value}")
                    #     break
                    for i in range(maximumNum):
                        temp_APs_list = STA[temp_STA_list[sorted_with_index[i][0]]].IDOfAPs[:]
                        # print(STA[AP[max_index].ID_Of_STAs[sorted_with_index[i][0]]].IDOfAPs)
                        # print(len(STA[AP[max_index].ID_Of_STAs[sorted_with_index[i][0]]].IDOfAPs))
                        # print(STA[AP[max_index].ID_Of_STAs[sorted_with_index[i][0]]].IDOfAP)
                        temp_dis_list=[]
                        temp_staaaaa=[]
                        if len(temp_APs_list)>1:
                            for k in range(len(temp_APs_list)):
                                AP[temp_APs_list[k]].Num_Of_STAs = len(AP[temp_APs_list[k]].ID_Of_STAs)
                                temp_staaaaa.append(AP[temp_APs_list[k]].Num_Of_STAs)
                            maxx=max(temp_staaaaa)
                            minn=min(temp_staaaaa)
                            dif=maxx-minn
                            if dif>1:
                                # print("test:===", test)
                                # if dif==2:
                                #     count=count+1
                                #
                                #     if count==2:
                                #         continue
                                # print("temp_staaaaa:", temp_staaaaa)
                                #
                                # print("dif:",dif)
                                # print("count:",count)
                                # print()
                                # print("original stations:",)
                                # print("The farthest station:",temp_STA_list[sorted_with_index[i][0]])
                                # input()
                                #calculate the distance for this station
                                # print("The original Aps:", temp_APs_list)
                                # print("The original AP:",STA[temp_STA_list[sorted_with_index[i][0]]].IDOfAP)
                                tempp_sta=[]
                                for j in range(len(temp_APs_list)):
                                    if  temp_APs_list[j]==max_index:
                                        dis=10000 #if the AP is the original one, then we exclude it, i.e., set a larger distance
                                    else:
                                        
                                        # if STA[temp_STA_list[sorted_with_index[i][0]]].z==h_sta:
                                        #     dis=((STA[temp_STA_list[sorted_with_index[i][0]]].x-AP[temp_APs_list[j]].x)**2+\
                                        #      (STA[temp_STA_list[sorted_with_index[i][0]]].y-AP[temp_APs_list[j]].y)**2)**(0.5)
        
                                        #     #3d distance
                                        #     dis=(dis**2+H_sta**2)**(0.5)
                                        # else:
                                        dis=((STA[temp_STA_list[sorted_with_index[i][0]]].x-AP[temp_APs_list[j]].x)**2+\
                                             (STA[temp_STA_list[sorted_with_index[i][0]]].y-AP[temp_APs_list[j]].y)**2\
                                                 +(STA[temp_STA_list[sorted_with_index[i][0]]].z-AP[temp_APs_list[j]].z)**2)**(0.5)
    
                                    temp_dis_list.append(dis)
                                    tempp_sta.append(AP[temp_APs_list[j]].Num_Of_STAs)
                                # print("temp_APs_list:",temp_APs_list)
                                # print("tempp-sta:",tempp_sta)
                                # input()
                                #Find the index of the minimum distance
                                # print(temp_APs_list)
                                # print(temp_dis_list)
                                min_dis = min(temp_dis_list)
                                min_index=temp_dis_list.index(min_dis)
                                # print("min dis:",min_dis)
                                # print("min index:",min_index)
                                # reassociation (the STA associates with another AP)
                                # print("The farthest station--test:", temp_STA_list[sorted_with_index[i][0]])
                                del AP[max_index].ID_Of_STAs[sorted_with_index[i][0]] #remove a station
                                # print("The final stations:",AP[max_index].ID_Of_STAs)
                                # print("The number of final stas:",len(AP[max_index].ID_Of_STAs))
                                del AP[max_index].Dis_Between_AP_and_STAs[sorted_with_index[i][0]]
                                # print("The final dis:",AP[max_index].Dis_Between_AP_and_STAs)
                                # print("The number of dis:",len(AP[max_index].Dis_Between_AP_and_STAs))
                                STA[temp_STA_list[sorted_with_index[i][0]]].IDOfAP = temp_APs_list[min_index]
                                # print("the sta need to re-associated:",temp_STA_list[sorted_with_index[i][0]])
                                # print("The final AP:",STA[temp_STA_list[sorted_with_index[i][0]]].IDOfAP)
                                #add the station to the new AP
                                # print("The ID of the sta:",temp_STA_list[sorted_with_index[i][0]])
                                # print("The farthest station:", temp_STA_list[sorted_with_index[i][0]])
    
                                # print("The original stations of AP2:",AP[min_index].ID_Of_STAs)
                                AP[temp_APs_list[min_index]].ID_Of_STAs.append(temp_STA_list[sorted_with_index[i][0]])
                                # print("The final stations of AP2:",AP[min_index].ID_Of_STAs)
    
                                # print("The origianl dis of AP2:",AP[min_index].Dis_Between_AP_and_STAs)
                                AP[temp_APs_list[min_index]].Dis_Between_AP_and_STAs.append(min_dis)
                                # print("The final dis of AP2:",AP[min_index].Dis_Between_AP_and_STAs)
    
                                # print("Re-associated============",temp_APs_list[min_index])
    
                                tempp_staa=[]
                                for i in range(len(temp_APs_list)):
                                    AP[temp_APs_list[i]].Num_Of_STAs = len(AP[temp_APs_list[i]].ID_Of_STAs)
                                    tempp_staa.append(AP[temp_APs_list[i]].Num_Of_STAs)
    
                                # print("Re-associated stations:==",tempp_staa)
                                # input()
                                dectect.append(difff)
                                # print("dectect:",dectect)
                                # print("length of detection:",len(dectect))
    
                                break
    # print("Reassociation success!")
    # input()

    for i in range(len(AP)):
        
        
        if LocalOptimizerMethod==1:
            AP[i].Num_Of_STAs=len(AP[i].ID_Of_STAs)
        
        
        AP[i].ID_Of_STAs_24G =AP[i].ID_Of_STAs
        AP[i].Dis_Between_AP_and_STAs_24G=AP[i].Dis_Between_AP_and_STAs

        # print(len(AP[i].ID_Of_STAs_24G))
        # print(len(AP[i].Dis_Between_AP_and_STAs_24G))
        # print(np.max(AP[i].Dis_Between_AP_and_STAs))
        # AP[i].Max_Dis

        # input()

        # print('AP ID',i)
        #
        # print(AP[i].ID_Of_STAs_24G)
        # print(AP[i].Dis_Between_AP_and_STAs_24G)
        # print(len(AP[i].ID_Of_STAs_24G))
        # print(len(AP[i].Dis_Between_AP_and_STAs_24G))
        # input()

        AP[i].ID_Of_STAs_5G1 =AP[i].ID_Of_STAs
        AP[i].Dis_Between_AP_and_STAs_5G1=AP[i].Dis_Between_AP_and_STAs

        AP[i].ID_Of_STAs_5G2 =AP[i].ID_Of_STAs
        AP[i].Dis_Between_AP_and_STAs_5G2=AP[i].Dis_Between_AP_and_STAs

        AP[i].ID_Of_STAs_6G =AP[i].ID_Of_STAs
        AP[i].Dis_Between_AP_and_STAs_6G=AP[i].Dis_Between_AP_and_STAs

    return


#Step 1: STA-AP association
# Association()
# input()

def Init_P_and_C():#randomly initialize the number of P and C of each AP
    for i in range(len(AP)):
        # P_index=random.randint(0, len(P))# Select a number in [0,len(p))
        # # AP[i].P=P[P_index]
        # AP[i].P =26
        AP[i].P_24G=27
        AP[i].P_5G1=28
        AP[i].P_5G2=28
        AP[i].P_6G=30



        C_index=random.randint(0,len(C)-1)
        AP[i].C=C[C_index]

        # 2.4G
        C_index=random.randint(0,len(C_2dot4G)-1)
        AP[i].C_24G=C_2dot4G[C_index]

        # 5G1
        C_index=random.randint(0,len(C_5GI)-1)
        AP[i].C_5G1=C_5GI[C_index]

        # 5G2
        C_index=random.randint(0,len(C_5GII)-1)
        AP[i].C_5G2=C_5GII[C_index]

        # 6G
        C_index=random.randint(0,len(C_6G)-1)
        AP[i].C_6G=C_6G[C_index]

# Init_P_and_C()

def Func_CalculateTXPower(DistanceX,B):
    if B=='24G':
        Radio_vector=Radio_vector_24G
        Powerlevel_vector=P_24G
    if B=='5G1':
        Radio_vector = Radio_vector_5G1
        Powerlevel_vector=P_5G1

    if B=='5G2':
        Radio_vector = Radio_vector_5G2
        Powerlevel_vector=P_5G2

    if B=='6G':
        Radio_vector = Radio_vector_6G
        Powerlevel_vector=P_6G

    # print(DistanceX)
    # input()

    # This condition should be updated later
    if DistanceX>Radio_vector[3]:
        TXPower=Powerlevel_vector[3]
        RadioRange=Radio_vector[3]

    if DistanceX>Radio_vector[2] and DistanceX<=Radio_vector[3]:
        TXPower=Powerlevel_vector[3]
        RadioRange=Radio_vector[3]

    if DistanceX>Radio_vector[1] and DistanceX<=Radio_vector[2]:
        TXPower=Powerlevel_vector[2]
        RadioRange = Radio_vector[2]

    if DistanceX>Radio_vector[0] and DistanceX<=Radio_vector[1]:
        TXPower=Powerlevel_vector[1]
        RadioRange = Radio_vector[1]

    if DistanceX<=Radio_vector[0]:
        TXPower=Powerlevel_vector[0]
        RadioRange = Radio_vector[0]
    # print(DistanceX)
    # input()
    # for i in range(len(AP)):
    #     print('P24g',AP[i].r_24G)
    #     print('P5g1', AP[i].r_5G1)
    #     print('P5g2', AP[i].r_5G2)
    #     print('P6g', AP[i].r_6G)
    # Plot_fig()
    # input()
    return TXPower,RadioRange

# Step 2: Calculate the power level of APs
def Func_PowerAdjustment():
    
    # print("I am in Func_PowerAdjustment")
    
    for i in range(len(AP)):
        # print("lenap:",len(AP))
        # print("the ith ap",i)
        for B in Band:
            # exec('N2=AP[i].Dis_Between_AP_and_STAs_{}'.format(B))
            # print(B)
            # Dts=locals()['N2']
            # print(Dts)
            # input()
            if B == '24G':
                ss=len(AP[i].Dis_Between_AP_and_STAs_24G)
                if ss==0:
                    print('Test error: ')
                    print('AP no:',i)
                    print('AP stas:',AP[i].ID_Of_STAs_24G)
                    print('Band:',B)
                    input()
            if B == '5G1':
                ss=len(AP[i].Dis_Between_AP_and_STAs_5G1)
                if ss==0:
                    print('Test error: ')
                    print('AP no:',i)
                    print('AP stas:', AP[i].ID_Of_STAs_5G1)
                    print('Band:', B)
                    input()
            if B == '5G2':
                ss=len(AP[i].Dis_Between_AP_and_STAs_5G2)
                if ss==0:
                    print('Test error: ')
                    print('AP no:',i)
                    print('AP stas:', AP[i].ID_Of_STAs_5G2)
                    print('Band:', B)
                    input()
            if B == '6G':
                ss=len(AP[i].Dis_Between_AP_and_STAs_6G)
                if ss==0:
                    print('Test error: ')
                    print('AP no:',i)
                    print('AP stas:', AP[i].ID_Of_STAs_6G)
                    print('Band:', B)
                    input()

            exec('N1=max(AP[i].Dis_Between_AP_and_STAs_{})'.format(B))
            Dis1=locals()['N1']
            if Dis1 != 0:
                # AP[i].P_24G=Func_CalculateTXPower(Dis1,B)
                # exec('AP[i].P_{},AP[i].r_{}=Func_CalculateTXPower(Dis1,B)'.format(B,B))
                P_val, r_val = Func_CalculateTXPower(Dis1, B)
                setattr(AP[i], f'P_{B}', P_val)
                setattr(AP[i], f'r_{B}', r_val)


    #Can give maximum power levels
    # for i in range(len(AP)):
    #     AP[i].P_24G=max(P_24G)
    #     AP[i].P_5G1 = max(P_5G1)
    #     AP[i].P_5G2 = max(P_5G2)
    #     AP[i].P_6G = max(P_6G)
    # print(AP[0].P_24G)
    # print(AP[0].P_5G1)
    # print(AP[0].P_5G2)
    # print(AP[0].P_6G)
    # input()

# Func_PowerAdjustment()
# input()


def Random_C():
    # Random_action=[]
    # # print(len(AP))
    # # input()
    # for i in range(len(AP)):
    #     #Four channels for each AP
    #     if ES==1:
    #         Random_action.append(random.randint(1,5))
    #         Random_action.append(5)
    #         Random_action.append(20)
    #         Random_action.append(28)
    #     else:
    #         Random_action.append(random.randint(1,5))
    #         Random_action.append(random.randint(5, 20))
    #         Random_action.append(random.randint(20, 28))
    #         Random_action.append(random.randint(28, 59))


    #     # Random_action.append(random.randint(1,2))
    #     # Random_action.append(random.randint(5, 6))
    #     # Random_action.append(random.randint(20, 21))
    #     # Random_action.append(random.randint(28, 29))

    #     #Four power levels for each AP
    #     # Random_action.append(random.randint(24, 28))
    #     # Random_action.append(random.randint(25, 29))
    #     # Random_action.append(random.randint(25, 29))
    #     # Random_action.append(random.randint(27, 31))

    #     Random_action.append(27)
    #     Random_action.append(28)
    #     Random_action.append(28)
    #     Random_action.append(30)

    #     # C_index=random.randint(0,len(C))
    #     # AP[i].C=C[C_index]
    # return Random_action



    action_dim = NumOfAPs * 8
    return np.random.uniform(0.0, 1.0, size=action_dim)

# a=Random_C()
# print(a)
# input()

def FixedResource():

    # print(C)
    # print(P)
    #
    # print(C[0])
    # print(P[0])
    #
    # input()
    if C_P_from_planneingPhase==1:
    
        C_fix=np.loadtxt('Channel_info.csv')
        P_fix=np.loadtxt('Power_info.csv')
    
    if Uniformly_Select==1:
        C_fix=np.loadtxt('Channel_info_ui.csv')
        P_fix=np.loadtxt('Power_info_ui.csv')

    Random_action=[]
    for i in range(len(AP)):
        Random_action.append(C_fix[i][0])
        Random_action.append(C_fix[i][1])
        Random_action.append(C_fix[i][2])
        Random_action.append(C_fix[i][3])

        # Random_action.append(random.randint(1,5))
        # Random_action.append(random.randint(5, 20))
        # Random_action.append(random.randint(20, 28))
        # Random_action.append(random.randint(28, 59))



        Random_action.append(P_fix[i][0])
        Random_action.append(P_fix[i][1])
        Random_action.append(P_fix[i][2])
        Random_action.append(P_fix[i][3])
        # C_index=random.randint(0,len(C))
        # AP[i].C=C[C_index]
        
        # print("action::::",Random_action)
    return Random_action


def Get_NeighboringAPList():
    for i in range(len(AP)):
        for j in range(len(AP)):
            if i!=j:
                for k in range(1,11):
                    exec('tempC1=C_ocs{}'.format(k))
                    TempC=locals()['tempC1']
                    if AP[i].C in TempC and AP[j].C in TempC:
                        if j not in AP[i].NeighborAPList:
                            AP[i].NeighborAPList.append(j)

    for i in range(len(AP)):
        for j in range(len(AP)):
            if i!=j:
                for k in range(1,3):
                    exec('tempC2=C_ocs{}'.format(k))
                    TempC=locals()['tempC2']
                    if AP[i].C_24G in TempC and AP[j].C_24G in TempC:
                        if j not in AP[i].NeighborAPList_24G:
                            AP[i].NeighborAPList_24G.append(j)

    for i in range(len(AP)):
        for j in range(len(AP)):
            if i!=j:
                for k in range(3,11):
                    exec('tempC3=C_ocs{}'.format(k))
                    TempC=locals()['tempC3']
                    if AP[i].C_5G1 in TempC and AP[j].C_5G1 in TempC:
                        if j not in AP[i].NeighborAPList_5G1:
                            AP[i].NeighborAPList_5G1.append(j)

    for i in range(len(AP)):
        for j in range(len(AP)):
            if i!=j:
                for k in range(11,15):
                    exec('tempC4=C_ocs{}'.format(k))
                    TempC=locals()['tempC4']
                    if AP[i].C_5G2 in TempC and AP[j].C_5G2 in TempC:
                        if j not in AP[i].NeighborAPList_5G2:
                            AP[i].NeighborAPList_5G2.append(j)

    for i in range(len(AP)):
        for j in range(len(AP)):
            if i!=j:
                for k in range(15,31):
                    exec('tempC5=C_ocs{}'.format(k))
                    TempC=locals()['tempC5']
                    if AP[i].C_6G in TempC and AP[j].C_6G in TempC:
                        if j not in AP[i].NeighborAPList_6G:
                            AP[i].NeighborAPList_6G.append(j)

# Get_NeighboringAPList()
# input()

def Func_GetNumOfInterferences():
    # === 预加载 C_ocs1~30 ===
    C_ocs_map = {k: globals()[f"C_ocs{k}"] for k in range(1, 31)}
    N = len(AP)

    # === 重置所有 AP 的干扰数 ===
    for ap in AP:
        ap.NumOfInterference_24G = 0
        ap.NumOfInterference_5G1 = 0
        ap.NumOfInterference_5G2 = 0
        ap.NumOfInterference_6G  = 0

    # === 定义各频段的规则 ===
    band_rules = {
        "24G": dict(attr="C_24G", num_attr="NumOfInterference_24G", ranges=[(1, 2)], special=(1, 1)),
        "5G1": dict(attr="C_5G1", num_attr="NumOfInterference_5G1", ranges=[(3, 11)]),
        "5G2": dict(attr="C_5G2", num_attr="NumOfInterference_5G2", ranges=[(11, 15)], special=(24, 24)),
        "6G":  dict(attr="C_6G",  num_attr="NumOfInterference_6G",  ranges=[(15, 31)]),
    }

    # === 遍历 AP 对（上三角） ===
    for i in range(N):
        for j in range(i + 1, N):
            for band, rules in band_rules.items():
                ci = getattr(AP[i], rules["attr"])
                cj = getattr(AP[j], rules["attr"])

                # 遍历干扰范围
                for start, end in rules["ranges"]:
                    for k in range(start, end):
                        if ci in C_ocs_map[k] and cj in C_ocs_map[k]:
                            setattr(AP[i], rules["num_attr"], getattr(AP[i], rules["num_attr"]) + 1)
                            setattr(AP[j], rules["num_attr"], getattr(AP[j], rules["num_attr"]) + 1)

                # 特殊规则（比如 2.4G 的 ==1，5G2 的 ==24）
                if "special" in rules:
                    val1, val2 = rules["special"]
                    if ci == val1 and cj == val2:
                        setattr(AP[i], rules["num_attr"], getattr(AP[i], rules["num_attr"]) + 1)
                        setattr(AP[j], rules["num_attr"], getattr(AP[j], rules["num_attr"]) + 1)


def Func_GetNumOfInterferences__origianl2025_10_1():

    for i in range(len(AP)):
        for j in range(len(AP)):
            if i!=j:
                for k in range(1,3):
                    exec('tempC2=C_ocs{}'.format(k))
                    TempC=locals()['tempC2']
                    if AP[i].C_24G in TempC and AP[j].C_24G in TempC:
                        AP[i].NumOfInterference_24G=AP[i].NumOfInterference_24G+1
                if AP[i].C_24G == 1 and AP[j].C_24G ==1:
                    AP[i].NumOfInterference_24G = AP[i].NumOfInterference_24G + 1


    for i in range(len(AP)):
        for j in range(len(AP)):
            if i!=j:
                for k in range(3,11):
                    exec('tempC3=C_ocs{}'.format(k))
                    TempC=locals()['tempC3']
                    if AP[i].C_5G1 in TempC and AP[j].C_5G1 in TempC:
                        AP[i].NumOfInterference_5G1 = AP[i].NumOfInterference_5G1 + 1


    for i in range(len(AP)):
        for j in range(len(AP)):
            if i!=j:
                for k in range(11,15):
                    exec('tempC4=C_ocs{}'.format(k))
                    TempC=locals()['tempC4']
                    if AP[i].C_5G2 in TempC and AP[j].C_5G2 in TempC:
                        AP[i].NumOfInterference_5G2 = AP[i].NumOfInterference_5G2 + 1
                if AP[i].C_5G2 == 24 and AP[j].C_5G2 ==24:
                    AP[i].NumOfInterference_5G2 = AP[i].NumOfInterference_5G2 + 1


    for i in range(len(AP)):
        for j in range(len(AP)):
            if i!=j:
                for k in range(15,31):
                    exec('tempC5=C_ocs{}'.format(k))
                    TempC=locals()['tempC5']
                    if AP[i].C_6G in TempC and AP[j].C_6G in TempC:
                        AP[i].NumOfInterference_6G = AP[i].NumOfInterference_6G + 1



def Func_GetNeiInt():  # Get interference AP list
    # === 清空邻居列表 ===
    for ap in AP:
        ap.NeiInt_24G = []
        ap.NeiInt_5G1 = []
        ap.NeiInt_5G2 = []
        ap.NeiInt_6G = []

    # === 把 C_ocs1...30 收集成列表 ===
    C_ocs_groups = [
        C_ocs1, C_ocs2,               # 24G
        C_ocs3, C_ocs4, C_ocs5, C_ocs6, C_ocs7, C_ocs8, C_ocs9, C_ocs10,   # 5G1
        C_ocs11, C_ocs12, C_ocs13, C_ocs14,                               # 5G2
        C_ocs15, C_ocs16, C_ocs17, C_ocs18, C_ocs19, C_ocs20,
        C_ocs21, C_ocs22, C_ocs23, C_ocs24, C_ocs25,
        C_ocs26, C_ocs27, C_ocs28, C_ocs29, C_ocs30                       # 6G
    ]

    # === 建立 channel -> group_id 的映射 ===
    max_channel = max(max(g) for g in C_ocs_groups)
    channel_to_group = np.full(max_channel + 1, -1, dtype=int)
    for gid, group in enumerate(C_ocs_groups):
        for ch in group:
            channel_to_group[ch] = gid

    # === 主循环：检查每个 band ===
    
    # import numpy as np

    # 将 AP 对象转换为坐标矩阵 N×3
    coords = np.array([[ap.x, ap.y, ap.z] for ap in AP])
    
    # 计算距离矩阵
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]  # N×N×3
    dist_matrix = np.linalg.norm(diff, axis=2)  # N×N
    
    # print(dist_matrix)
    # print(dist_matrix[0])
    # print(dist_matrix[0][2])
    # print(dist_matrix[0,2])  用这种方式取索引
    # input()

    
    for i, ap_i in enumerate(AP):
        for j, ap_j in enumerate(AP):
            if i == j:
                continue

            # 24G
            if ap_i.C_24G == ap_j.C_24G or (
                channel_to_group[ap_i.C_24G] != -1
                and channel_to_group[ap_i.C_24G] == channel_to_group[ap_j.C_24G]
            ):
                
                if dist_matrix[i,j]<Dij_24G:
                
                    ap_i.NeiInt_24G.append(j)
                    # print("dist_matrix[i,j]",dist_matrix[i,j])
                    # print("Dij_24G",Dij_24G)
                    # print("24Gint:",j)

            # 5G1
            if ap_i.C_5G1 == ap_j.C_5G1 or (
                channel_to_group[ap_i.C_5G1] != -1
                and channel_to_group[ap_i.C_5G1] == channel_to_group[ap_j.C_5G1]
            ):
                if dist_matrix[i,j]<Dij_5G1:
                    ap_i.NeiInt_5G1.append(j)
                    # print("5G1int:",j)

            # 5G2
            if ap_i.C_5G2 == ap_j.C_5G2 or (
                channel_to_group[ap_i.C_5G2] != -1
                and channel_to_group[ap_i.C_5G2] == channel_to_group[ap_j.C_5G2]
            ):
                if dist_matrix[i,j]<Dij_5G2:
                    ap_i.NeiInt_5G2.append(j)
                    # print("5G2int:",j)

            # 6G
            if ap_i.C_6G == ap_j.C_6G or (
                channel_to_group[ap_i.C_6G] != -1
                and channel_to_group[ap_i.C_6G] == channel_to_group[ap_j.C_6G]
            ):
                if dist_matrix[i,j]<Dij_6G:
                    ap_i.NeiInt_6G.append(j)
                    # print("dist_matrix[i,j]",dist_matrix[i,j])
                    # print("Dij_6G",Dij_6G)
                    # print("6Gint:",j)
                    
    # input()
    
    # for i, ap_i in enumerate(AP):
    #     ap_i.NeiInt_24G=[]
    #     ap_i.NeiInt_5G1=[]
    #     ap_i.NeiInt_5G2=[]
    #     ap_i.NeiInt_6G=[]
    #     print("AP:",i)
    #     print(ap_i.NeiInt_24G)
    #     print("---")
    #     print(ap_i.NeiInt_5G1)
    #     print("---")
    #     print(ap_i.NeiInt_5G2)
    #     print("---")
    #     print(ap_i.NeiInt_6G)
    #     print("---")
    # input()



def Func_GetNeiInt_original():#Get interference AP list

    for i in range(len(AP)):
        AP[i].NeiInt_24G=[]
        AP[i].NeiInt_5G1 = []
        AP[i].NeiInt_5G2 = []
        AP[i].NeiInt_6G = []

    for i in range(len(AP)):
        for j in range(len(AP)):
            if i!=j:
                if AP[i].C_24G!=AP[j].C_24G:
                    for k in range(1,3):

                        exec('tempC2=C_ocs{}'.format(k))
                        TempC=locals()['tempC2']
                        # print('TempC=',TempC)
                        if AP[i].C_24G in TempC and AP[j].C_24G in TempC:
                            if j not in AP[i].NeiInt_24G:
                                AP[i].NeiInt_24G.append(j)
                            # AP[i].NumOfInterference_24G=AP[i].NumOfInterference_24G+1
                    # if AP[i].C_24G == 1 and AP[j].C_24G ==1:
                    #     AP[i].NeiInt_24G.append(j)
                if AP[i].C_24G==AP[j].C_24G:
                    AP[i].NeiInt_24G.append(j)
            #
            # print("AP[i].c=",AP[i].C_24G)
            # print("AP[j].c=", AP[j].C_24G)
            # print('i=',i)
            # print("j=",j)
            # print(AP[i].NeiInt_24G)
            # input()

    for i in range(len(AP)):
        for j in range(len(AP)):
            if i!=j:
                if AP[i].C_5G1 != AP[j].C_5G1:
                    for k in range(3,11):
                        exec('tempC3=C_ocs{}'.format(k))
                        TempC=locals()['tempC3']
                        if AP[i].C_5G1 in TempC and AP[j].C_5G1 in TempC:
                            if j not in AP[i].NeiInt_5G1:
                                AP[i].NeiInt_5G1.append(j)
                if AP[i].C_5G1==AP[j].C_5G1:
                    AP[i].NeiInt_5G1.append(j)

            # print("AP[i].c=",AP[i].C_5G1)
            # print("AP[j].c=", AP[j].C_5G1)
            # print('i=',i)
            # print("j=",j)
            # print(AP[i].NeiInt_5G1)
            # input()


    for i in range(len(AP)):
        for j in range(len(AP)):
            if i!=j:
                if AP[i].C_5G2 != AP[j].C_5G2:
                    for k in range(11,15):
                        exec('tempC4=C_ocs{}'.format(k))
                        TempC=locals()['tempC4']
                        if AP[i].C_5G2 in TempC and AP[j].C_5G2 in TempC:
                            if j not in AP[i].NeiInt_5G2:
                                AP[i].NeiInt_5G2.append(j)
                if AP[i].C_5G2 == AP[j].C_5G2:
                    AP[i].NeiInt_5G2.append(j)


    for i in range(len(AP)):
        for j in range(len(AP)):
            if i!=j:
                if AP[i].C_6G != AP[j].C_6G:
                    for k in range(15,31):
                        exec('tempC5=C_ocs{}'.format(k))
                        TempC=locals()['tempC5']
                        if AP[i].C_6G in TempC and AP[j].C_6G in TempC:
                            if j not in AP[i].NeiInt_6G:
                                AP[i].NeiInt_6G.append(j)
                if AP[i].C_6G == AP[j].C_6G:
                    AP[i].NeiInt_6G.append(j)

def Func_GetNeighbors_All():#因为原始信道是固定的[哎，为何不解释清楚啊：这是初始状态，后续还要更新]，即所有都是邻居

    # print("I am in Func_GetNeighbors_All")
    for i in range(len(AP)):
        Nei=np.arange(0,len(AP),1)
        AP[i].NeighborAPList_24G=np.delete(Nei,[i])
        AP[i].NeighborAPList_5G1 = np.delete(Nei, [i])
        AP[i].NeighborAPList_5G2 = np.delete(Nei, [i])
        AP[i].NeighborAPList_6G = np.delete(Nei, [i])
#Step 3
# Func_GetNeighbors_All()



#Step 3: Channel assignment
# ```python
def Func_ChannelAssignment():
    # print("I am in Func_ChannelAssignment")
    index_24G, index_5G1, index_5G2, index_6G = [], [], [], []

    # === 统计每个AP的STA数量 ===
    for i in range(len(AP)):
        for B in Band:
            index_temp = len(getattr(AP[i], f"ID_Of_STAs_{B}"))
            locals()[f"index_{B}"].append(index_temp)

    # === 根据用户数量排序（降序） ===
    Index_24G = np.argsort(index_24G)[::-1]
    Index_5G1 = np.argsort(index_5G1)[::-1]
    Index_5G2 = np.argsort(index_5G2)[::-1]
    Index_6G  = np.argsort(index_6G)[::-1]

    for i in range(len(AP)):
        # print("total ap:", len(AP))
        # print("now channels for ith ap:", i)

        for B in Band:
            # === 选择候选信道集 ===
            if B == '24G':
                C_temp = set(C_2dot4G)
                Index_B = Index_24G
            elif B == '5G1':
                C_temp = set(C_5GI)
                Index_B = Index_5G1
            elif B == '5G2':
                C_temp = set(C_5GII)
                Index_B = Index_5G2
            elif B == '6G':
                C_temp = set(C_6G)
                Index_B = Index_6G

            ap_obj = AP[Index_B[i]]
            neighbor_list = getattr(ap_obj, f"NeighborAPList_{B}")
            Lencc = len(neighbor_list)

            # === 遍历邻居，移除冲突信道 ===
            for j in range(Lencc):
                neighbor_ap = AP[neighbor_list[j]]
                ap_c = getattr(neighbor_ap, f"C_{B}")

                # 用映射表代替 if-else
                remove_map = {
                    1: [1],
                    2: C_ocs1,
                    3: C_ocs2,
                    4: C_ocs1 + C_ocs2,
                    5: C_ocs3,
                    6: C_ocs4,
                    7: C_ocs5,
                    8: C_ocs6,
                    9: C_ocs7,
                    10: C_ocs8,
                    11: C_ocs9,
                    12: C_ocs10,
                    13: C_ocs3 + C_ocs4,
                    14: C_ocs5 + C_ocs6,
                    15: C_ocs7 + C_ocs8,
                    16: C_ocs9 + C_ocs10,
                    17: C_ocs3 + C_ocs4 + C_ocs5 + C_ocs6,
                    18: C_ocs7 + C_ocs8 + C_ocs9 + C_ocs10,
                    19: list(C_temp),  # 全部移除
                    20: C_ocs11,
                    21: C_ocs12,
                    22: C_ocs13,
                    23: C_ocs14,
                    24: [24],
                    25: C_ocs11 + C_ocs12,
                    26: C_ocs13 + C_ocs14,
                    27: [c for c in C_temp if c != 24],  # 只保留24
                    28: C_ocs15,
                    29: C_ocs16,
                    30: C_ocs17,
                    31: C_ocs18,
                    32: C_ocs19,
                    33: C_ocs20,
                    34: C_ocs21,
                    35: C_ocs22,
                    36: C_ocs23,
                    37: C_ocs24,
                    38: C_ocs25,
                    39: C_ocs26,
                    40: C_ocs27,
                    41: C_ocs28,
                    42: C_ocs29,
                    43: C_ocs30,
                    44: C_ocs15 + C_ocs16,
                    45: C_ocs17 + C_ocs18,
                    46: C_ocs19 + C_ocs20,
                    47: C_ocs21 + C_ocs22,
                    48: C_ocs23 + C_ocs24,
                    49: C_ocs25 + C_ocs26,
                    50: C_ocs27 + C_ocs28,
                    51: C_ocs29 + C_ocs30,
                    52: C_ocs15 + C_ocs16 + C_ocs17 + C_ocs18,
                    53: C_ocs19 + C_ocs20 + C_ocs21 + C_ocs22,
                    54: C_ocs23 + C_ocs24 + C_ocs25 + C_ocs26,
                    55: C_ocs27 + C_ocs28 + C_ocs29 + C_ocs30,
                    56: C_ocs15 + C_ocs16 + C_ocs17 + C_ocs18 + 
                        C_ocs19 + C_ocs20 + C_ocs21 + C_ocs22,
                    57: C_ocs23 + C_ocs24 + C_ocs25 + C_ocs26 +
                        C_ocs27 + C_ocs28 + C_ocs29 + C_ocs30,
                    58: list(C_temp),  # 全部移除
                }

                if ap_c in remove_map:
                    C_temp -= set(remove_map[ap_c])

            # === 排序并分配信道 ===
            C_temp = sorted(C_temp)
            if len(C_temp) > 0:
                setattr(ap_obj, f"CandidateChannelsList_{B}", C_temp)
                setattr(ap_obj, f"C_{B}", C_temp[0])
            else:
                # 没有可用信道，选择干扰最少的
                cc = np.inf
                overlapchannel = 0
                for q in range(Lencc):
                    neighbor_ap = AP[neighbor_list[q]]
                    lenqq = getattr(neighbor_ap, f"NumOfInterference_{B}")
                    cqq = getattr(neighbor_ap, f"C_{B}")
                    if lenqq <= cc and cqq != 0:
                        cc = lenqq
                        overlapchannel = cqq

                if overlapchannel > 0:
                    setattr(ap_obj, f"C_{B}", overlapchannel)
                    # 重置干扰数并重新计算
                    for temp_ap in AP:
                        temp_ap.NumOfInterference_24G = 0
                        temp_ap.NumOfInterference_5G1 = 0
                        temp_ap.NumOfInterference_5G2 = 0
                        temp_ap.NumOfInterference_6G  = 0
                    Func_GetNumOfInterferences()
                else:
                    print('No channels can be used!')



def Func_ChannelAssignment____original():#original version 2025-9-28
    print("I am in  Func_ChannelAssignment")
    index_24G=[]
    index_5G1=[]
    index_5G2=[]
    index_6G=[]
    for i in range(len(AP)):
        for B in Band:
            exec('index_temp=len(AP[i].ID_Of_STAs_{})'.format(B))
            indexx=locals()['index_temp']
            exec('index_{}.append(indexx)'.format(B))
    Index_24G=np.argsort(index_24G)
    Index_24G=Index_24G[::-1]
    Index_5G1=np.argsort(index_5G1)
    Index_5G1=Index_5G1[::-1]
    Index_5G2=np.argsort(index_5G2)
    Index_5G2=Index_5G2[::-1]
    Index_6G=np.argsort(index_6G)
    Index_6G=Index_6G[::-1]
    # print(index_5G2)
    # print(Index_5G2)
    # print(Index_5G2)
    # input()

    for i in range(len(AP)):
        print("total ap:",len(AP))
        print("now channels for ith ap:",i,)
        for B in Band:
            if B=='24G':
                C_temp=C_2dot4G
            if B=='5G1':
                C_temp=C_5GI
            if B=='5G2':
                C_temp=C_5GII
            if B=='6G':
                C_temp=C_6G
            # lencc=len(AP[Index_24G[i]].NeighborAPList_24G)
            exec('lencc=len(AP[Index_{}[i]].NeighborAPList_{})'.format(B,B))
            Lencc=locals()['lencc']
            # print(B)
            # print(Lencc)
            # input()
            # temp_c=AP[AP[Index_24G[i]].NeighborAPList_24G[j]].C_24G
            # exec('temp_c=AP[Index_{}[i]].C_{}'.format(B,B))
            # ap_c=locals()['temp_c']
            # print('apc',ap_c)
            # input()
            for j in range(Lencc):
                exec('temp_c=AP[AP[Index_{}[i]].NeighborAPList_{}[j]].C_{}'.format(B,B,B))
                ap_c = locals()['temp_c']
                if ap_c==1:
                    # print(C_temp)
                    # input()
                    # C_temp.remove(1)
                    C_temp=list(set(C_temp)-set([1]))
                if ap_c==2:
                    C_temp=list(set(C_temp)-set(C_ocs1))
                if ap_c==3:
                    C_temp=list(set(C_temp)-set(C_ocs2))
                if ap_c==4:
                    C_temp=list(set(C_temp)-set(C_ocs1))
                    C_temp=list(set(C_temp)-set(C_ocs2))
                if ap_c==5:
                    C_temp=list(set(C_temp)-set(C_ocs3))
                if ap_c==6:
                    C_temp=list(set(C_temp)-set(C_ocs4))
                if ap_c==7:
                    C_temp=list(set(C_temp)-set(C_ocs5))
                if ap_c==8:
                    C_temp=list(set(C_temp)-set(C_ocs6))
                if ap_c==9:
                    C_temp=list(set(C_temp)-set(C_ocs7))
                if ap_c==10:
                    C_temp=list(set(C_temp)-set(C_ocs8))
                if ap_c==11:
                    C_temp=list(set(C_temp)-set(C_ocs9))
                if ap_c==12:
                    C_temp=list(set(C_temp)-set(C_ocs10))
                if ap_c==13:
                    C_temp=list(set(C_temp)-set(C_ocs3))
                    C_temp = list(set(C_temp) - set(C_ocs4))
                if ap_c==14:
                    C_temp=list(set(C_temp)-set(C_ocs5))
                    C_temp = list(set(C_temp) - set(C_ocs6))
                if ap_c==15:
                    C_temp=list(set(C_temp)-set(C_ocs7))
                    C_temp = list(set(C_temp) - set(C_ocs8))
                if ap_c==16:
                    C_temp=list(set(C_temp)-set(C_ocs9))
                    C_temp = list(set(C_temp) - set(C_ocs10))
                if ap_c==17:
                    C_temp=list(set(C_temp)-set(C_ocs3))
                    C_temp = list(set(C_temp) - set(C_ocs4))
                    C_temp = list(set(C_temp) - set(C_ocs5))
                    C_temp = list(set(C_temp) - set(C_ocs6))
                if ap_c==18:
                    C_temp=list(set(C_temp)-set(C_ocs7))
                    C_temp = list(set(C_temp) - set(C_ocs8))
                    C_temp = list(set(C_temp) - set(C_ocs9))
                    C_temp = list(set(C_temp) - set(C_ocs10))
                if ap_c==19:
                    C_temp=[]
                if ap_c==20:
                    C_temp = list(set(C_temp) - set(C_ocs11))
                if ap_c==21:
                    C_temp = list(set(C_temp) - set(C_ocs12))
                if ap_c == 22:
                    C_temp = list(set(C_temp) - set(C_ocs13))
                if ap_c == 23:
                    C_temp = list(set(C_temp) - set(C_ocs14))
                if ap_c == 24:
                    # C_temp.remove(24)
                    C_temp = list(set(C_temp) - set([24]))
                if ap_c==25:
                    C_temp = list(set(C_temp) - set(C_ocs11))
                    C_temp = list(set(C_temp) - set(C_ocs12))
                if ap_c==26:
                    C_temp = list(set(C_temp) - set(C_ocs13))
                    C_temp = list(set(C_temp) - set(C_ocs14))
                if ap_c==27:
                    C_temp=[24]
                if ap_c==28:
                    C_temp = list(set(C_temp) - set(C_ocs15))
                if ap_c==29:
                    C_temp = list(set(C_temp) - set(C_ocs16))
                if ap_c==30:
                    C_temp = list(set(C_temp) - set(C_ocs17))
                if ap_c==31:
                    C_temp = list(set(C_temp) - set(C_ocs18))
                if ap_c==32:
                    C_temp = list(set(C_temp) - set(C_ocs19))
                if ap_c==33:
                    C_temp = list(set(C_temp) - set(C_ocs20))
                if ap_c==34:
                    C_temp = list(set(C_temp) - set(C_ocs21))
                if ap_c==35:
                    C_temp = list(set(C_temp) - set(C_ocs22))
                if ap_c==36:
                    C_temp = list(set(C_temp) - set(C_ocs23))
                if ap_c==37:
                    C_temp = list(set(C_temp) - set(C_ocs24))
                if ap_c==38:
                    C_temp = list(set(C_temp) - set(C_ocs25))
                if ap_c==39:
                    C_temp = list(set(C_temp) - set(C_ocs26))
                if ap_c==40:
                    C_temp = list(set(C_temp) - set(C_ocs27))
                if ap_c==41:
                    C_temp = list(set(C_temp) - set(C_ocs28))
                if ap_c==42:
                    C_temp = list(set(C_temp) - set(C_ocs29))
                if ap_c==43:
                    C_temp = list(set(C_temp) - set(C_ocs30))
                if ap_c==44:
                    C_temp = list(set(C_temp) - set(C_ocs15))
                    C_temp = list(set(C_temp) - set(C_ocs16))
                if ap_c==45:
                    C_temp = list(set(C_temp) - set(C_ocs17))
                    C_temp = list(set(C_temp) - set(C_ocs18))
                if ap_c==46:
                    C_temp = list(set(C_temp) - set(C_ocs19))
                    C_temp = list(set(C_temp) - set(C_ocs20))
                if ap_c==47:
                    C_temp = list(set(C_temp) - set(C_ocs21))
                    C_temp = list(set(C_temp) - set(C_ocs22))
                if ap_c==48:
                    C_temp = list(set(C_temp) - set(C_ocs23))
                    C_temp = list(set(C_temp) - set(C_ocs24))
                if ap_c==49:
                    C_temp = list(set(C_temp) - set(C_ocs25))
                    C_temp = list(set(C_temp) - set(C_ocs26))
                if ap_c==50:
                    C_temp = list(set(C_temp) - set(C_ocs27))
                    C_temp = list(set(C_temp) - set(C_ocs28))
                if ap_c==51:
                    C_temp = list(set(C_temp) - set(C_ocs29))
                    C_temp = list(set(C_temp) - set(C_ocs30))
                if ap_c==52:
                    C_temp = list(set(C_temp) - set(C_ocs15))
                    C_temp = list(set(C_temp) - set(C_ocs16))
                    C_temp = list(set(C_temp) - set(C_ocs17))
                    C_temp = list(set(C_temp) - set(C_ocs18))
                if ap_c==53:
                    C_temp = list(set(C_temp) - set(C_ocs19))
                    C_temp = list(set(C_temp) - set(C_ocs20))
                    C_temp = list(set(C_temp) - set(C_ocs21))
                    C_temp = list(set(C_temp) - set(C_ocs22))
                if ap_c==54:
                    C_temp = list(set(C_temp) - set(C_ocs23))
                    C_temp = list(set(C_temp) - set(C_ocs24))
                    C_temp = list(set(C_temp) - set(C_ocs25))
                    C_temp = list(set(C_temp) - set(C_ocs26))
                if ap_c==55:
                    C_temp = list(set(C_temp) - set(C_ocs27))
                    C_temp = list(set(C_temp) - set(C_ocs28))
                    C_temp = list(set(C_temp) - set(C_ocs29))
                    C_temp = list(set(C_temp) - set(C_ocs30))
                if ap_c==56:
                    C_temp = list(set(C_temp) - set(C_ocs15))
                    C_temp = list(set(C_temp) - set(C_ocs16))
                    C_temp = list(set(C_temp) - set(C_ocs17))
                    C_temp = list(set(C_temp) - set(C_ocs18))
                    C_temp = list(set(C_temp) - set(C_ocs19))
                    C_temp = list(set(C_temp) - set(C_ocs20))
                    C_temp = list(set(C_temp) - set(C_ocs21))
                    C_temp = list(set(C_temp) - set(C_ocs22))
                if ap_c==57:
                    C_temp = list(set(C_temp) - set(C_ocs23))
                    C_temp = list(set(C_temp) - set(C_ocs24))
                    C_temp = list(set(C_temp) - set(C_ocs25))
                    C_temp = list(set(C_temp) - set(C_ocs26))
                    C_temp = list(set(C_temp) - set(C_ocs27))
                    C_temp = list(set(C_temp) - set(C_ocs28))
                    C_temp = list(set(C_temp) - set(C_ocs29))
                    C_temp = list(set(C_temp) - set(C_ocs30))
                if ap_c==58:
                    C_temp = []

            C_temp.sort()

            if len(C_temp) > 0:
                # CandidateChannelsList_24G=C_temp
                exec('AP[Index_{}[i]].CandidateChannelsList_{}=C_temp'.format(B,B))
                exec('AP[Index_{}[i]].C_{}=C_temp[0]'.format(B, B))
            if len(C_temp)==0:
                cc=np.inf
                overlapchannel=0
                for q in range(Lencc): # Find the AP with the least number of interferences
                    exec('lenq=AP[AP[Index_{}[i]].NeighborAPList_{}[q]].NumOfInterference_{}'.format(B,B,B))
                    lenqq=locals()['lenq']
                    exec('cq=AP[AP[Index_{}[i]].NeighborAPList_{}[q]].C_{}'.format(B,B,B))
                    cqq=locals()['cq']
                    if lenqq <=cc and cqq != 0:
                        cc=lenqq
                        overlapchannel=cqq
                        qq=q


                if overlapchannel>0:
                    exec('AP[Index_{}[i]].C_{}=overlapchannel'.format(B,B))

                    for temp_i in range(len(AP)):
                        AP[temp_i].NumOfInterference_24G=0
                        AP[temp_i].NumOfInterference_5G1 = 0
                        AP[temp_i].NumOfInterference_5G2 = 0
                        AP[temp_i].NumOfInterference_6G = 0
                    Func_GetNumOfInterferences()
                        # exec('AP[Index_{}[i]].NumOfInterference_{}=AP[Index_{}[i]].NumOfInterference_{}+1'.format(B,B,B,B))
                        # exec('AP[AP[Index_{}[i]].NeighborAPList_{}[qq]].NumOfInterference_{}=AP[AP[Index_{}[i]].NeighborAPList_{}[qq]].NumOfInterference_{}+1'.format(B,B,B,B,B,B))

                else:
                    print('No channels can be used!')


    if LocalOptimizerMethod==0:
        for i in range(len(AP)):
            for B in Band:
                if B =='24G':
                    # indexJ_24G=np.arange(4,3,-1)
                    indexJ_24G=[4]
                    # tempC=C_2dot4G
                if B=='5G1':
                    indexJ_5G1=np.arange(19,12,-1)
                    # tempC = C_5GI
                if B=='5G2':
                    indexJ_5G2=np.arange(27,24,-1)
                    # tempC = C_5GII
                if B=='6G':
                    indexJ_6G=np.arange(58,43,-1)
                    # tempC = C_6G
    
                exec('IndexJ=indexJ_{}'.format(B))
                Ind=locals()['IndexJ']
                # print(Ind)
                # print(B)
                # input()
                for j in range(len(Ind)):
                    indicator=0
                    exec('lenccc=len(AP[Index_{}[i]].NeighborAPList_{})'.format(B,B))
                    Lenccc=locals()['lenccc']
                    for k in range(Lenccc):
                        # apc=AP[AP[Index_24G[i]].NeighborAPList_24G[k]].C_24G
                        exec('apc=AP[AP[Index_{}[i]].NeighborAPList_{}[k]].C_{}'.format(B,B,B))
                        apcc=locals()['apc']
                        if apcc==0:
                            indicator=0
                        if apcc==1:
                            if Ind[j] in [1]:
                                indicator=1
                        if apcc==2:
                            # if tempC[j] in C_ocs1:
                            indicator=1
                        if apcc==3:
                            # if tempC[j] in C_ocs2:
                            indicator=1
                        if apcc==4:
                            # if tempC[j] in C_ocs1 or tempC[j] in C_ocs2:
                            indicator=1
                        if apcc==5:
                            if Ind[j] in C_ocs3:
                                indicator=1
                        if apcc==6:
                            if Ind[j] in C_ocs4:
                                indicator=1
                        if apcc==7:
                            if Ind[j] in C_ocs5:
                                indicator=1
                        if apcc==8:
                            if Ind[j] in C_ocs6:
                                indicator=1
                        if apcc==9:
                            if Ind[j] in C_ocs7:
                                indicator=1
                        if apcc==10:
                            if Ind[j] in C_ocs8:
                                indicator=1
                        if apcc==11:
                            if Ind[j] in C_ocs9:
                                indicator=1
                        if apcc==12:
                            if Ind[j] in C_ocs10:
                                indicator=1
                        if apcc==13:
                            if Ind[j] in C_ocs3 or Ind[j] in C_ocs4:
                                indicator=1
                        if apcc==14:
                            if Ind[j] in C_ocs5 or Ind[j] in C_ocs6:
                                indicator=1
                        if apcc==15:
                            if Ind[j] in C_ocs7 or Ind[j] in C_ocs8:
                                indicator=1
                        if apcc==16:
                            if Ind[j] in C_ocs9 or Ind[j] in C_ocs10:
                                indicator=1
                        if apcc==17:
                            # if tempC[j] in C_ocs3 or tempC[j] in C_ocs4 or tempC[j] in C_ocs5 or tempC[j] in C_ocs6:
                            indicator=1
                        if apcc==18:
                            # if tempC[j] in C_ocs7 or tempC[j] in C_ocs8 or tempC[j] in C_ocs9 or tempC[j] in C_ocs10:
                            indicator=1
                        if apcc==19:
                            # if tempC[j] in C_ocs1:
                            indicator=1
                        if apcc==20:
                            if Ind[j] in C_ocs11:
                                indicator=1
                        if apcc==21:
                            if Ind[j] in C_ocs12:
                                indicator=1
                        if apcc==22:
                            if Ind[j] in C_ocs13:
                                indicator=1
                        if apcc==23:
                            if Ind[j] in C_ocs14:
                                indicator=1
                        if apcc==24:
                            if Ind[j] in [24]:
                                indicator=1
                        if apcc==25:
                            # if tempC[j] in C_ocs1:
                            indicator=1
                        if apcc==26:
                            # if tempC[j] in C_ocs1:
                            indicator=1
                        if apcc==27:
                            # if tempC[j] in C_ocs1:
                            indicator=1
                        if apcc==28:
                            if Ind[j] in C_ocs15:
                                indicator=1
                        if apcc==29:
                            if Ind[j] in C_ocs16:
                                indicator=1
                        if apcc==30:
                            if Ind[j] in C_ocs17:
                                indicator=1
                        if apcc==31:
                            if Ind[j] in C_ocs18:
                                indicator=1
                        if apcc==32:
                            if Ind[j] in C_ocs19:
                                indicator=1
                        if apcc==33:
                            if Ind[j] in C_ocs20:
                                indicator=1
                        if apcc==34:
                            if Ind[j] in C_ocs21:
                                indicator=1
                        if apcc==35:
                            if Ind[j] in C_ocs22:
                                indicator=1
                        if apcc==36:
                            if Ind[j] in C_ocs23:
                                indicator=1
                        if apcc==37:
                            if Ind[j] in C_ocs24:
                                indicator=1
                        if apcc==38:
                            if Ind[j] in C_ocs25:
                                indicator=1
                        if apcc==39:
                            if Ind[j] in C_ocs26:
                                indicator=1
                        if apcc==40:
                            if Ind[j] in C_ocs27:
                                indicator=1
                        if apcc==41:
                            if Ind[j] in C_ocs28:
                                indicator=1
                        if apcc==42:
                            if Ind[j] in C_ocs29:
                                indicator=1
                        if apcc==43:
                            if Ind[j] in C_ocs30:
                                indicator=1
                        if apcc==44:
                            if Ind[j] in C_ocs15 or Ind[j] in C_ocs16:
                                indicator=1
                        if apcc==45:
                            if Ind[j] in C_ocs17 or Ind[j] in C_ocs18:
                                indicator=1
                        if apcc==46:
                            if Ind[j] in C_ocs19 or Ind[j] in C_ocs20:
                                indicator=1
                        if apcc==47:
                            if Ind[j] in C_ocs21 or Ind[j] in C_ocs22:
                                indicator=1
                        if apcc==48:
                            if Ind[j] in C_ocs23 or Ind[j] in C_ocs24:
                                indicator=1
                        if apcc==49:
                            if Ind[j] in C_ocs25 or Ind[j] in C_ocs26:
                                indicator=1
                        if apcc==50:
                            if Ind[j] in C_ocs27 or Ind[j] in C_ocs28:
                                indicator=1
                        if apcc==51:
                            if Ind[j] in C_ocs29 or Ind[j] in C_ocs30:
                                indicator=1
                        if apcc==52:
                            if Ind[j] in C_ocs15 or Ind[j] in C_ocs16 or Ind[j] in C_ocs17 or Ind[j] in C_ocs18:
                                indicator=1
                        if apcc==53:
                            if Ind[j] in C_ocs19 or Ind[j] in C_ocs20 or Ind[j] in C_ocs21 or Ind[j] in C_ocs22:
                                indicator=1
                        if apcc==54:
                            if Ind[j] in C_ocs23 or Ind[j] in C_ocs24 or Ind[j] in C_ocs25 or Ind[j] in C_ocs26:
                                indicator=1
                        if apcc==55:
                            if Ind[j] in C_ocs27 or Ind[j] in C_ocs28 or Ind[j] in C_ocs29 or Ind[j] in C_ocs30:
                                indicator=1
                        if apcc==56:
                            # if tempC[j] in C_ocs1:
                            indicator=1
                        if apcc==57:
                            # if tempC[j] in C_ocs1:
                            indicator=1
                        if apcc==58:
                            # if tempC[j] in C_ocs1:
                            indicator=1
                    if indicator==0:
                        # testt=0
                        # print('test')
                        # AP[Index_24G[i]].C_24G=tempC[j]
                        exec('AP[Index_{}[i]].C_{}=Ind[j]'.format(B,B))
    
        for i in range(len(AP)):
            for B in Band:
                # if B =='24G':
                #     # indexJ_24G=np.arange(4,3,-1)
                #     indexJ_24G=[4]
                #     # tempC=C_2dot4G
                # if B=='5G1':
                #     indexJ_5G1=np.arange(19,12,-1)
                #     # tempC = C_5GI
                if B=='5G2':
                    indexJ_5G2=np.arange(27,24,-1)
                    tempC = C_5GII
                if B=='6G':
                    indexJ_6G=np.arange(58,51,-1)
                    # tempC = C_6G
    
                exec('IndexJ=indexJ_{}'.format(B))
                Ind=locals()['IndexJ']
                # print(Ind)
                # print(B)
                # input()
                for j in range(len(Ind)):
                    indicator=0
                    exec('lenccc=len(AP[Index_{}[i]].NeighborAPList_{})'.format(B,B))
                    Lenccc=locals()['lenccc']
                    for k in range(Lenccc):
                        # apc=AP[AP[Index_24G[i]].NeighborAPList_24G[k]].C_24G
                        exec('apc=AP[AP[Index_{}[i]].NeighborAPList_{}[k]].C_{}'.format(B,B,B))
                        apcc=locals()['apc']
                        if apcc==0:
                            indicator=0
                        if apcc==1:
                            if Ind[j] in [1]:
                                indicator=1
                        if apcc==2:
                            # if tempC[j] in C_ocs1:
                            indicator=1
                        if apcc==3:
                            # if tempC[j] in C_ocs2:
                            indicator=1
                        if apcc==4:
                            # if tempC[j] in C_ocs1 or tempC[j] in C_ocs2:
                            indicator=1
                        if apcc==5:
                            if Ind[j] in C_ocs3:
                                indicator=1
                        if apcc==6:
                            if Ind[j] in C_ocs4:
                                indicator=1
                        if apcc==7:
                            if Ind[j] in C_ocs5:
                                indicator=1
                        if apcc==8:
                            if Ind[j] in C_ocs6:
                                indicator=1
                        if apcc==9:
                            if Ind[j] in C_ocs7:
                                indicator=1
                        if apcc==10:
                            if Ind[j] in C_ocs8:
                                indicator=1
                        if apcc==11:
                            if Ind[j] in C_ocs9:
                                indicator=1
                        if apcc==12:
                            if Ind[j] in C_ocs10:
                                indicator=1
                        if apcc==13:
                            if Ind[j] in C_ocs3 or Ind[j] in C_ocs4:
                                indicator=1
                        if apcc==14:
                            if Ind[j] in C_ocs5 or Ind[j] in C_ocs6:
                                indicator=1
                        if apcc==15:
                            if Ind[j] in C_ocs7 or Ind[j] in C_ocs8:
                                indicator=1
                        if apcc==16:
                            if Ind[j] in C_ocs9 or Ind[j] in C_ocs10:
                                indicator=1
                        if apcc==17:
                            # if tempC[j] in C_ocs3 or tempC[j] in C_ocs4 or tempC[j] in C_ocs5 or tempC[j] in C_ocs6:
                            indicator=1
                        if apcc==18:
                            # if tempC[j] in C_ocs7 or tempC[j] in C_ocs8 or tempC[j] in C_ocs9 or tempC[j] in C_ocs10:
                            indicator=1
                        if apcc==19:
                            # if tempC[j] in C_ocs1:
                            indicator=1
                        if apcc==20:
                            if Ind[j] in C_ocs11:
                                indicator=1
                        if apcc==21:
                            if Ind[j] in C_ocs12:
                                indicator=1
                        if apcc==22:
                            if Ind[j] in C_ocs13:
                                indicator=1
                        if apcc==23:
                            if Ind[j] in C_ocs14:
                                indicator=1
                        if apcc==24:
                            if Ind[j] in [24]:
                                indicator=1
                        if apcc==25:
                            # if tempC[j] in C_ocs1:
                            indicator=1
                        if apcc==26:
                            # if tempC[j] in C_ocs1:
                            indicator=1
                        if apcc==27:
                            # if tempC[j] in C_ocs1:
                            indicator=1
                        if apcc==28:
                            if Ind[j] in C_ocs15:
                                indicator=1
                        if apcc==29:
                            if Ind[j] in C_ocs16:
                                indicator=1
                        if apcc==30:
                            if Ind[j] in C_ocs17:
                                indicator=1
                        if apcc==31:
                            if Ind[j] in C_ocs18:
                                indicator=1
                        if apcc==32:
                            if Ind[j] in C_ocs19:
                                indicator=1
                        if apcc==33:
                            if Ind[j] in C_ocs20:
                                indicator=1
                        if apcc==34:
                            if Ind[j] in C_ocs21:
                                indicator=1
                        if apcc==35:
                            if Ind[j] in C_ocs22:
                                indicator=1
                        if apcc==36:
                            if Ind[j] in C_ocs23:
                                indicator=1
                        if apcc==37:
                            if Ind[j] in C_ocs24:
                                indicator=1
                        if apcc==38:
                            if Ind[j] in C_ocs25:
                                indicator=1
                        if apcc==39:
                            if Ind[j] in C_ocs26:
                                indicator=1
                        if apcc==40:
                            if Ind[j] in C_ocs27:
                                indicator=1
                        if apcc==41:
                            if Ind[j] in C_ocs28:
                                indicator=1
                        if apcc==42:
                            if Ind[j] in C_ocs29:
                                indicator=1
                        if apcc==43:
                            if Ind[j] in C_ocs30:
                                indicator=1
                        if apcc==44:
                            if Ind[j] in C_ocs15 or Ind[j] in C_ocs16:
                                indicator=1
                        if apcc==45:
                            if Ind[j] in C_ocs17 or Ind[j] in C_ocs18:
                                indicator=1
                        if apcc==46:
                            if Ind[j] in C_ocs19 or Ind[j] in C_ocs20:
                                indicator=1
                        if apcc==47:
                            if Ind[j] in C_ocs21 or Ind[j] in C_ocs22:
                                indicator=1
                        if apcc==48:
                            if Ind[j] in C_ocs23 or Ind[j] in C_ocs24:
                                indicator=1
                        if apcc==49:
                            if Ind[j] in C_ocs25 or Ind[j] in C_ocs26:
                                indicator=1
                        if apcc==50:
                            if Ind[j] in C_ocs27 or Ind[j] in C_ocs28:
                                indicator=1
                        if apcc==51:
                            if Ind[j] in C_ocs29 or Ind[j] in C_ocs30:
                                indicator=1
                        if apcc==52:
                            if Ind[j] in C_ocs15 or Ind[j] in C_ocs16 or Ind[j] in C_ocs17 or Ind[j] in C_ocs18:
                                indicator=1
                        if apcc==53:
                            if Ind[j] in C_ocs19 or Ind[j] in C_ocs20 or Ind[j] in C_ocs21 or Ind[j] in C_ocs22:
                                indicator=1
                        if apcc==54:
                            if Ind[j] in C_ocs23 or Ind[j] in C_ocs24 or Ind[j] in C_ocs25 or Ind[j] in C_ocs26:
                                indicator=1
                        if apcc==55:
                            if Ind[j] in C_ocs27 or Ind[j] in C_ocs28 or Ind[j] in C_ocs29 or Ind[j] in C_ocs30:
                                indicator=1
                        if apcc==56:
                            # if tempC[j] in C_ocs1:
                            indicator=1
                        if apcc==57:
                            # if tempC[j] in C_ocs1:
                            indicator=1
                        if apcc==58:
                            # if tempC[j] in C_ocs1:
                            indicator=1
                    if indicator==0:
                        # testt=0
                        # print('test')
                        # AP[Index_24G[i]].C_24G=tempC[j]
                        exec('AP[Index_{}[i]].C_{}=Ind[j]'.format(B,B))

    if len(AP)==4:
        for B in Band:
            if B=="5G2":
                AP[Index_5G2[0]].C_5G2=25
                AP[Index_5G2[1]].C_5G2=23
                AP[Index_5G2[2]].C_5G2=22
                AP[Index_5G2[3]].C_5G2=24
            if B=="6G":
                AP[0].C_6G=52
                AP[1].C_6G=53
                AP[2].C_6G=54
                AP[3].C_6G=55






    # print(index_24G,index_5G1,index_5G2,index_6G)
    # print(Index_24G,Index_5G1,Index_5G2,Index_6G)
    # input()
#Step 4 , 5
# Func_ChannelAssignment()
# Func_GetNeiInt() #Get interference AP list of each AP
# input()

# import numpy as np
# from copy import deepcopy
def Func_PowerReAdjustment():
    # print("I am in Func_PowerReAdjustment")

    # === 备份功率，而不是 deepcopy 整个 AP ===
    P_backup = [{B: getattr(ap, f"P_{B}") for B in Band} for ap in AP]

    # === 信道 -> 分组映射表（用数组比 dict 快） ===
    max_channel = max(max(g) for g in [
        C_ocs1, C_ocs2, C_ocs3, C_ocs4, C_ocs5,
        C_ocs6, C_ocs7, C_ocs8, C_ocs9, C_ocs10,
        C_ocs11, C_ocs12, C_ocs13, C_ocs14, C_ocs15,
        C_ocs16, C_ocs17, C_ocs18, C_ocs19, C_ocs20,
        C_ocs21, C_ocs22, C_ocs23, C_ocs24, C_ocs25,
        C_ocs26, C_ocs27, C_ocs28, C_ocs29, C_ocs30
    ])
    channel_to_group = np.full(max_channel + 1, -1, dtype=int)
    for group_id, group in enumerate([
        C_ocs1, C_ocs2, C_ocs3, C_ocs4, C_ocs5,
        C_ocs6, C_ocs7, C_ocs8, C_ocs9, C_ocs10,
        C_ocs11, C_ocs12, C_ocs13, C_ocs14, C_ocs15,
        C_ocs16, C_ocs17, C_ocs18, C_ocs19, C_ocs20,
        C_ocs21, C_ocs22, C_ocs23, C_ocs24, C_ocs25,
        C_ocs26, C_ocs27, C_ocs28, C_ocs29, C_ocs30
    ]):
        for ch in group:
            channel_to_group[ch] = group_id

    # === 一次性计算所有邻居，避免多次调用 ===
    Func_GetNeiInt()  
    all_nei = {i: {B: getattr(ap, f"NeiInt_{B}") for B in Band} for i, ap in enumerate(AP)}

    # === 主循环 ===
    for i, ap in enumerate(AP):
        for B in Band:
            Power_vec = globals()[f"P_{B}"]
            P_temp = getattr(ap, f"P_{B}")

            for j in Power_vec:
                if j <= P_temp:  # 跳过没意义的值
                    continue

                setattr(ap, f"P_{B}", j)
                nei_list = all_nei[i][B]

                if not nei_list:
                    continue

                AP_C1 = getattr(ap, f"C_{B}")
                AP_C2_list = [getattr(AP[k], f"C_{B}") for k in nei_list]

                # 用数组查表比 dict.get() 快
                g1 = channel_to_group[AP_C1]
                g2 = channel_to_group[AP_C2_list]

                temp_CCI = np.sum(g2 == g1)
                NumInts = getattr(ap, f"NumOfInterference_{B}")

                if temp_CCI > NumInts:
                    # 回退
                    setattr(ap, f"P_{B}", P_backup[i][B])
                    break
                else:
                    P_backup[i][B] = j


def Func_PowerReAdjustment_newV1():
    print("I am in Func_PowerReAdjustment")
    global AP
    APPP = deepcopy(AP)

    # 把 C_ocs1 ... C_ocs30 收集到一个列表
    C_ocs_groups = [
        C_ocs1, C_ocs2, C_ocs3, C_ocs4, C_ocs5,
        C_ocs6, C_ocs7, C_ocs8, C_ocs9, C_ocs10,
        C_ocs11, C_ocs12, C_ocs13, C_ocs14, C_ocs15,
        C_ocs16, C_ocs17, C_ocs18, C_ocs19, C_ocs20,
        C_ocs21, C_ocs22, C_ocs23, C_ocs24, C_ocs25,
        C_ocs26, C_ocs27, C_ocs28, C_ocs29, C_ocs30
    ]

    # 建立 channel -> group_id 的映射，加速查找
    channel_to_group = {}
    for group_id, group in enumerate(C_ocs_groups):
        for ch in group:
            channel_to_group[ch] = group_id

    for i, ap in enumerate(AP):
        # print(len(AP))
        # print("the ith ap", i)

        for B in Band:
            Powerlevel_vector = globals()[f"P_{B}"]
            P_temp = getattr(ap, f"P_{B}")

            for j in Powerlevel_vector:
                flag = 0
                if j <= P_temp:
                    continue

                setattr(ap, f"P_{B}", j)
                setattr(ap, f"NeiInt_{B}", [])

                Func_GetNeiInt()
                NeiIntt = getattr(ap, f"NeiInt_{B}")

                if not NeiIntt:
                    continue

                # flag = 0
                AP_C1 = getattr(ap, f"C_{B}")
                AP_C2_list = np.array([getattr(AP[k], f"C_{B}") for k in NeiIntt])

                # 获取对应 group_id
                g1 = channel_to_group.get(AP_C1, -1)
                g2 = np.array([channel_to_group.get(c, -1) for c in AP_C2_list])

                # 向量化比较：AP_C1 和每个 AP_C2 是否在同一组
                temp_CCI = np.sum(g2 == g1)

                NumInts = getattr(ap, f"NumOfInterference_{B}")

                if temp_CCI > NumInts:
                    setattr(ap, f"P_{B}", getattr(APPP[i], f"P_{B}"))
                    flag = 1
                    break
                else:
                    setattr(APPP[i], f"P_{B}", j)

            if flag == 1:
                break




def Func_PowerReAdjustment___origianl():#original 2025-9-28
    print("I am in Func_PowerReAdjustment")
    global AP
    APPP=deepcopy(AP)
    for i in range(len(AP)):
        print(len(AP))
        print("the ith ap",i)
        for B in Band:
            exec('PLv=P_{}'.format(B))
            Powerlevel_vector=locals()['PLv']
            for j in Powerlevel_vector:
                temp_CCI=0
                flag=0
                exec('p_temp=AP[i].P_{}'.format(B))
                P_temp=locals()['p_temp']
                if j>P_temp:
                    exec('AP[i].P_{}=j'.format(B))
                    exec('AP[i].NeiInt_{}=[]'.format(B))

                    # print('call 1')
                    Func_GetNeiInt()
                    exec('NeiInt_temp=AP[i].NeiInt_{}'.format(B))
                    NeiIntt=locals()['NeiInt_temp']

                    if len(NeiIntt)>0:

                        for k in NeiIntt:
                            exec('AP_c1=AP[i].C_{}'.format(B))
                            AP_C1=locals()['AP_c1']

                            exec('AP_c2=AP[k].C_{}'.format(B))
                            AP_C2 = locals()['AP_c2']
                            if AP_C1 in C_ocs1 and AP_C2 in C_ocs1:
                                temp_CCI=temp_CCI+1
                            if AP_C1 in C_ocs2 and AP_C2 in C_ocs2:
                                temp_CCI=temp_CCI+1
                            if AP_C1 in C_ocs3 and AP_C2 in C_ocs3:
                                temp_CCI=temp_CCI+1
                            if AP_C1 in C_ocs4 and AP_C2 in C_ocs4:
                                temp_CCI=temp_CCI+1
                            if AP_C1 in C_ocs5 and AP_C2 in C_ocs5:
                                temp_CCI=temp_CCI+1
                            if AP_C1 in C_ocs6 and AP_C2 in C_ocs6:
                                temp_CCI=temp_CCI+1
                            if AP_C1 in C_ocs7 and AP_C2 in C_ocs7:
                                temp_CCI=temp_CCI+1
                            if AP_C1 in C_ocs8 and AP_C2 in C_ocs8:
                                temp_CCI=temp_CCI+1
                            if AP_C1 in C_ocs9 and AP_C2 in C_ocs9:
                                temp_CCI=temp_CCI+1
                            if AP_C1 in C_ocs10 and AP_C2 in C_ocs10:
                                temp_CCI=temp_CCI+1
                            if AP_C1 in C_ocs11 and AP_C2 in C_ocs11:
                                temp_CCI=temp_CCI+1
                            if AP_C1 in C_ocs12 and AP_C2 in C_ocs12:
                                temp_CCI=temp_CCI+1
                            if AP_C1 in C_ocs13 and AP_C2 in C_ocs13:
                                temp_CCI=temp_CCI+1
                            if AP_C1 in C_ocs14 and AP_C2 in C_ocs14:
                                temp_CCI=temp_CCI+1
                            if AP_C1 in C_ocs15 and AP_C2 in C_ocs15:
                                temp_CCI=temp_CCI+1
                            if AP_C1 in C_ocs16 and AP_C2 in C_ocs16:
                                temp_CCI=temp_CCI+1
                            if AP_C1 in C_ocs17 and AP_C2 in C_ocs17:
                                temp_CCI=temp_CCI+1
                            if AP_C1 in C_ocs18 and AP_C2 in C_ocs18:
                                temp_CCI=temp_CCI+1
                            if AP_C1 in C_ocs19 and AP_C2 in C_ocs19:
                                temp_CCI=temp_CCI+1
                            if AP_C1 in C_ocs20 and AP_C2 in C_ocs20:
                                temp_CCI=temp_CCI+1
                            if AP_C1 in C_ocs21 and AP_C2 in C_ocs21:
                                temp_CCI=temp_CCI+1
                            if AP_C1 in C_ocs22 and AP_C2 in C_ocs22:
                                temp_CCI=temp_CCI+1
                            if AP_C1 in C_ocs23 and AP_C2 in C_ocs23:
                                temp_CCI=temp_CCI+1
                            if AP_C1 in C_ocs24 and AP_C2 in C_ocs24:
                                temp_CCI=temp_CCI+1
                            if AP_C1 in C_ocs25 and AP_C2 in C_ocs25:
                                temp_CCI=temp_CCI+1
                            if AP_C1 in C_ocs26 and AP_C2 in C_ocs26:
                                temp_CCI=temp_CCI+1
                            if AP_C1 in C_ocs27 and AP_C2 in C_ocs27:
                                temp_CCI=temp_CCI+1
                            if AP_C1 in C_ocs28 and AP_C2 in C_ocs28:
                                temp_CCI=temp_CCI+1
                            if AP_C1 in C_ocs29 and AP_C2 in C_ocs29:
                                temp_CCI=temp_CCI+1
                            if AP_C1 in C_ocs30 and AP_C2 in C_ocs30:
                                temp_CCI=temp_CCI+1

                            exec('NumII=AP[i].NumOfInterference_{}'.format(B))
                            NumInts=locals()['NumII']

                            if temp_CCI>NumInts:
                                exec('AP[i].P_{}=APPP[i].P_{}'.format(B,B))
                                # AP[i].P_24G=APP[i].P_24G
                                flag=1
                                break
                            else:
                                exec('APPP[i].P_{}=j'.format(B))
                if flag==1:
                    break



#Calculate the interference  of APs and STAs
def Get_Interference_Of_APs_and_STAs():
    global P_interferenceTop_W
    if Num_of_interferences==0:
        
        if SafetyMargin==1: #just for Table Ix
            ###P_InterferenceTop_dBm
            P_InterferenceTop_dBm=-66
            
            P_interferenceTop_W=10**((P_InterferenceTop_dBm)/10)/1000
        
        
        
        for i in range(len(AP)):
            AP[i].Total_Interference_24G=P_interferenceTop_W
            AP[i].Total_Interference_5G1 = P_interferenceTop_W
            AP[i].Total_Interference_5G2 = P_interferenceTop_W
            AP[i].Total_Interference_6G = P_interferenceTop_W
        for j in range(len(STA)):
            STA[j].Total_Interference_24G=P_interferenceTop_W
            STA[j].Total_Interference_5G1 = P_interferenceTop_W
            STA[j].Total_Interference_5G2 = P_interferenceTop_W
            STA[j].Total_Interference_6G = P_interferenceTop_W





    if Num_of_interferences>0:
        for i in range(len(AP)):
            for j in range(Num_of_interferences):

#3d distance new
                dis_from_AP_to_IS = ((AP[i].x - IS[j].x) ** 2 + (AP[i].y - IS[j].y) ** 2+ (AP[i].z - IS[j].z) ** 2) ** (1 / 2)

                #3d distance old
                # dis_from_AP_to_IS=(dis_from_AP_to_IS ** 2 + H_sta ** 2) ** (0.5)

                # print('j=',j)
                #
                # print('dis_from_AP_to_IS=',dis_from_AP_to_IS)
                # input()
                AP[i].Dis_to_IS.append(dis_from_AP_to_IS)



                for k in range(1,3): #2.4G
                    exec('tempC6=C_ocs{}'.format(k))
                    TempC = locals()['tempC6']

                    # print("TempC=",TempC)
                    # print('AP[i].C_24G=',AP[i].C_24G)
                    # print('IS[j].c=',IS[j].c)
                    # input()

                    if AP[i].C_24G in TempC and IS[j].c in TempC:#不用采用信道相等，采用都属于同个Ocs即可
                        # print('i=',i)
                        # print('j=',j)
                        #
                        # print('AP[i].C_24G=',AP[i].C_24G)
                        # print('IS[j].c=',IS[j].c)
                        #
                        # input()
                        # if AP[i].C in C_2dot4G:
                        f = fc_2dot4G
                        # if AP[i].C in C_5GI:
                        #     f = fc_5GI
                        # dis_from_AP_to_IS=((AP[i].x-IS[j].x)**2+(AP[i].y-IS[j].y)**2)**(1/2)
                        # AP[i].Dis_to_IS.append(dis_from_AP_to_IS)
                        Prx_from_IS=IS[j].p-PL(dis_from_AP_to_IS,f)#in dBm
                        Rx_Power=10**(Prx_from_IS/10)/1000 #in W
                        AP[i].Total_Interference_24G=AP[i].Total_Interference_24G+Rx_Power

                        # print('IS[j].p=',IS[j].p)
                        # print('dis_from_AP_to_IS=',dis_from_AP_to_IS)
                        # print('PL(dis_from_AP_to_IS,f)=',PL(dis_from_AP_to_IS,f))
                        # print('Prx_from_IS=',Prx_from_IS)
                        # print('Rx_Power_inW=',Rx_Power)
                        # print('AP[i].Total_Interference_24G=',AP[i].Total_Interference_24G)
                        #
                        # input()

                        for k in range(len(AP[i].ID_Of_STAs_24G)):
                            
                            #3D distance new
                            dis_from_STA_to_IS=((STA[AP[i].ID_Of_STAs_24G[k]].x-IS[j].x)**2+(STA[AP[i].ID_Of_STAs_24G[k]].y-IS[j].y)**2\
                                                +(STA[AP[i].ID_Of_STAs_24G[k]].z-IS[j].z)**2)**(1/2)
                                
                            STA[AP[i].ID_Of_STAs_24G[k]].Dis_to_IS.append(dis_from_STA_to_IS)
                            Prx_from_IS=IS[j].p-PL(dis_from_STA_to_IS,f)#in dBm
                            Rx_Power = 10 ** (Prx_from_IS / 10) / 1000  # in W
                            # print('source_j=',j)
                            # print('Rx_Power',Rx_Power)
                            STA[AP[i].ID_Of_STAs_24G[k]].Total_Interference_24G=STA[AP[i].ID_Of_STAs_24G[k]].Total_Interference_24G+Rx_Power
                            # print('inter=',STA[AP[i].ID_Of_STAs_24G[k]].Total_Interference)
                            # input()

                        break

                    if AP[i].C_24G ==1 and IS[j].c ==1:#不用采用信道相等，采用都属于同个Ocs即可
                        # print('i=',i)
                        # print('j=',j)
                        #
                        # print('AP[i].C_24G=',AP[i].C_24G)
                        # print('IS[j].c=',IS[j].c)
                        #
                        # input()
                        # if AP[i].C in C_2dot4G:
                        f = fc_2dot4G
                        # if AP[i].C in C_5GI:
                        #     f = fc_5GI
                        # dis_from_AP_to_IS=((AP[i].x-IS[j].x)**2+(AP[i].y-IS[j].y)**2)**(1/2)
                        # AP[i].Dis_to_IS.append(dis_from_AP_to_IS)
                        Prx_from_IS=IS[j].p-PL(dis_from_AP_to_IS,f)#in dBm
                        Rx_Power=10**(Prx_from_IS/10)/1000 #in W
                        AP[i].Total_Interference_24G=AP[i].Total_Interference_24G+Rx_Power

                        # print('IS[j].p=',IS[j].p)
                        # print('dis_from_AP_to_IS=',dis_from_AP_to_IS)
                        # print('PL(dis_from_AP_to_IS,f)=',PL(dis_from_AP_to_IS,f))
                        # print('Prx_from_IS=',Prx_from_IS)
                        # print('Rx_Power_inW=',Rx_Power)
                        # print('AP[i].Total_Interference_24G=',AP[i].Total_Interference_24G)
                        #
                        # input()

                        for k in range(len(AP[i].ID_Of_STAs_24G)):
                            
                            #3D distance new
                            dis_from_STA_to_IS=((STA[AP[i].ID_Of_STAs_24G[k]].x-IS[j].x)**2+(STA[AP[i].ID_Of_STAs_24G[k]].y-IS[j].y)**2\
                                                +(STA[AP[i].ID_Of_STAs_24G[k]].z-IS[j].z)**2)**(1/2)
                            
                            STA[AP[i].ID_Of_STAs_24G[k]].Dis_to_IS.append(dis_from_STA_to_IS)
                            Prx_from_IS=IS[j].p-PL(dis_from_STA_to_IS,f)#in dBm
                            Rx_Power = 10 ** (Prx_from_IS / 10) / 1000  # in W
                            # print('source_j=',j)
                            # print('Rx_Power',Rx_Power)
                            STA[AP[i].ID_Of_STAs_24G[k]].Total_Interference_24G=STA[AP[i].ID_Of_STAs_24G[k]].Total_Interference_24G+Rx_Power
                            # print('inter=',STA[AP[i].ID_Of_STAs_24G[k]].Total_Interference)
                            # input()

                        break





                for k in range(3,11): #5G1
                    exec('tempC7=C_ocs{}'.format(k))
                    TempC = locals()['tempC7']
                    if AP[i].C_5G1 in TempC and IS[j].c in TempC:#不用采用信道相等，采用都属于同个Ocs即可
                        # if AP[i].C in C_2dot4G:

                        # print('i=', i)
                        # print('j=', j)
                        #
                        # print('AP[i].C_5G1=', AP[i].C_5G1)
                        # print('IS[j].c=', IS[j].c)
                        # input()


                        f = fc_5GI
                        # if AP[i].C in C_5GI:
                        #     f = fc_5GI
                        # dis_from_AP_to_IS=((AP[i].x-IS[j].x)**2+(AP[i].y-IS[j].y)**2)**(1/2)
                        # AP[i].Dis_to_IS.append(dis_from_AP_to_IS)
                        Prx_from_IS=IS[j].p-PL(dis_from_AP_to_IS,f)#in dBm



                        Rx_Power=10**(Prx_from_IS/10)/1000 #in W
                        AP[i].Total_Interference_5G1=AP[i].Total_Interference_5G1+Rx_Power

                        # print('IS[j].p=',IS[j].p)
                        # print('dis_from_AP_to_IS=', dis_from_AP_to_IS)
                        # print('PL(dis_from_AP_to_IS,f)=',PL(dis_from_AP_to_IS,f))
                        # print('Prx_from_IS=',Prx_from_IS)
                        # print('Rx_Power_inW=',Rx_Power)
                        # print('AP[i].Total_Interference_5G1=',AP[i].Total_Interference_5G1)
                        #
                        # input()



                        for k in range(len(AP[i].ID_Of_STAs_5G1)):
                            
                            #3D distance
                            dis_from_STA_to_IS=((STA[AP[i].ID_Of_STAs_5G1[k]].x-IS[j].x)**2+(STA[AP[i].ID_Of_STAs_5G1[k]].y-IS[j].y)**2\
                                                +(STA[AP[i].ID_Of_STAs_5G1[k]].z-IS[j].z)**2)**(1/2)
                            STA[AP[i].ID_Of_STAs_5G1[k]].Dis_to_IS.append(dis_from_STA_to_IS)
                            Prx_from_IS=IS[j].p-PL(dis_from_STA_to_IS,f)#in dBm
                            Rx_Power = 10 ** (Prx_from_IS / 10) / 1000  # in W
                            STA[AP[i].ID_Of_STAs_5G1[k]].Total_Interference_5G1=STA[AP[i].ID_Of_STAs_5G1[k]].Total_Interference_5G1+Rx_Power

                        break


                for k in range(11,15): #5G2
                    exec('tempC8=C_ocs{}'.format(k))
                    TempC = locals()['tempC8']
                    if AP[i].C_5G2 in TempC and IS[j].c in TempC:#不用采用信道相等，采用都属于同个Ocs即可
                        # if AP[i].C in C_2dot4G:
                        # print('i=', i)
                        # print('j=', j)
                        #
                        # print('AP[i].C_5G2=', AP[i].C_5G2)
                        # print('IS[j].c=', IS[j].c)
                        #
                        # input()
                        f = fc_5GII
                        # if AP[i].C in C_5GI:
                        #     f = fc_5GI
                        # dis_from_AP_to_IS=((AP[i].x-IS[j].x)**2+(AP[i].y-IS[j].y)**2)**(1/2)
                        # AP[i].Dis_to_IS.append(dis_from_AP_to_IS)
                        Prx_from_IS=IS[j].p-PL(dis_from_AP_to_IS,f)#in dBm
                        Rx_Power=10**(Prx_from_IS/10)/1000 #in W
                        AP[i].Total_Interference_5G2=AP[i].Total_Interference_5G2+Rx_Power

                        # print('IS[j].p=',IS[j].p)
                        # print('dis_from_AP_to_IS=', dis_from_AP_to_IS)
                        # print('PL(dis_from_AP_to_IS,f)=',PL(dis_from_AP_to_IS,f))
                        # print('Prx_from_IS=',Prx_from_IS)
                        # print('Rx_Power_inW=',Rx_Power)
                        # print('AP[i].Total_Interference_5G2=',AP[i].Total_Interference_5G2)
                        #
                        # input()

                        for k in range(len(AP[i].ID_Of_STAs_5G2)):
                            #3D distance
                            dis_from_STA_to_IS=((STA[AP[i].ID_Of_STAs_5G2[k]].x-IS[j].x)**2+(STA[AP[i].ID_Of_STAs_5G2[k]].y-IS[j].y)**2\
                                                +(STA[AP[i].ID_Of_STAs_5G2[k]].z-IS[j].z)**2)**(1/2)
                            STA[AP[i].ID_Of_STAs_5G2[k]].Dis_to_IS.append(dis_from_STA_to_IS)
                            Prx_from_IS=IS[j].p-PL(dis_from_STA_to_IS,f)#in dBm
                            Rx_Power = 10 ** (Prx_from_IS / 10) / 1000  # in W
                            STA[AP[i].ID_Of_STAs_5G2[k]].Total_Interference_5G2=STA[AP[i].ID_Of_STAs_5G2[k]].Total_Interference_5G2+Rx_Power
                        break

                    if AP[i].C_5G2 ==24 and IS[j].c ==24:#不用采用信道相等，采用都属于同个Ocs即可
                        # if AP[i].C in C_2dot4G:
                        # print('i=', i)
                        # print('j=', j)
                        #
                        # print('AP[i].C_5G2=', AP[i].C_5G2)
                        # print('IS[j].c=', IS[j].c)
                        #
                        # input()
                        f = fc_5GII
                        # if AP[i].C in C_5GI:
                        #     f = fc_5GI
                        # dis_from_AP_to_IS=((AP[i].x-IS[j].x)**2+(AP[i].y-IS[j].y)**2)**(1/2)
                        # AP[i].Dis_to_IS.append(dis_from_AP_to_IS)
                        Prx_from_IS=IS[j].p-PL(dis_from_AP_to_IS,f)#in dBm
                        Rx_Power=10**(Prx_from_IS/10)/1000 #in W
                        AP[i].Total_Interference_5G2=AP[i].Total_Interference_5G2+Rx_Power

                        # print('IS[j].p=',IS[j].p)
                        # print('dis_from_AP_to_IS=', dis_from_AP_to_IS)
                        # print('PL(dis_from_AP_to_IS,f)=',PL(dis_from_AP_to_IS,f))
                        # print('Prx_from_IS=',Prx_from_IS)
                        # print('Rx_Power_inW=',Rx_Power)
                        # print('AP[i].Total_Interference_5G2=',AP[i].Total_Interference_5G2)
                        #
                        # input()

                        for k in range(len(AP[i].ID_Of_STAs_5G2)):
                            #3D distance
                            dis_from_STA_to_IS=((STA[AP[i].ID_Of_STAs_5G2[k]].x-IS[j].x)**2+(STA[AP[i].ID_Of_STAs_5G2[k]].y-IS[j].y)**2\
                                                +(STA[AP[i].ID_Of_STAs_5G2[k]].z-IS[j].z)**2)**(1/2)
                            STA[AP[i].ID_Of_STAs_5G2[k]].Dis_to_IS.append(dis_from_STA_to_IS)
                            Prx_from_IS=IS[j].p-PL(dis_from_STA_to_IS,f)#in dBm
                            Rx_Power = 10 ** (Prx_from_IS / 10) / 1000  # in W
                            STA[AP[i].ID_Of_STAs_5G2[k]].Total_Interference_5G2=STA[AP[i].ID_Of_STAs_5G2[k]].Total_Interference_5G2+Rx_Power
                        break

                for k in range(15,31): #6G
                    exec('tempC9=C_ocs{}'.format(k))
                    TempC = locals()['tempC9']
                    # print('IS[j].c=', IS[j].c)
                    # input()
                    if AP[i].C_6G in TempC and IS[j].c in TempC:#不用采用信道相等，采用都属于同个Ocs即可
                        # if AP[i].C in C_2dot4G:
                        # print('i=', i)
                        # print('j=', j)
                        #
                        # print('AP[i].C_6G=', AP[i].C_6G)
                        # print('IS[j].c=', IS[j].c)
                        #
                        # input()
                        f = fc_6G
                        # if AP[i].C in C_5GI:
                        #     f = fc_5GI
                        # dis_from_AP_to_IS=((AP[i].x-IS[j].x)**2+(AP[i].y-IS[j].y)**2)**(1/2)
                        # AP[i].Dis_to_IS.append(dis_from_AP_to_IS)

                        # if dis_from_AP_to_IS<dis_from_AP_to_IS:
                        #     dis_from_AP_to_IS=dis_from_AP_to_IS
                        #     print('dd=', dis_from_AP_to_IS)

                        Prx_from_IS=IS[j].p-PL(dis_from_AP_to_IS,f)#in dBm++

                        # print('IS[j].p=',IS[j].p)

                        # print('LossFromIS=',PL(dis_from_AP_to_IS,f))
                        #
                        # print('Prx_from_IS=',Prx_from_IS)
                        # input()
                        Rx_Power=10**(Prx_from_IS/10)/1000 #in W
                        AP[i].Total_Interference_6G=AP[i].Total_Interference_6G+Rx_Power

                        # print('IS[j].c=', IS[j].c)
                        # print('AP[i].C_6G=',AP[i].C_6G)
                        # print('IS[j].p=',IS[j].p)
                        # print('dis_from_AP_to_IS=', dis_from_AP_to_IS)
                        # print('PL(dis_from_AP_to_IS,f)=',PL(dis_from_AP_to_IS,f))
                        # print('Prx_from_IS=',Prx_from_IS)
                        # print('Rx_Power_inW=',Rx_Power)
                        # print('AP[i].Total_Interference_6G=',AP[i].Total_Interference_6G)
                        #
                        # input()

                        for k in range(len(AP[i].ID_Of_STAs_6G)):
                            #3D distance
                            dis_from_STA_to_IS=((STA[AP[i].ID_Of_STAs_6G[k]].x-IS[j].x)**2+(STA[AP[i].ID_Of_STAs_6G[k]].y-IS[j].y)**2\
                                                +(STA[AP[i].ID_Of_STAs_6G[k]].z-IS[j].z)**2)**(1/2)
                            STA[AP[i].ID_Of_STAs_6G[k]].Dis_to_IS.append(dis_from_STA_to_IS)
                            Prx_from_IS=IS[j].p-PL(dis_from_STA_to_IS,f)#in dBm
                            Rx_Power = 10 ** (Prx_from_IS / 10) / 1000  # in W
                            STA[AP[i].ID_Of_STAs_6G[k]].Total_Interference_6G=STA[AP[i].ID_Of_STAs_6G[k]].Total_Interference_6G+Rx_Power

                        break



#Step 6
# Get_Interference_Of_APs_and_STAs()
# input()
# print(P_index,P0)
# print(C_index,C0)

# Plot_fig()
# input()

#bluetooth 5.0 输出功率-20dBm ～ 8dBm
#cc2420 -25,-15,-10,-5,0 dBm
# a=np.log10(1000)

# Prx=-20
# Pmw1=10**((Prx)/10)
# print(Pmw1)
#
# Prx=2
# Pmw2=10**((Prx)/10)
# print(Pmw2)
# a=(Pmw2)/(Pmw1)
# print(a)

# dB=10*np.log10((Pmw2)/(Pmw1))
# print(dB)

# self.ID_Of_STAs
# self.Num_Of_STAs
# self.Dis_Between_AP_and_STAs

Throughput_of_STAs=[]
def RUassignment():

    for i in range(len(AP)):#2.4G
        # print(i)
        for B in Band:
            if Band.index(B)==0:
                FF=[20,40]
                f=fc_2dot4G
            if Band.index(B)==1:
                FF=[20,40,80,160]
                f=fc_5GI
            if Band.index(B)==2:
                FF=[20,40,80]
                f=fc_5GII
            if Band.index(B)==3:
                FF=[20,40,80,160,320]
                f=fc_6G

            for F in FF:
                # Fre = []
                exec('Fre=Frequence{}_{}'.format(F,B))
                Freq=locals()['Fre']
                exec('chh=AP[i].C_{}'.format(B))
                ch=locals()['chh']
                # print(ch)
                # input()
                if ch in Freq:
                # if AP[i].C_24G in Freq:
                    if F==20:
                        #Maximum number of 106-tone RUs
                        max_num_of_RU=2
                    if F==40:
                        max_num_of_RU = 4
                    if F==80:
                        max_num_of_RU = 8
                    if F==160:
                        max_num_of_RU = 16
                    if F==320:
                        max_num_of_RU = 32

                    Max_transmissions=Num_of_antennas*max_num_of_RU
                    # b=AP[i].Num_Of_STAs//Max_transmissions #Quotient,
                    # a=AP[i].Num_Of_STAs%Max_transmissions #Remainder
                    exec('numSTAs=len(AP[i].ID_Of_STAs_{})'.format(B))
                    NumSTAs=locals()['numSTAs']
                    # print(NumSTAs,len(AP[i].ID_Of_STAs_24G))
                    # input()
                    round=int(np.ceil(NumSTAs/Max_transmissions))
                    # print('i',i)
                    # print('B',B)
                    # print('round:',round)
                    exec('AP[i].Round_{}=round'.format(B))


                    for r in range(round):
                        exec('AP[i].Group_{}.append([])'.format(B))

                    for j in range(NumSTAs):
                        r = j % round
                        exec('AP[i].Group_{}[r].append(AP[i].ID_Of_STAs_{}[j])'.format(B,B))
                        # print(AP[i].Group_24G[r])
                        # input()
                    # input()

                    for g in range(round):
                        exec('len1=len(AP[i].Group_{}[g])'.format(B))
                        len_temp=locals()['len1']
                        b=len_temp//Num_of_antennas#Quotient
                        a =len_temp%Num_of_antennas# Remainder
                        if a==0:
                            # print('a=0')
                            for j in range(len_temp):
                                temp_rem=j%max_num_of_RU
                                #exec('STA[AP[i].Group_{}[g][j]].NumOfRU=RU{}_{}[temp_rem]'.format(B,F,max_num_of_RU))
                                exec('STA[AP[i].Group_{}[g][j]].NumOfRU_{}=RU{}_{}[temp_rem]'.format(B, B, F, max_num_of_RU))
                        if a!=0:
                            SS=[]
                            # print('a!=0')
                            for k in range(Num_of_antennas):
                                SS.append([])
                            for j in range(len_temp):
                                temp_rem=j%Num_of_antennas
                                exec('SS[temp_rem].append(AP[i].Group_{}[g][j])'.format(B))
                                # SS[temp_rem].append(AP[i].Group_24G[g][j])
                            len_ss=np.zeros(Num_of_antennas)
                            for j in range(Num_of_antennas):
                                len_ss[j]=len(SS[j])
                            max_RU=max(len_ss)


                            for Max_RU in range(1,max_num_of_RU+1):
                                if max_RU==Max_RU:
                                    for j in range(Num_of_antennas):
                                        if len(SS[j]) != 0:
                                            for k in range(len(SS[j])):
                                                # exec('STA[AP[i].ID_Of_STAs[SS[j][k]]].NumOfRU = RU20_{}[k]'.format(Max_RU))
                                                exec('STA[SS[j][k]].NumOfRU_{} = RU{}_{}[k]'.format(B,F,Max_RU))
                            # AP[i].Group=AP[i].Group+SS
                           # input()
                        # Calculate the data rate
                    # if AP[i].C in C_2dot4G:
                    # input()
                    # f=fc_2dot4G
                    # if AP[i].C in C_5GI:
                    #     f=fc_5GI
                    #
                    # for j in range(AP[i].Num_Of_STAs):
                    #     P_RX=AP[i].P-PL(AP[i].Dis_Between_AP_and_STAs[j],f)
                    exec('lenSTAs=len(AP[i].ID_Of_STAs_{})'.format(B))
                    LenSTAs=locals()['lenSTAs']
                    for j in range(LenSTAs):
                        # print(AP[i].Dis_Between_AP_and_STAs)
                        # print(AP[i].ID_Of_STAs)
                        # print(len(AP[i].Dis_Between_AP_and_STAs))
                        # print(len(AP[i].ID_Of_STAs))

                        exec('IDSTA=AP[i].ID_Of_STAs_{}[j]'.format(B))
                        idSTA=locals()['IDSTA']
                        index_sta=AP[i].ID_Of_STAs.index(idSTA)

                        exec('STA[AP[i].ID_Of_STAs_{}[j]].Rx_power_dBm_{}=AP[i].P_{}-PL(AP[i].Dis_Between_AP_and_STAs[index_sta],f)'.format(B,B,B))#in dBm

                        exec('STA[AP[i].ID_Of_STAs_{}[j]].Rx_power_W_{}=10**(STA[AP[i].ID_Of_STAs_{}[j]].Rx_power_dBm_{}/10)/1000'.format(B,B,B,B)) #in W

                        exec('temp=STA[AP[i].ID_Of_STAs_{}[j]].Rx_power_W_{}/(STA[AP[i].ID_Of_STAs_{}[j]].Total_Interference_{}+P_noise_W)'.format(B,B,B,B))
                        XX=locals()['temp']

                        exec('STA[AP[i].ID_Of_STAs_{}[j]].SINR_{}=10*np.log10(XX)'.format(B,B))
                        # STA[AP[i].ID_Of_STAs[j]].DataRate_DL=STA[AP[i].ID_Of_STAs[j]].NumOfRU*Wsc*np.log2(1+STA[AP[i].ID_Of_STAs[j]].SINR)/1000000
                        exec('STA[AP[i].ID_Of_STAs_{}[j]].DataRate_DL_{}=STA[AP[i].ID_Of_STAs_{}[j]].NumOfRU_{}*Wsc*Get_pbs_per_Hz(STA[AP[i].ID_Of_STAs_{}[j]].SINR_{})/1000/1000'.format(B,B,B,B,B,B))
                        # print('test---')
                        # input()
                    exec('len_g=len(AP[i].Group_{})'.format(B))
                    Len_G=locals()['len_g']
                    for g in range(Len_G):
                        # max_dis = 0
                        # for sta in range(len(AP[i].Group[g])):
                        #     if STA[AP[i].Group[g][sta]].Dis_to_AP>max_dis:
                        #         # print('a>b', STA[AP[i].Group[g][sta]].Dis_to_AP, max_dis)
                        #         max_dis=STA[AP[i].Group[g][sta]].Dis_to_AP
                                # print('max_dis=', max_dis)

                        exec('lengg=len(AP[i].Group_{}[g])'.format(B))
                        LenGG=locals()['lengg']
                        for sta in range(LenGG):
                            exec('POWER=AP[i].P_{}'.format(B))
                            Power=locals()['POWER']

                            exec('DIS=STA[AP[i].Group_{}[g][sta]].Dis_to_AP'.format(B))
                            Diss=locals()['DIS']

                            exec('Rx_power_dBmm = AP[i].P_{} - PL(STA[AP[i].Group_{}[g][sta]].Dis_to_AP, f)'.format(B,B))
                            Rx_power_dBm=locals()['Rx_power_dBmm']

                            # Rx_power_dBm = AP[i].P - PL(STA[AP[i].Group[g][sta]].Dis_to_AP, f)  # in dBm

                            Rx_power_W = 10 ** (Rx_power_dBm / 10) / 1000  # in W

                            exec('TOTAL_Int=AP[i].Total_Interference_{}'.format(B))
                            total_ints=locals()['TOTAL_Int']

                            exec('X_temp = Rx_power_W / (AP[i].Total_Interference_{} + P_noise_W)'.format(B))
                            X=locals()['X_temp']
                            # X = Rx_power_W / (AP[i].Total_Interference + P_noise_W)
                            # print()
                            # print('Total interference:', total_ints)
                            #
                            # print('X=',X)
                            SINR = 10 * np.log10(X)

                            if SINR <0:
                                print("SINR<0!!")
                                print('Debug error:')
                                print('SINR:',SINR)
                                print('Band:',B)
                                print('TX power dBm:', Power)
                                print('Rx_power_dBm:',Rx_power_dBm)
                                print('Rx_power_w=',Rx_power_W)
                                print('Total interference:',total_ints)
                                print('Distance:',Diss)


                                # print (SINR, B)
                                # input()

                            exec('AP[i].SINR_{}.append(SINR)'.format(B))
                            # AP[i].SINR.append(SINR)
                            exec('STA[AP[i].Group_{}[g][sta]].SINR_UP_{}=SINR'.format(B,B))
                            # STA[AP[i].Group[g][sta]].SINR_UP=SINR
                            # STA[AP[i].ID_Of_STAs[j]].DataRate_DL=STA[AP[i].ID_Of_STAs[j]].NumOfRU*Wsc*np.log2(1+STA[AP[i].ID_Of_STAs[j]].SINR)/1000000
                            exec('STA[AP[i].Group_{}[g][sta]].DataRate_UL_{} = STA[AP[i].Group_{}[g][sta]].NumOfRU_{} * Wsc * Get_pbs_per_Hz(SINR) / 1000 / 1000'.format(B,B,B,B))
                            # STA[AP[i].Group[g][sta]].DataRate_UL = STA[AP[i].Group[g][sta]].NumOfRU * Wsc * Get_pbs_per_Hz(SINR) / 1000 / 1000


    # input()

    for i in range(len(AP)):
        # print('i=',i)
        ttt=[]
        nnn=[]
        for B in Band:
            if B=='24G':
                T_SIFS = 10 * 10 ** (-6)
            if B=='5G1':
                T_SIFS = 16 * 10 ** (-6)
            if B=='5G2':
                T_SIFS = 16 * 10 ** (-6)
            if B=='6G':
                T_SIFS = 16 * 10 ** (-6)
            T_DL = T_DIFS + T_backoff + T_dl + T_SIFS + T_OFDMA
            T_UL = T_DIFS + T_backoff + T_TF + 2 * T_SIFS + T_ul + T_MBA

            exec('Round_time_temp = AP[i].Round_{}*(T_UL+T_DL)'.format(B))
            Round_time=locals()['Round_time_temp']
            # print('Round_time',Round_time)
            # ttt.append(Round_time)
            # input()
            # Round_time = AP[i].Round*(T_UL+T_DL)
            exec('lenNeighbor1=len(AP[i].NeiInt_{})'.format(B))
            lenNeighbor=locals()['lenNeighbor1']
            # nnn.append(lenNeighbor)
            # print('band=',B)
            # print('num of neighbor',lenNeighbor)
            # print('len=',len(AP[i].NeiInt_24G))
            # print('AP neighbor list=', AP[i].NeiInt_24G)
            if lenNeighbor>0:
                for j in range(lenNeighbor):
                    # print('j=',j)
                    # print('AP neighbor list=',AP[i].NeiInt_24G)
                    #
                    # print('round of neighbor=',AP[AP[i].NeiInt_24G[j]].Round_24G*(T_UL+T_DL))
                    # exec('Round_time = Round_time+AP[AP[i].NeiInt_{}[j]].Round_{}*(T_UL+T_DL)'.format(B,B))
                    # print(Round_time)
                    # input()

                    if B=='24G':
                        Round_time = Round_time + AP[AP[i].NeiInt_24G[j]].Round_24G * (T_UL + T_DL)
                    if B=='5G1':
                        Round_time = Round_time + AP[AP[i].NeiInt_5G1[j]].Round_5G1 * (T_UL + T_DL)
                    if B=='5G2':
                        Round_time = Round_time + AP[AP[i].NeiInt_5G2[j]].Round_5G2 * (T_UL + T_DL)
                    if B=='6G':
                        Round_time = Round_time + AP[AP[i].NeiInt_6G[j]].Round_6G * (T_UL + T_DL)
                    # print('non exec')
                    # print(Round_time)
                    # input()



                    # Round_time = Round_time+AP[AP[i].NeiInt[j]].Round*(T_UL+T_DL)
            # if lenNeighbor==0:
            #     exec('Round_time = Round_time+AP[AP[i].NeiInt_{}[j]].Round_{}*(T_UL+T_DL)'.format(B, B))
            exec('AP[i].Total_time_{}=Round_time'.format(B))
        # print(AP[AP[i].NeiInt_24G[1]].Round_24G*(T_UL+T_DL))
        # print(AP[i].C_24G)
        # print(AP[i].C_5G1)
        # print(AP[i].C_5G2)
        # print(AP[i].C_6G)
        # print(AP[i].Round_24G)
        # print(AP[i].Round_5G1)
        # print(AP[i].Round_5G2)
        # print(AP[i].Round_6G)
        # print(AP[i].NeiInt_24G)
        # print(AP[i].NeiInt_5G1)
        # print(AP[i].NeiInt_5G2)
        # print(AP[i].NeiInt_6G)
        # print(AP[i].Total_time_24G)
        # print(AP[i].Total_time_5G1)
        # print(AP[i].Total_time_5G2)
        # print(AP[i].Total_time_6G)
        # # print(ttt)
        # # print(nnn)
        # # print('ttt')
        # input()


    # count=0
    global MaxThroughput
    MaxThroughput = 0
    global MinThroughput
    MinThroughput = np.inf
    for i in range(len(AP)):
        # print(i)
        for B in Band:

            exec('len_temp10=len(AP[i].ID_Of_STAs_{})'.format(B))
            numStas=locals()['len_temp10']
            for j in range(numStas):
                # count=count+1
                exec('idOfsta=AP[i].ID_Of_STAs_{}[j]'.format(B))
                idd_sta=locals()['idOfsta']

                # T_SIFS = 10 * 10 ** (-6)


                # print(AP[i].Total_time_24G)
                # input()

                exec('debug_t=AP[i].Total_time_{}'.format(B))
                debug_tt=locals()['debug_t']
                if debug_tt==0:
                    print("Total_time = 0!!")
                    print("B=",B)
                    print("AP No.:",i)
                    print('There is a error!')
                    input()

                exec('STA[idd_sta].Throughput_{}=(STA[idd_sta].DataRate_DL_{}*T_dl+STA[idd_sta].DataRate_UL_{}*T_ul)/AP[i].Total_time_{}'.format(B,B,B,B))


    idd_sta=0
    for idd_sta in range(len(STA)):
        STA[idd_sta].Throughput=STA[idd_sta].Throughput_24G+STA[idd_sta].Throughput_5G1+STA[idd_sta].Throughput_5G2+STA[idd_sta].Throughput_6G
        if STA[idd_sta].Throughput==0:
            print("Throughput=0!!")
            print('STA =',idd_sta)
            print('AP =',STA[idd_sta].IDOfAP)
            print('DL SINR =',STA[idd_sta].SINR)
            print('UL SINR =', STA[idd_sta].SINR_UP)
            print('Throughput error!')
            # Plot_fig()
            # input()
        # else:
        #     print(count)
        Throughput_of_STAs.append(STA[idd_sta].Throughput)

        if STA[idd_sta].Throughput>MaxThroughput:
            MaxThroughput=STA[idd_sta].Throughput
            APNo=STA[idd_sta].IDOfAP
            idofsta=idd_sta

            ChannelNo24G = AP[STA[idd_sta].IDOfAP].C_24G
            ChannelNo5G1 = AP[STA[idd_sta].IDOfAP].C_5G1
            ChannelNo5G2 = AP[STA[idd_sta].IDOfAP].C_5G2
            ChannelNo6G = AP[STA[idd_sta].IDOfAP].C_6G
            IDOfstas_24G=AP[STA[idd_sta].IDOfAP].ID_Of_STAs_24G
            IDOfstas_5G1 = AP[STA[idd_sta].IDOfAP].ID_Of_STAs_5G1
            IDOfstas_5G2 = AP[STA[idd_sta].IDOfAP].ID_Of_STAs_5G2
            IDOfstas_6G = AP[STA[idd_sta].IDOfAP].ID_Of_STAs_6G

        if STA[idd_sta].Throughput<MinThroughput:

            MinThroughput=STA[idd_sta].Throughput
            APNo1 = STA[idd_sta].IDOfAP
            idofsta1 = idd_sta
            ChannelNo24G1 = AP[STA[idd_sta].IDOfAP].C_24G
            ChannelNo5G11= AP[STA[idd_sta].IDOfAP].C_5G1
            ChannelNo5G21 = AP[STA[idd_sta].IDOfAP].C_5G2
            ChannelNo6G1 = AP[STA[idd_sta].IDOfAP].C_6G
            IDOfstas_24G1 = AP[STA[idd_sta].IDOfAP].ID_Of_STAs_24G
            IDOfstas_5G11 = AP[STA[idd_sta].IDOfAP].ID_Of_STAs_5G1
            IDOfstas_5G21 = AP[STA[idd_sta].IDOfAP].ID_Of_STAs_5G2
            IDOfstas_6G1 = AP[STA[idd_sta].IDOfAP].ID_Of_STAs_6G

    # print('MaxThroughput',MaxThroughput)
    # print('APNo',APNo)
    # print('ChannelNo24G',ChannelNo24G)
    # print('ChannelNo5G1',ChannelNo5G1)
    # print('ChannelNo5G2',ChannelNo5G2)
    # print('ChannelNo6G',ChannelNo6G)
    # print('IDOfstas_24G',IDOfstas_24G)
    # print('IDOfstas_5G1', IDOfstas_5G1)
    # print('IDOfstas_5G2', IDOfstas_5G2)
    # print('IDOfstas_6G', IDOfstas_6G)
    # print('idofsta',idofsta)
    # print('---------')
    # print('MinThroughput',MinThroughput)
    # print('APNo1',APNo1)
    # print('ChannelNo24G1',ChannelNo24G1)
    # print('ChannelNo5G11',ChannelNo5G11)
    # print('ChannelNo5G21',ChannelNo5G21)
    # print('ChannelNo6G1',ChannelNo6G1)
    # print('IDOfstas_24G1',IDOfstas_24G1)
    # print('IDOfstas_5G11', IDOfstas_5G11)
    # print('IDOfstas_5G21', IDOfstas_5G21)
    # print('IDOfstas_6G1', IDOfstas_6G1)
    # print('idofsta1',idofsta1)

                        #Calculate the throughput of STAs
                        # Ti=(T_DL+T_UL)*len(AP[i].Group)
                        # STA[AP[i].Group[g][sta]].Throughput=(STA[AP[i].Group[g][sta]].DataRate_UL*T_ul+STA[AP[i].Group[g][sta]].DataRate_DL*T_dl)/(Ti*)

    #-------------------------------------------------------

    return

# This function is for APCoordination for RU assignment
# The bigger RU is assigned with the farther station
def RUassignment_APCoordination():
    #Determine the channel width
    # print('begin RU assignment')
    # input()

    # for i in range(len(AP)):
    #     print('AP_24G_C=',AP[i].C_24G)
    #     print('AP_5G1_C=', AP[i].C_5G1)
    #     print('AP_5G2_C=', AP[i].C_5G2)
    #     print('AP_6G_C=', AP[i].C_6G)
    #     input()
    # 确保频率表存在（对主进程或并行子进程都安全）
    define_frequencies()
    define_beam_variables()


    for i in range(len(AP)):
        
        # print(AP[i].Num_Of_STAs)
        
        # input()
        
        if AP[i].C_24G in C20M:
            AP[i].NumRUs_24G = 9
            # AP[i].RUperUser_24G = AP[i].NumRUs_24G / AP[i].Num_Of_STAs
        if AP[i].C_24G in C40M:
            AP[i].NumRUs_24G = 18
            # AP[i].RUperUser_24G = AP[i].NumRUs_24G / AP[i].Num_Of_STAs
        if AP[i].C_24G in C80M:
            AP[i].NumRUs_24G = 36
            # AP[i].RUperUser_24G = AP[i].NumRUs_24G / AP[i].Num_Of_STAs
        if AP[i].C_24G in C160M:
            AP[i].NumRUs_24G = 72
            # AP[i].RUperUser_24G = AP[i].NumRUs_24G / AP[i].Num_Of_STAs
        if AP[i].C_24G in C320M:
            AP[i].NumRUs_24G = 144
            # AP[i].RUperUser_24G = AP[i].NumRUs_24G / AP[i].Num_Of_STAs

        if AP[i].C_5G1 in C20M:
            AP[i].NumRUs_5G1 = 9
            # AP[i].RUperUser_5G1 = AP[i].NumRUs_5G1 / AP[i].Num_Of_STAs
        if AP[i].C_5G1 in C40M:
            AP[i].NumRUs_5G1 = 18
            # AP[i].RUperUser_5G1 = AP[i].NumRUs_5G1 / AP[i].Num_Of_STAs
        if AP[i].C_5G1 in C80M:
            AP[i].NumRUs_5G1 = 36
            # AP[i].RUperUser_5G1 = AP[i].NumRUs_5G1 / AP[i].Num_Of_STAs
        if AP[i].C_5G1 in C160M:
            AP[i].NumRUs_5G1 = 72
            # AP[i].RUperUser_24G = AP[i].NumRUs_24G / AP[i].Num_Of_STAs
        if AP[i].C_5G1 in C320M:
            AP[i].NumRUs_5G1 = 144
            # AP[i].RUperUser_24G = AP[i].NumRUs_24G / AP[i].Num_Of_STAs

        if AP[i].C_5G2 in C20M:
            AP[i].NumRUs_5G2 = 9
            # AP[i].RUperUser_5G2 = AP[i].NumRUs_5G2 / AP[i].Num_Of_STAs
        if AP[i].C_5G2 in C40M:
            AP[i].NumRUs_5G2 = 18
            # AP[i].RUperUser_5G2 = AP[i].NumRUs_5G2 / AP[i].Num_Of_STAs
        if AP[i].C_5G2 in C80M:
            AP[i].NumRUs_5G2 = 36
            # AP[i].RUperUser_5G2 = AP[i].NumRUs_5G2 / AP[i].Num_Of_STAs
        if AP[i].C_5G2 in C160M:
            AP[i].NumRUs_5G2 = 72
            # AP[i].RUperUser_24G = AP[i].NumRUs_24G / AP[i].Num_Of_STAs
        if AP[i].C_5G2 in C320M:
            AP[i].NumRUs_5G2 = 144
            # AP[i].RUperUser_24G = AP[i].NumRUs_24G / AP[i].Num_Of_STAs

        if AP[i].C_6G in C20M:
            AP[i].NumRUs_6G = 9
            # AP[i].RUperUser_6G = AP[i].NumRUs_6G / AP[i].Num_Of_STAs
        if AP[i].C_6G in C40M:
            AP[i].NumRUs_6G = 18
            # AP[i].RUperUser_6G = AP[i].NumRUs_6G / AP[i].Num_Of_STAs
        if AP[i].C_6G in C80M:
            AP[i].NumRUs_6G = 36
            # AP[i].RUperUser_6G = AP[i].NumRUs_6G / AP[i].Num_Of_STAs
        if AP[i].C_6G in C160M:
            AP[i].NumRUs_6G = 72
            # AP[i].RUperUser_24G = AP[i].NumRUs_24G / AP[i].Num_Of_STAs
        if AP[i].C_6G in C320M:
            AP[i].NumRUs_6G = 144
            # AP[i].RUperUser_24G = AP[i].NumRUs_24G / AP[i].Num_Of_STAs

    RUlist24G=[]
    RUlist5G1 = []
    RUlist5G2 = []
    RUlist6G = []
    for i in range(len(AP)):
        RUlist24G.append(AP[i].NumRUs_24G)
        RUlist5G1.append(AP[i].NumRUs_5G1)
        RUlist5G2.append(AP[i].NumRUs_5G2)
        RUlist6G.append(AP[i].NumRUs_6G)
        # print("Current RU:")
        # print('i=', i)
        # print(AP[i].NumRUs_6G)
    # print(RUlist24G)
    # print(RUlist5G1)
    # print(RUlist5G2)
    # print(RUlist6G)
    # input()
    if LocalOptimizerMethod==0:
        Flag24G=0
        Flag5G1=0
        Flag5G2 = 0
        Flag6G = 0
        while(1):
            for i in range(len(AP)):
                # if AP[i].C_24G in C20M:
                #     AP[i].NumRUs_24G=9
                AP[i].RUperUser_24G=AP[i].NumRUs_24G/AP[i].Num_Of_STAs
                
                # print(AP[i].Num_Of_STAs)
                
                # input()
                # if AP[i].C_24G in C40M:
                #     AP[i].NumRUs_24G=18
                # AP[i].RUperUser_24G = AP[i].NumRUs_24G / AP[i].Num_Of_STAs
                # if AP[i].C_24G in C80M:
                #     AP[i].NumRUs_24G=36
                # AP[i].RUperUser_24G = AP[i].NumRUs_24G / AP[i].Num_Of_STAs
    
                # if AP[i].C_5G1 in C20M:
                #     AP[i].NumRUs_5G1=9
                AP[i].RUperUser_5G1 = AP[i].NumRUs_5G1 / AP[i].Num_Of_STAs
                # if AP[i].C_5G1 in C40M:
                #     AP[i].NumRUs_5G1=18
                # AP[i].RUperUser_5G1 = AP[i].NumRUs_5G1 / AP[i].Num_Of_STAs
                # if AP[i].C_5G1 in C80M:
                #     AP[i].NumRUs_5G1=36
                # AP[i].RUperUser_5G1 = AP[i].NumRUs_5G1 / AP[i].Num_Of_STAs
    
                # if AP[i].C_5G2 in C20M:
                #     AP[i].NumRUs_5G2=9
                AP[i].RUperUser_5G2 = AP[i].NumRUs_5G2 / AP[i].Num_Of_STAs
                # if AP[i].C_5G2 in C40M:
                #     AP[i].NumRUs_5G2=18
                # AP[i].RUperUser_5G2 = AP[i].NumRUs_5G2 / AP[i].Num_Of_STAs
                # if AP[i].C_5G2 in C80M:
                #     AP[i].NumRUs_5G2=36
                # AP[i].RUperUser_5G2 = AP[i].NumRUs_5G2 / AP[i].Num_Of_STAs
    
                # if AP[i].C_6G in C20M:
                #     AP[i].NumRUs_6G=9
                AP[i].RUperUser_6G = AP[i].NumRUs_6G / AP[i].Num_Of_STAs
                # if AP[i].C_6G in C40M:
                #     AP[i].NumRUs_6G=18
                # AP[i].RUperUser_6G = AP[i].NumRUs_6G / AP[i].Num_Of_STAs
                # if AP[i].C_6G in C80M:
                #     AP[i].NumRUs_6G=36
                # AP[i].RUperUser_6G = AP[i].NumRUs_6G / AP[i].Num_Of_STAs
    
                # print(AP[i].NumRUs_6G)
                # print('i=',i)
                # print('test')
                # input()
    
            MaxRU_24G=0
            MaxRU_5G1 = 0
            MaxRU_5G2 = 0
            MaxRU_6G = 0
            MinRU_24G=np.inf
            MinRU_5G1 = np.inf
            MinRU_5G2 = np.inf
            MinRU_6G = np.inf
            Mini_24G=np.inf
            Mini_5G1 = np.inf
            Mini_5G2 = np.inf
            Mini_6G = np.inf
            Maxi_24G=np.inf
            Maxi_5G1 = np.inf
            Maxi_5G2 = np.inf
            Maxi_6G = np.inf
            for i in range(len(AP)):
    
                if AP[i].RUperUser_24G > MaxRU_24G:
                    MaxRU_24G=AP[i].RUperUser_24G
                    Maxi_24G=i
                if AP[i].RUperUser_5G1 > MaxRU_5G1:
                    MaxRU_5G1=AP[i].RUperUser_5G1
                    Maxi_5G1=i
                if AP[i].RUperUser_5G2 > MaxRU_5G2:
                    MaxRU_5G2=AP[i].RUperUser_5G2
                    Maxi_5G2=i
                if AP[i].RUperUser_6G > MaxRU_6G:
                    MaxRU_6G=AP[i].RUperUser_6G
                    Maxi_6G=i
    
                if AP[i].RUperUser_24G < MinRU_24G:
                    MinRU_24G=AP[i].RUperUser_24G
                    Mini_24G=i
                if AP[i].RUperUser_5G1 < MinRU_5G1:
                    MinRU_5G1=AP[i].RUperUser_5G1
                    Mini_5G1=i
                if AP[i].RUperUser_5G2 < MinRU_5G2:
                    MinRU_5G2=AP[i].RUperUser_5G2
                    Mini_5G2=i
                if AP[i].RUperUser_6G < MinRU_6G:
                    MinRU_6G=AP[i].RUperUser_6G
                    Mini_6G=i
            # print("Max24G",Maxi_24G)
            # print("Max5G1", Maxi_5G1)
            # print("Max5G2", Maxi_5G2)
            # print("Max6G", Maxi_6G)
            #
            # print("Min24G", Mini_24G)
            # print("Min5G1", Mini_5G1)
            # print("Min5G2", Mini_5G2)
            # print("Min6G", Mini_6G)
            #
            # print("Max24G_RU",AP[Maxi_24G].NumRUs_24G)
            # print("Max5G1_RU", AP[Maxi_5G1].NumRUs_5G1)
            # print("Max5G2_RU", AP[Maxi_5G2].NumRUs_5G2)
            # print("Max6G_RU", AP[Maxi_6G].NumRUs_6G)
            #
            # print("Min24G_RU", AP[Mini_24G].NumRUs_24G)
            # print("Min5G1_RU", AP[Mini_5G1].NumRUs_5G1)
            # print("Min5G2_RU", AP[Mini_5G2].NumRUs_5G2)
            # print("Min6G_RU", AP[Mini_6G].NumRUs_6G)
    
    
    
            if AP[Maxi_24G].NumRUs_24G-AP[Mini_24G].NumRUs_24G>=2:
                AP[Maxi_24G].NumRUs_24G=AP[Maxi_24G].NumRUs_24G-1
                AP[Mini_24G].NumRUs_24G=AP[Mini_24G].NumRUs_24G+1
            else:
                Flag24G=1
            if AP[Maxi_6G].NumRUs_6G-AP[Mini_6G].NumRUs_6G>=2:
                AP[Maxi_6G].NumRUs_6G=AP[Maxi_6G].NumRUs_6G-1
                AP[Mini_6G].NumRUs_6G=AP[Mini_6G].NumRUs_6G+1
            else:
                Flag6G=1
            if AP[Maxi_5G1].NumRUs_5G1-AP[Mini_5G1].NumRUs_5G1>=2:
                AP[Maxi_5G1].NumRUs_5G1=AP[Maxi_5G1].NumRUs_5G1-1
                AP[Mini_5G1].NumRUs_5G1=AP[Mini_5G1].NumRUs_5G1+1
            else:
                Flag5G1=1
            if AP[Maxi_5G2].NumRUs_5G2-AP[Mini_5G2].NumRUs_5G2>=2:
                AP[Maxi_5G2].NumRUs_5G2=AP[Maxi_5G2].NumRUs_5G2-1
                AP[Mini_5G2].NumRUs_5G2=AP[Mini_5G2].NumRUs_5G2+1
            else:
                Flag5G2=1
    
    
    
            # print('AP coordination:')
            #
            # print("Max6G_RU", AP[Maxi_6G].NumRUs_6G)
            # print("Min6G_RU", AP[Mini_6G].NumRUs_6G)
            # input()
            if Flag24G==1 and Flag5G1==1  and Flag5G2==1 and Flag6G==1:
                break
    
        RUlist24G = []
        RUlist5G1 = []
        RUlist5G2 = []
        RUlist6G = []
        for i in range(len(AP)):
            RUlist24G.append(AP[i].NumRUs_24G)
            RUlist5G1.append(AP[i].NumRUs_5G1)
            RUlist5G2.append(AP[i].NumRUs_5G2)
            RUlist6G.append(AP[i].NumRUs_6G)
            # print("Current RU:")
            # print('i=', i)
            # print(AP[i].NumRUs_6G)
        # print(RUlist24G)
        # print(RUlist5G1)
        # print(RUlist5G2)
        # print(RUlist6G)
        # input()








        #
        # AP[i].RUperUser_24G
        # AP[i].RUperUser_5G1
        # AP[i].RUperUser_5G2
        # AP[i].RUperUser_6G



    for i in range(len(AP)):#2.4G
        # print('number of AP:',i)
        for B in Band:
            if Band.index(B)==0:
                FF=[20,40]
                f=fc_2dot4G
            if Band.index(B)==1:
                FF=[20,40,80,160]
                f=fc_5GI
            if Band.index(B)==2:
                FF=[20,40,80]
                f=fc_5GII
            if Band.index(B)==3:
                FF=[20,40,80,160,320]
                f=fc_6G

            for F in FF:
                # Fre = []
                # exec('Fre=Frequence{}_{}'.format(F,B))
                # Freq=locals()['Fre']
                
                # exec('chh=AP[i].C_{}'.format(B))
                # ch=locals()['chh']
                
                # 获取全局变量 FrequenceF_B
                Freq = globals()[f'Frequence{F}_{B}']  # 注意 globals() 是字典
            
                # 获取 AP[i] 的属性 C_B
                ch = getattr(AP[i], f'C_{B}')
                # print(ch)
                # input()
                if ch in Freq:
                # if AP[i].C_24G in Freq:
                    if F==20:
                        #Maximum number of 106-tone RUs 即106RU的个数，原始带宽
                        max_num_of_RU=2
                    if F==40:
                        max_num_of_RU = 4
                    if F==80:
                        max_num_of_RU = 8
                    if F==160:
                        max_num_of_RU = 16
                    if F==320:
                        max_num_of_RU = 32

                    #Determine the Maximum number of 106-tone RUs
                    # exec('max_num_of_RUU=AP[i].NumRUs_{}'.format(B))
                    # max_num_of_RUuu = locals()['max_num_of_RUU']
                    
                    max_num_of_RUuu = getattr(AP[i], f'NumRUs_{B}')
                    
                    # max_num_of_RUuu=15
                    max_num_of_RU=max_num_of_RUuu//4  #经过AP协调之后，每个AP的106RU的个数，可能是原始带宽+来自其他AP的RU
                    RU_rem=max_num_of_RUuu%4

                    # print(max_num_of_RUuu)
                    # print(max_num_of_RU)
                    # print(RU_rem)
                    # input()

                    RUset_Full=[]
                    RUset_Part =[]
                    for ru in range(0,max_num_of_RU):
                        RUset_Full.append(4*26+2)
                    if RU_rem !=0:
                        for j in range(0,RU_rem):
                            # print('j=',j)
                            # print('max_num_of_ru=',max_num_of_RU)
                            # print('AP_c=',ch)
                            ind=j%max_num_of_RU

                            RUset_Full[ind]=RUset_Full[ind]+26

                        # for rem_ru in range(0,RU_rem):
                        #     print('RUset_full')
                        #     print(RUset_Full)
                        #     RUset_Full[rem_ru]=RUset_Full[rem_ru] + 26
                        #     print('rem_ru', rem_ru)
                    # print('======')
                    # print('RUset_full',RUset_Full)
                    # print('max_num_of_RU',max_num_of_RU)
                    # print('======')
                    # input()


                    # print(B)
                    # print('max_num_of_RU',max_num_of_RU)
                    # print('max_num_of_RU', int(max_num_of_RU))
                    # input()
                    # b=AP[i].Num_Of_STAs//Max_transmissions #Quotient,
                    # a=AP[i].Num_Of_STAs%Max_transmissions #Remainder




#Beamforming revised
                    Max_transmissions=Num_of_antennas*max_num_of_RU
                    #Max_transmissions=max_num_of_RU #波束的个数
                    
                    
                        
                    
                    # print("Beamforming nmus: ",max_num_of_RU)
                    
                    # b=AP[i].Num_Of_STAs//Max_transmissions #Quotient,
                    # a=AP[i].Num_Of_STAs%Max_transmissions #Remainder
                    # exec('numSTAs=len(AP[i].ID_Of_STAs_{})'.format(B))
                    # NumSTAs=locals()['numSTAs']
                    NumSTAs = len(getattr(AP[i], f'ID_Of_STAs_{B}'))
                    
                    # print(NumSTAs,len(AP[i].ID_Of_STAs_24G))
                    # input()
                    round=int(np.ceil(NumSTAs/Max_transmissions))
                    # print('i',i)
                    # print('B',B)
                    # print('round:',round)
                    # exec('AP[i].Round_{}=round'.format(B))
                    setattr(AP[i], f'Round_{B}', round)


                    for r in range(round):
                        # exec('AP[i].Group_{}.append([])'.format(B))
                        getattr(AP[i], f'Group_{B}').append([])

                    # The stations should be ordered
                    for j in range(NumSTAs):
                        r = j % round
                        # exec('AP[i].Group_{}[r].append(AP[i].ID_Of_STAs_{}[j])'.format(B,B))
                        getattr(AP[i], f'Group_{B}')[r].append(getattr(AP[i], f'ID_Of_STAs_{B}')[j])
                        # print(AP[i].Group_24G[r])
                        # input()
                    # input()

                    for g in range(round):
                        # Sort the stations according to the distances

                        # list = [34, 23, 77, 54, 99, 10]
                        # sorted_id = sorted(range(len(list)), key=lambda x: list[x], reverse=True)
                        # print(sorted_id)



                        # exec('len1=len(AP[i].Group_{}[g])'.format(B))
                        # len_temp=locals()['len1']
                        len_temp = len(getattr(AP[i], f'Group_{B}')[g])
                        b=len_temp//Num_of_antennas#Quotient
                        a =len_temp%Num_of_antennas# Remainder
                        if a==0:
                            a_temp=0
                            # print('a=0')

                            # exec('listt=AP[i].Group_{}[g]'.format(B))
                            # list_temp = locals()['listt']
                            # # print('list_temp',list_temp)
                            # Diss_temp=[]
                            # for d in range(0,len(list_temp)):
                            #     diss=((AP[i].x-STA[list_temp[d]].x)**2+(AP[i].y-STA[list_temp[d]].y)**2)**(0.5)
                            #     # print(list_temp[d])
                            #     # input()
                            #     Diss_temp.append(diss)
                            # # print('dis_temp',Diss_temp)
                            #
                            #
                            #
                            #
                            # for jj in range(len_temp):
                            #     temp_rem=jj%max_num_of_RU
                            #     #exec('STA[AP[i].Group_{}[g][j]].NumOfRU=RU{}_{}[temp_rem]'.format(B,F,max_num_of_RU))
                            #     # exec('STA[AP[i].Group_{}[g][j]].NumOfRU_{}=RU{}_{}[temp_rem]'.format(B, B, F, max_num_of_RU))
                            #
                            #     # print('len_temp',len_temp)
                            #     # print('RUset_Full',RUset_Full)
                            #     # print('max_num_of_RU', max_num_of_RU)
                            #     # print('jj',jj)
                            #     # print('temp_rem',temp_rem)
                            #     #
                            #     # print('g',g)
                            #     #
                            #     # exec('group=AP[i].Group_{}[g][jj]'.format(B))
                            #     # group_temp = locals()['group']
                            #     # exec('groupsta=AP[i].Group_{}[g]'.format(B))
                            #     # group_temp_sta = locals()['groupsta']
                            #     # print('group-sta',group_temp)
                            #     # print('group',group_temp_sta)
                            #
                            #
                            #
                            #     # input()
                            #     exec('STA[AP[i].Group_{}[g][jj]].NumOfRU_{}=RUset_Full[temp_rem]'.format(B, B))
                        if a==0 or a >0:

                            # exec('listt=AP[i].Group_{}[g]'.format(B))
                            # list_temp = locals()['listt']
                            # print('list_temp', list_temp)
                            # Diss_temp = []
                            # for d in range(0, len(list_temp)):
                            #     diss = ((AP[i].x - STA[list_temp[d]].x) ** 2 + (
                            #                 AP[i].y - STA[list_temp[d]].y) ** 2) ** (0.5)
                            #     # print(list_temp[d])
                            #     # input()
                            #     Diss_temp.append(diss)
                            # print('dis_temp', Diss_temp)

                            SS=[]
                            # print('a!=0')
                            for k in range(Num_of_antennas):
                                SS.append([])
                            for j in range(len_temp):
                                temp_rem=j%Num_of_antennas
                                # exec('SS[temp_rem].append(AP[i].Group_{}[g][j])'.format(B))
                                SS[temp_rem].append(getattr(AP[i], f'Group_{B}')[g][j])
                                # SS[temp_rem].append(AP[i].Group_24G[g][j])
                                # print('temp_rem',temp_rem)
                                # print('SS[temp]',SS)
                                # input()
                            len_ss=np.zeros(Num_of_antennas)
                            for j in range(Num_of_antennas):
                                len_ss[j]=len(SS[j])
                            max_RU=max(len_ss)

                            # print(SS)




                            for Max_RU in range(1,max_num_of_RU+1):
                                # print('max_RU',max_RU)
                                # print('Max_RU',Max_RU)
                                # input()

                                if max_RU==Max_RU: # Why use this condition? Because we need to determine the RU set!
                                    # print('part_index')
                                    # print(max_RU)
                                    # input()

                                    for j in range(Num_of_antennas):

                                        # Sort the stations for each spatial stream. Bigger RU for farther station
                                        # exec('listt=SS[j]'.format(B))  ##.format(B)
                                        # list_temp = locals()['listt']
                                        
                                        list_temp = SS[j]
                                        # print('list_temp', list_temp)
                                        Diss_temp = []
                                        for d in range(0, len(list_temp)):
                                            
                                            # if STA[list_temp[d]].z==h_sta:
                                                
                                            
                                            #     diss = ((AP[i].x - STA[list_temp[d]].x) ** 2 + (
                                            #             AP[i].y - STA[list_temp[d]].y) ** 2) ** (0.5)
                                            #     # print(list_temp[d])
                                            #     # input()
    
                                            #     #3D distance
                                            #     diss=(diss**2+H_sta**2)**(0.5)
                                                
                                            # else:
                                            diss = ((AP[i].x - STA[list_temp[d]].x) ** 2 + (
                                                        AP[i].y - STA[list_temp[d]].y) ** 2+(AP[i].z - STA[list_temp[d]].z) ** 2) ** (0.5)

                                            Diss_temp.append(diss)
                                        # print('dis_temp', Diss_temp)


                                        sorted_id = sorted(range(len(Diss_temp)), key=lambda x: Diss_temp[x], reverse=True)
                                        # print(sorted_id)
                                        # input()



                                        if len(SS[j]) != 0:

                                            # exec('max_num_of_RUUT=AP[i].NumRUs_{}'.format(B))
                                            # max_num_of_RUuuT = locals()['max_num_of_RUUT']
                                            max_num_of_RUuuT = getattr(AP[i], f'NumRUs_{B}')
                                            # max_num_of_RUuu=15
                                            # Max_RU here is the number of RU subsets
                                            # max_num_of_RU here is the number of 26-tone RUs
                                            max_num_of_RUt = max_num_of_RUuuT // len(SS[j])
                                            RU_remt = max_num_of_RUuuT % len(SS[j])

                                            # print('max_RU',Max_RU)
                                            # print('max_num_of_RUuuT',max_num_of_RUuuT)
                                            # print('max_num_of_RU',max_num_of_RU)
                                            # print('RU_rem',RU_rem)
                                            # print('number of AP i:',i)
                                            # input()

                                            RUset_Part = []
                                            for ru in range(0, len(SS[j])):
                                                RU_element = max_num_of_RUt * 26
                                                if max_num_of_RUt * 26 >= 16 * 4 and max_num_of_RUt * 26 < 16 * 8:
                                                    RU_element = max_num_of_RUt * 26 + 2
                                                if max_num_of_RUt * 26 >= 16 * 8:
                                                    RU_element = max_num_of_RUt * 26 + 4
                                                RUset_Part.append(RU_element)
                                            if RU_remt != 0:
                                                for jj in range(0, RU_remt):
                                                    ind = jj % len(SS[j])
                                                    RUset_Part[ind] = RUset_Part[ind] + 26

                                                # for rem_ru in range(0, RU_rem):
                                                #     RUset_Part[rem_ru] = RUset_Part[rem_ru] + 26

                                            # print('RUset_Part', RUset_Part)
                                            # input()

                                            for k in range(len(SS[j])):
                                                # exec('STA[AP[i].ID_Of_STAs[SS[j][k]]].NumOfRU = RU20_{}[k]'.format(Max_RU))
                                                # print(RUset_Part)
                                                # print(RUlist24G)
                                                # print(RUlist5G1)
                                                # print(RUlist5G2)
                                                # print(RUlist6G)
                                                # input()
                                                # exec('STA[SS[j][k]].NumOfRU_{} = RU{}_{}[k]'.format(B,F,Max_RU))
                                                # print('RUset_Part=',RUset_Part)
                                                # print('k=',k)
                                                # print('j=',j)
                                                # print('ss=',SS)
                                                # print('SS[j][k]=',SS[j][k])
                                                # exec('axx=STA[SS[j][k]].NumOfRU_{} = RUset_Part[k]'.format(B))

                                                # exec('STA[SS[j][k]].NumOfRU_{} = RUset_Part[k]'.format(B))
                                                
                                                # exec('STA[list_temp[sorted_id[k]]].NumOfRU_{} = RUset_Part[k]'.format(B))
                                                setattr(STA[list_temp[sorted_id[k]]], f'NumOfRU_{B}', RUset_Part[k])
                                                # exec('testRU=STA[list_temp[sorted_id[k]]].NumOfRU_{}'.format(B))
                                                # testRUU= locals()['testRU']
                                                # exec('testSTA=list_temp[sorted_id[k]]')
                                                # testSTAa= locals()['testSTA']
                                                # print('testSTAa',testSTAa)
                                                # print('testRUU',testRUU)
                                                # input()

                            # AP[i].Group=AP[i].Group+SS
                           # input()
                        # Calculate the data rate
                    # if AP[i].C in C_2dot4G:
                    # input()
                    # f=fc_2dot4G
                    # if AP[i].C in C_5GI:
                    #     f=fc_5GI
                    #
                    # for j in range(AP[i].Num_Of_STAs):
                    #     P_RX=AP[i].P-PL(AP[i].Dis_Between_AP_and_STAs[j],f)
                    # exec('lenSTAs=len(AP[i].ID_Of_STAs_{})'.format(B))
                    # LenSTAs=locals()['lenSTAs']
                    LenSTAs = len(getattr(AP[i], f'ID_Of_STAs_{B}'))
                    
                    # for j in range(LenSTAs):
                    #     idSTA = getattr(AP[i], f'ID_Of_STAs_{B}')[j]
                    #     index_sta = AP[i].ID_Of_STAs.index(idSTA)
                    
                    #     # dBi, dBi_Rx 已经定义
                    #     # 发射功率转换
                    #     P_mw = 10 ** (getattr(AP[i], f'P_{B}') / 10)
                    #     P_beam_mw = P_mw / Num_of_antennas  # allocate power for each beam
                    #     P_beam_dbm = 10 * np.log10(P_beam_mw)
                    
                    #     # Rx_power_dBm
                    #     print("B =", B)
                    #     print("Trying to access:", f'G_beam{B}_AP')
                    #     print("Available attributes:", dir(AP[i]))
                        
                        
                    #     rx_power_dbm = (
                    #         dBi
                    #         + dBi_Rx
                    #         + G_arr_AP
                    #         + P_beam_dbm
                    #         + getattr(AP[i], f'G_beam{B}_AP')
                    #         - PL(AP[i].Dis_Between_AP_and_STAs[index_sta], f)
                    #     )
                    #     setattr(STA[idSTA], f'Rx_power_dBm_{B}', rx_power_dbm)
                    
                    #     # Rx_power_W
                    #     rx_power_w = 10 ** (rx_power_dbm / 10) / 1000
                    #     setattr(STA[idSTA], f'Rx_power_W_{B}', rx_power_w)
                    
                    #     # SINR
                    #     total_interference = getattr(STA[idSTA], f'Total_Interference_{B}')
                    #     temp = rx_power_w / (total_interference + P_noise_W)
                    #     sinr = 10 * np.log10(temp)
                    #     setattr(STA[idSTA], f'SINR_{B}', sinr)
                    
                    #     # DataRate_DL
                    #     num_ru = getattr(STA[idSTA], f'NumOfRU_{B}')
                    #     datarate = num_ru * Wsc * Get_pbs_per_Hz(sinr) / 1000 / 1000
                    #     setattr(STA[idSTA], f'DataRate_DL_{B}', datarate)

                    
                    
                    for j in range(LenSTAs):# original 2025-9-30
                        # print(AP[i].Dis_Between_AP_and_STAs)
                        # print(AP[i].ID_Of_STAs)
                        # print(len(AP[i].Dis_Between_AP_and_STAs))
                        # print(len(AP[i].ID_Of_STAs))

                        # exec('IDSTA=AP[i].ID_Of_STAs_{}[j]'.format(B))
                        # idSTA=locals()['IDSTA']
                        idSTA = getattr(AP[i], f'ID_Of_STAs_{B}')[j]
                        index_sta=AP[i].ID_Of_STAs.index(idSTA)

                        #dBi
                        #dBi_Rx
                        
                        # exec('P_mww=10**(AP[i].P_{}/10)'.format(B))
                        # P_mw=locals()['P_mww']
                        P_mw = 10 ** (getattr(AP[i], f'P_{B}') / 10)
                        P_beam_mw=P_mw/(Num_of_antennas) #allocate power for each beam
                        P_beam_dbm=10*np.log10(P_beam_mw)                        
                        
                        # print(P_mw,P_beam_dbm)
                        
                        # exec('STA[AP[i].ID_Of_STAs_{}[j]].Rx_power_dBm_{}=dBi+dBi_Rx+G_arr_AP+P_beam_dbm+G_beam{}_AP-PL(AP[i].Dis_Between_AP_and_STAs[index_sta],f)'.format(B,B,B))   
                        G_beam_var = globals()[f'G_beam{B}_AP']  # 或 locals()，取变量值
                        rx_power_dbm = (
                            dBi + dBi_Rx + G_arr_AP + P_beam_dbm + G_beam_var - PL(AP[i].Dis_Between_AP_and_STAs[index_sta], f)
                        )
                        setattr(STA[idSTA], f'Rx_power_dBm_{B}', rx_power_dbm)

                        
                        
                        # exec('STA[AP[i].ID_Of_STAs_{}[j]].Rx_power_W_{}=10**(STA[AP[i].ID_Of_STAs_{}[j]].Rx_power_dBm_{}/10)/1000'.format(B,B,B,B)) #in W
                        # 获取 STA 对象
                        sta_obj = STA[idSTA]                        
                        # 读取对应频段的 Rx_power_dBm
                        rx_power_dbm = getattr(sta_obj, f'Rx_power_dBm_{B}')                        
                        # 转换为瓦特
                        rx_power_w = 10 ** (rx_power_dbm / 10) / 1000                        
                        # 设置对应频段的 Rx_power_W 属性
                        setattr(sta_obj, f'Rx_power_W_{B}', rx_power_w)

                        
                        
                        # exec('temp=STA[AP[i].ID_Of_STAs_{}[j]].Rx_power_W_{}/(STA[AP[i].ID_Of_STAs_{}[j]].Total_Interference_{}+P_noise_W)'.format(B,B,B,B))
                        # XX=locals()['temp']
                        # exec('STA[AP[i].ID_Of_STAs_{}[j]].SINR_{}=10*np.log10(XX)'.format(B,B))
                        # exec('STA[AP[i].ID_Of_STAs_{}[j]].DataRate_DL_{}=STA[AP[i].ID_Of_STAs_{}[j]].NumOfRU_{}*Wsc*Get_pbs_per_Hz(STA[AP[i].ID_Of_STAs_{}[j]].SINR_{})/1000/1000'.format(B,B,B,B,B,B))
                        # print('test---')
                        # 1️⃣ 计算 SINR 分子 /分母
                        rx_power_w = getattr(sta_obj, f'Rx_power_W_{B}')
                        total_interference = getattr(sta_obj, f'Total_Interference_{B}')
                        
                        sinr_linear = rx_power_w / (total_interference + P_noise_W)
                        
                        # 2️⃣ 保存 SINR（dB）
                        sinr_db = 10 * np.log10(sinr_linear)
                        setattr(sta_obj, f'SINR_{B}', sinr_db)
                        
                        # 3️⃣ 计算下行数据速率
                        num_ru = getattr(sta_obj, f'NumOfRU_{B}')
                        datarate_dl = num_ru * Wsc * Get_pbs_per_Hz(sinr_db) / 1000 / 1000  # Mbps
                        setattr(sta_obj, f'DataRate_DL_{B}', datarate_dl)

                        
                        
                        
                        
                        
                        
                        
                        
                        # print(STA[AP[i].ID_Of_STAs_24G[j]].NumOfRU_24G)
                        # input()
                    # exec('len_g=len(AP[i].Group_{})'.format(B))
                    # Len_G=locals()['len_g']
                    Len_G = len(getattr(AP[i], f'Group_{B}'))
                    for g in range(Len_G):
                        # max_dis = 0
                        # for sta in range(len(AP[i].Group[g])):
                        #     if STA[AP[i].Group[g][sta]].Dis_to_AP>max_dis:
                        #         # print('a>b', STA[AP[i].Group[g][sta]].Dis_to_AP, max_dis)
                        #         max_dis=STA[AP[i].Group[g][sta]].Dis_to_AP
                                # print('max_dis=', max_dis)

                        # exec('lengg=len(AP[i].Group_{}[g])'.format(B))
                        # LenGG=locals()['lengg']
                        LenGG = len(getattr(AP[i], f'Group_{B}')[g])
                        for sta in range(LenGG):
                            # exec('POWER=AP[i].P_{}'.format(B))
                            # Power=locals()['POWER']
                            
                            Power = getattr(AP[i], f'P_{B}')

                            # exec('DIS=STA[AP[i].Group_{}[g][sta]].Dis_to_AP'.format(B))
                            # Diss=locals()['DIS']
                            group_attr = getattr(AP[i], f'Group_{B}')  # 获取 AP[i].Group_B
                            sta_idx = group_attr[g][sta]               # 获取具体 STA 索引
                            Diss = STA[sta_idx].Dis_to_AP              # 获取距离

#add two items:G_arr_sta=10*np.log10(2), G_beam24G_sta, etc.
                            # exec('Rx_power_dBmm = AP[i].P_{} +G_arr_sta+G_beam{}_sta- PL(STA[AP[i].Group_{}[g][sta]].Dis_to_AP, f)'.format(B,B,B))
                            # Rx_power_dBm=locals()['Rx_power_dBmm']
                            
                            # 假设 sta_idx 已经是 STA 索引
                            sta_obj = STA[sta_idx]                            
                            # AP 发射功率
                            Power = getattr(AP[i], f'P_{B}')                            
                            # 获取波束增益变量值
                            G_beam_sta = globals()[f'G_beam{B}_sta']  # 如果变量在函数内，可以用 locals()[...]                             
                            # STA 到 AP 的距离
                            Diss = sta_obj.Dis_to_AP                            
                            # 计算接收功率 dBm
                            Rx_power_dBm = Power + G_arr_sta + G_beam_sta - PL(Diss, f)                            
                            # 保存到 STA 对象
                            setattr(sta_obj, f'Rx_power_dBm_{B}', Rx_power_dBm)


                            
                            
                            # 获取 STA 索引
                            # group_attr = getattr(AP[i], f'Group_{B}')   # AP[i].Group_24G 之类
                            # sta_idx = group_attr[g][sta]                # 具体 STA 索引
                            
                            # # 计算接收功率
                            # Rx_power_dBm = (
                            #     getattr(AP[i], f'P_{B}') +   # AP 发射功率
                            #     G_arr_sta + 
                            #     getattr(AP[i], f'G_beam{B}_sta') - 
                            #     PL(STA[sta_idx].Dis_to_AP, f)
                            # )

                            

                            # Rx_power_dBm = AP[i].P - PL(STA[AP[i].Group[g][sta]].Dis_to_AP, f)  # in dBm

                            Rx_power_W = 10 ** (Rx_power_dBm / 10) / 1000  # in W

                            # exec('TOTAL_Int=AP[i].Total_Interference_{}'.format(B))
                            # total_ints=locals()['TOTAL_Int']
                            total_ints = getattr(AP[i], f'Total_Interference_{B}')

                            # exec('X_temp = Rx_power_W / (AP[i].Total_Interference_{} + P_noise_W)'.format(B))
                            # X=locals()['X_temp']
                            X = Rx_power_W / (total_ints + P_noise_W)
                            # X = Rx_power_W / (AP[i].Total_Interference + P_noise_W)
                            # print()
                            # print('Total interference:', total_ints)
                            #
                            # print('X=',X)
                            SINR = 10 * np.log10(X)

                            if SINR <0:
                                print("SINR<0!!")
                                print('Debug error:')
                                print('SINR:',SINR)
                                print('Band:',B)
                                print('TX power dBm:', Power)
                                print('Rx_power_dBm:',Rx_power_dBm)
                                print('Rx_power_w=',Rx_power_W)
                                print('Total interference:',total_ints)
                                print('Distance:',Diss)


                                # print (SINR, B)
                                # input()

                            # exec('AP[i].SINR_{}.append(SINR)'.format(B))
                            getattr(AP[i], f'SINR_{B}').append(SINR)
                            # AP[i].SINR.append(SINR)
                            
                            # exec('STA[AP[i].Group_{}[g][sta]].SINR_UP_{}=SINR'.format(B,B))
                            setattr(sta_obj, f'SINR_UP_{B}', SINR)
                            
                            # setattr(STA[AP[i].Group_[g][sta]], f'SINR_UP_{B}', SINR)
                            # STA[AP[i].Group[g][sta]].SINR_UP=SINR
                            # STA[AP[i].ID_Of_STAs[j]].DataRate_DL=STA[AP[i].ID_Of_STAs[j]].NumOfRU*Wsc*np.log2(1+STA[AP[i].ID_Of_STAs[j]].SINR)/1000000
                            
                            
                            # exec('STA[AP[i].Group_{}[g][sta]].DataRate_UL_{} = STA[AP[i].Group_{}[g][sta]].NumOfRU_{} * Wsc *Get_pbs_per_Hz(SINR) / 1000 / 1000'.format(B,B,B,B))
                            datarate_ul = getattr(sta_obj, f'NumOfRU_{B}') * Wsc * Get_pbs_per_Hz(SINR) / 1000 / 1000
                            setattr(sta_obj, f'DataRate_UL_{B}', datarate_ul)

                            
                            # STA[AP[i].Group[g][sta]].DataRate_UL = STA[AP[i].Group[g][sta]].NumOfRU * Wsc * Get_pbs_per_Hz(SINR) / 1000 / 1000
                            # sta_obj = STA[AP[i].Group_[g][sta]]
                            # ul_rate = sta_obj.NumOfRU_[B] * Wsc * Get_pbs_per_Hz(SINR) / 1e6
                            # setattr(sta_obj, f'DataRate_UL_{B}', ul_rate)


    # input()

    for i in range(len(AP)):
        # print('i=',i)
        ttt=[]
        nnn=[]
        for B in Band:
            if B=='24G':
                T_SIFS = 10 * 10 ** (-6)
            if B=='5G1':
                T_SIFS = 16 * 10 ** (-6)
            if B=='5G2':
                T_SIFS = 16 * 10 ** (-6)
            if B=='6G':
                T_SIFS = 16 * 10 ** (-6)
            T_DL = T_DIFS + T_backoff + T_dl + T_SIFS + T_OFDMA
            T_UL = T_DIFS + T_backoff + T_TF + 2 * T_SIFS + T_ul + T_MBA

            # exec('Round_time_temp = AP[i].Round_{}*(T_UL+T_DL)'.format(B))
            # Round_time=locals()['Round_time_temp']
            Round_time = getattr(AP[i], f'Round_{B}') * (T_UL + T_DL)
            
            # print('Round_time',Round_time)
            # ttt.append(Round_time)
            # input()
            # Round_time = AP[i].Round*(T_UL+T_DL)
            
            # exec('lenNeighbor1=len(AP[i].NeiInt_{})'.format(B))
            # lenNeighbor=locals()['lenNeighbor1']
            lenNeighbor = len(getattr(AP[i], f'NeiInt_{B}'))
            
            # nnn.append(lenNeighbor)
            # print('band=',B)
            # print('num of neighbor',lenNeighbor)
            # print('len=',len(AP[i].NeiInt_24G))
            # print('AP neighbor list=', AP[i].NeiInt_24G)
            if lenNeighbor>0:
                for j in range(lenNeighbor):
                    # print('j=',j)
                    # print('AP neighbor list=',AP[i].NeiInt_24G)
                    #
                    # print('round of neighbor=',AP[AP[i].NeiInt_24G[j]].Round_24G*(T_UL+T_DL))
                    # exec('Round_time = Round_time+AP[AP[i].NeiInt_{}[j]].Round_{}*(T_UL+T_DL)'.format(B,B))
                    # print(Round_time)
                    # input()

                    if B=='24G':
                        Round_time = Round_time + AP[AP[i].NeiInt_24G[j]].Round_24G * (T_UL + T_DL)
                    if B=='5G1':
                        Round_time = Round_time + AP[AP[i].NeiInt_5G1[j]].Round_5G1 * (T_UL + T_DL)
                    if B=='5G2':
                        Round_time = Round_time + AP[AP[i].NeiInt_5G2[j]].Round_5G2 * (T_UL + T_DL)
                    if B=='6G':
                        Round_time = Round_time + AP[AP[i].NeiInt_6G[j]].Round_6G * (T_UL + T_DL)
                    # print('non exec')
                    # print(Round_time)
                    # input()



                    # Round_time = Round_time+AP[AP[i].NeiInt[j]].Round*(T_UL+T_DL)
            # if lenNeighbor==0:
            #     exec('Round_time = Round_time+AP[AP[i].NeiInt_{}[j]].Round_{}*(T_UL+T_DL)'.format(B, B))
            # exec('AP[i].Total_time_{}=Round_time'.format(B))
            setattr(AP[i], f'Total_time_{B}', Round_time)


    # rou=0
    # Ro=[]
    # NuSTA=[]
    # sta=0
    # for i in range(len(AP)):
    #     Ro.append(AP[i].Round_24G)
    #     NuSTA.append(AP[i].Num_Of_STAs)
    #     rou=rou+AP[i].Round_24G
    #     sta=sta+AP[i].Num_Of_STAs
    # print('numOfAPs:',len(AP))
    # print('numOfrou',rou)
    # print('RoList',Ro)
    # print('NuSTA',NuSTA)
    # print('numOfsta',sta)
    # Plot_fig()
    #
    # input()
    global MaxThroughput
    MaxThroughput = 0
    global MinThroughput
    MinThroughput = np.inf
    for i in range(len(AP)):
        # print(i)
        for B in Band:

            # exec('len_temp10=len(AP[i].ID_Of_STAs_{})'.format(B))
            # numStas=locals()['len_temp10']
            numStas = len(getattr(AP[i], f'ID_Of_STAs_{B}'))
            for j in range(numStas):
                # count=count+1
                # exec('idOfsta=AP[i].ID_Of_STAs_{}[j]'.format(B))
                # idd_sta=locals()['idOfsta']
                idd_sta = getattr(AP[i], f'ID_Of_STAs_{B}')[j]

                # T_SIFS = 10 * 10 ** (-6)


                # print(AP[i].Total_time_24G)
                # input()

                # exec('debug_t=AP[i].Total_time_{}'.format(B))
                # debug_tt=locals()['debug_t']
                debug_tt = getattr(AP[i], f'Total_time_{B}')
                
                if debug_tt==0:
                    print("Total_time = 0!!")
                    print("B=",B)
                    print("AP No.:",i)
                    print('There is a error!')
                    input()

                # exec('STA[idd_sta].Throughput_{}=(STA[idd_sta].DataRate_DL_{}*T_dl+STA[idd_sta].DataRate_UL_{}*T_ul)/AP[i].Total_time_{}'.format(B,B,B,B))
                throughput = (getattr(STA[idd_sta], f'DataRate_DL_{B}') * T_dl +
                              getattr(STA[idd_sta], f'DataRate_UL_{B}') * T_ul) / getattr(AP[i], f'Total_time_{B}')
                setattr(STA[idd_sta], f'Throughput_{B}', throughput)



    idd_sta=0
    for idd_sta in range(len(STA)):
        STA[idd_sta].Throughput=STA[idd_sta].Throughput_24G+STA[idd_sta].Throughput_5G1+STA[idd_sta].Throughput_5G2+STA[idd_sta].Throughput_6G
        if STA[idd_sta].Throughput==0:
            print("Throughput=0!!")
            print('STA =',idd_sta)
            print('AP =',STA[idd_sta].IDOfAP)
            print('DL SINR =',STA[idd_sta].SINR)
            print('UL SINR =', STA[idd_sta].SINR_UP)
            print('Throughput error!')
            # Plot_fig()
            # input()
        # else:
        #     print(count)
        Throughput_of_STAs.append(STA[idd_sta].Throughput)

        # print(Throughput_of_STAs)
        # input()

        if STA[idd_sta].Throughput>MaxThroughput:
            MaxThroughput=STA[idd_sta].Throughput
            APNo=STA[idd_sta].IDOfAP
            idofsta=idd_sta

            # print('STA[idd_sta].IDOfAP',STA[idd_sta].IDOfAP)
            # input()


            ChannelNo24G = AP[STA[idd_sta].IDOfAP].C_24G
            ChannelNo5G1 = AP[STA[idd_sta].IDOfAP].C_5G1
            ChannelNo5G2 = AP[STA[idd_sta].IDOfAP].C_5G2
            ChannelNo6G = AP[STA[idd_sta].IDOfAP].C_6G
            IDOfstas_24G=AP[STA[idd_sta].IDOfAP].ID_Of_STAs_24G
            IDOfstas_5G1 = AP[STA[idd_sta].IDOfAP].ID_Of_STAs_5G1
            IDOfstas_5G2 = AP[STA[idd_sta].IDOfAP].ID_Of_STAs_5G2
            IDOfstas_6G = AP[STA[idd_sta].IDOfAP].ID_Of_STAs_6G

        if STA[idd_sta].Throughput<MinThroughput:

            MinThroughput=STA[idd_sta].Throughput
            APNo1 = STA[idd_sta].IDOfAP
            idofsta1 = idd_sta
            ChannelNo24G1 = AP[STA[idd_sta].IDOfAP].C_24G
            ChannelNo5G11= AP[STA[idd_sta].IDOfAP].C_5G1
            ChannelNo5G21 = AP[STA[idd_sta].IDOfAP].C_5G2
            ChannelNo6G1 = AP[STA[idd_sta].IDOfAP].C_6G
            IDOfstas_24G1 = AP[STA[idd_sta].IDOfAP].ID_Of_STAs_24G
            IDOfstas_5G11 = AP[STA[idd_sta].IDOfAP].ID_Of_STAs_5G1
            IDOfstas_5G21 = AP[STA[idd_sta].IDOfAP].ID_Of_STAs_5G2
            IDOfstas_6G1 = AP[STA[idd_sta].IDOfAP].ID_Of_STAs_6G

    # print('MaxThroughput--',MaxThroughput)
    # print('MinThroughput--',MinThroughput)
    
    # input()
    # print('APNo',APNo)
    # print('ChannelNo24G',ChannelNo24G)
    # print('ChannelNo5G1',ChannelNo5G1)
    # print('ChannelNo5G2',ChannelNo5G2)
    # print('ChannelNo6G',ChannelNo6G)
    # print('IDOfstas_24G',IDOfstas_24G)
    # print('IDOfstas_5G1', IDOfstas_5G1)
    # print('IDOfstas_5G2', IDOfstas_5G2)
    # print('IDOfstas_6G', IDOfstas_6G)
    # print('idofsta',idofsta)
    # print('---------')

    # For savety margin test
    if SafetyMargin==1:
        print('MinThroughput',MinThroughput)
        print('For safety margin test has done!')
        input()

    # print('APNo1',APNo1)
    # print('ChannelNo24G1',ChannelNo24G1)
    # print('ChannelNo5G11',ChannelNo5G11)
    # print('ChannelNo5G21',ChannelNo5G21)
    # print('ChannelNo6G1',ChannelNo6G1)
    # print('IDOfstas_24G1',IDOfstas_24G1)
    # print('IDOfstas_5G11', IDOfstas_5G11)
    # print('IDOfstas_5G21', IDOfstas_5G21)
    # print('IDOfstas_6G1', IDOfstas_6G1)
    # print('idofsta1',idofsta1)

                        #Calculate the throughput of STAs
                        # Ti=(T_DL+T_UL)*len(AP[i].Group)
                        # STA[AP[i].Group[g][sta]].Throughput=(STA[AP[i].Group[g][sta]].DataRate_UL*T_ul+STA[AP[i].Group[g][sta]].DataRate_DL*T_dl)/(Ti*)

    #-------------------------------------------------------
    # atest=0
    # for i in range(len(AP)):
    #     print("2.4GC",AP[i].C_24G)
    #     print("5G1C", AP[i].C_5G1)
    #     print("5G2C", AP[i].C_5G2)
    #     print("6GC", AP[i].C_6G)
    #
    #     print("2.4GRU",AP[i].NumRUs_24G)
    #     print("5G1RU", AP[i].NumRUs_5G1)
    #     print("5G2RU", AP[i].NumRUs_5G2)
    #     print("6GRU", AP[i].NumRUs_6G)
    #
    #     print("2.4Ggroup",len(AP[i].Group_24G))
    #     print("5G1group", len(AP[i].Group_5G1))
    #     print("5G2group", len(AP[i].Group_5G2))
    #     print("6Ggroup", len(AP[i].Group_6G))
    #
    #     print('NumOfstations=',AP[i].Num_Of_STAs)
    #     input()
    
    # print("Leave RUtest2")

    return

# This function is for APCoordination for RU assignment
# The bigger RU is assigned with the farther station
def RUassignment_APCoordination_modelVerify():
    #Determine the channel width
    # print('begin RU assignment')
    # input()

    # for i in range(len(AP)):
    #     print('AP_24G_C=',AP[i].C_24G)
    #     print('AP_5G1_C=', AP[i].C_5G1)
    #     print('AP_5G2_C=', AP[i].C_5G2)
    #     print('AP_6G_C=', AP[i].C_6G)
    #     input()

    for i in range(len(AP)):
        
        # print(AP[i].Num_Of_STAs)
        
        # input()
        
        if AP[i].C_24G in C20M:
            AP[i].NumRUs_24G = 9
            # AP[i].RUperUser_24G = AP[i].NumRUs_24G / AP[i].Num_Of_STAs
        if AP[i].C_24G in C40M:
            AP[i].NumRUs_24G = 18
            # AP[i].RUperUser_24G = AP[i].NumRUs_24G / AP[i].Num_Of_STAs
        if AP[i].C_24G in C80M:
            AP[i].NumRUs_24G = 36
            # AP[i].RUperUser_24G = AP[i].NumRUs_24G / AP[i].Num_Of_STAs
        if AP[i].C_24G in C160M:
            AP[i].NumRUs_24G = 72
            # AP[i].RUperUser_24G = AP[i].NumRUs_24G / AP[i].Num_Of_STAs
        if AP[i].C_24G in C320M:
            AP[i].NumRUs_24G = 144
            # AP[i].RUperUser_24G = AP[i].NumRUs_24G / AP[i].Num_Of_STAs

        if AP[i].C_5G1 in C20M:
            AP[i].NumRUs_5G1 = 9
            # AP[i].RUperUser_5G1 = AP[i].NumRUs_5G1 / AP[i].Num_Of_STAs
        if AP[i].C_5G1 in C40M:
            AP[i].NumRUs_5G1 = 18
            # AP[i].RUperUser_5G1 = AP[i].NumRUs_5G1 / AP[i].Num_Of_STAs
        if AP[i].C_5G1 in C80M:
            AP[i].NumRUs_5G1 = 36
            # AP[i].RUperUser_5G1 = AP[i].NumRUs_5G1 / AP[i].Num_Of_STAs
        if AP[i].C_5G1 in C160M:
            AP[i].NumRUs_5G1 = 72
            # AP[i].RUperUser_24G = AP[i].NumRUs_24G / AP[i].Num_Of_STAs
        if AP[i].C_5G1 in C320M:
            AP[i].NumRUs_5G1 = 144
            # AP[i].RUperUser_24G = AP[i].NumRUs_24G / AP[i].Num_Of_STAs

        if AP[i].C_5G2 in C20M:
            AP[i].NumRUs_5G2 = 9
            # AP[i].RUperUser_5G2 = AP[i].NumRUs_5G2 / AP[i].Num_Of_STAs
        if AP[i].C_5G2 in C40M:
            AP[i].NumRUs_5G2 = 18
            # AP[i].RUperUser_5G2 = AP[i].NumRUs_5G2 / AP[i].Num_Of_STAs
        if AP[i].C_5G2 in C80M:
            AP[i].NumRUs_5G2 = 36
            # AP[i].RUperUser_5G2 = AP[i].NumRUs_5G2 / AP[i].Num_Of_STAs
        if AP[i].C_5G2 in C160M:
            AP[i].NumRUs_5G2 = 72
            # AP[i].RUperUser_24G = AP[i].NumRUs_24G / AP[i].Num_Of_STAs
        if AP[i].C_5G2 in C320M:
            AP[i].NumRUs_5G2 = 144
            # AP[i].RUperUser_24G = AP[i].NumRUs_24G / AP[i].Num_Of_STAs

        if AP[i].C_6G in C20M:
            AP[i].NumRUs_6G = 9
            # AP[i].RUperUser_6G = AP[i].NumRUs_6G / AP[i].Num_Of_STAs
        if AP[i].C_6G in C40M:
            AP[i].NumRUs_6G = 18
            # AP[i].RUperUser_6G = AP[i].NumRUs_6G / AP[i].Num_Of_STAs
        if AP[i].C_6G in C80M:
            AP[i].NumRUs_6G = 36
            # AP[i].RUperUser_6G = AP[i].NumRUs_6G / AP[i].Num_Of_STAs
        if AP[i].C_6G in C160M:
            AP[i].NumRUs_6G = 72
            # AP[i].RUperUser_24G = AP[i].NumRUs_24G / AP[i].Num_Of_STAs
        if AP[i].C_6G in C320M:
            AP[i].NumRUs_6G = 144
            # AP[i].RUperUser_24G = AP[i].NumRUs_24G / AP[i].Num_Of_STAs

    RUlist24G=[]
    RUlist5G1 = []
    RUlist5G2 = []
    RUlist6G = []
    for i in range(len(AP)):
        RUlist24G.append(AP[i].NumRUs_24G)
        RUlist5G1.append(AP[i].NumRUs_5G1)
        RUlist5G2.append(AP[i].NumRUs_5G2)
        RUlist6G.append(AP[i].NumRUs_6G)
        # print("Current RU:")
        # print('i=', i)
        # print(AP[i].NumRUs_6G)
    # print(RUlist24G)
    # print(RUlist5G1)
    # print(RUlist5G2)
    # print(RUlist6G)
    # input()
    if LocalOptimizerMethod==0:
        Flag24G=0
        Flag5G1=0
        Flag5G2 = 0
        Flag6G = 0
        while(1):
            for i in range(len(AP)):
                # if AP[i].C_24G in C20M:
                #     AP[i].NumRUs_24G=9
                AP[i].RUperUser_24G=AP[i].NumRUs_24G/AP[i].Num_Of_STAs
                
                # print(AP[i].Num_Of_STAs)
                
                # input()
                # if AP[i].C_24G in C40M:
                #     AP[i].NumRUs_24G=18
                # AP[i].RUperUser_24G = AP[i].NumRUs_24G / AP[i].Num_Of_STAs
                # if AP[i].C_24G in C80M:
                #     AP[i].NumRUs_24G=36
                # AP[i].RUperUser_24G = AP[i].NumRUs_24G / AP[i].Num_Of_STAs
    
                # if AP[i].C_5G1 in C20M:
                #     AP[i].NumRUs_5G1=9
                AP[i].RUperUser_5G1 = AP[i].NumRUs_5G1 / AP[i].Num_Of_STAs
                # if AP[i].C_5G1 in C40M:
                #     AP[i].NumRUs_5G1=18
                # AP[i].RUperUser_5G1 = AP[i].NumRUs_5G1 / AP[i].Num_Of_STAs
                # if AP[i].C_5G1 in C80M:
                #     AP[i].NumRUs_5G1=36
                # AP[i].RUperUser_5G1 = AP[i].NumRUs_5G1 / AP[i].Num_Of_STAs
    
                # if AP[i].C_5G2 in C20M:
                #     AP[i].NumRUs_5G2=9
                AP[i].RUperUser_5G2 = AP[i].NumRUs_5G2 / AP[i].Num_Of_STAs
                # if AP[i].C_5G2 in C40M:
                #     AP[i].NumRUs_5G2=18
                # AP[i].RUperUser_5G2 = AP[i].NumRUs_5G2 / AP[i].Num_Of_STAs
                # if AP[i].C_5G2 in C80M:
                #     AP[i].NumRUs_5G2=36
                # AP[i].RUperUser_5G2 = AP[i].NumRUs_5G2 / AP[i].Num_Of_STAs
    
                # if AP[i].C_6G in C20M:
                #     AP[i].NumRUs_6G=9
                AP[i].RUperUser_6G = AP[i].NumRUs_6G / AP[i].Num_Of_STAs
                # if AP[i].C_6G in C40M:
                #     AP[i].NumRUs_6G=18
                # AP[i].RUperUser_6G = AP[i].NumRUs_6G / AP[i].Num_Of_STAs
                # if AP[i].C_6G in C80M:
                #     AP[i].NumRUs_6G=36
                # AP[i].RUperUser_6G = AP[i].NumRUs_6G / AP[i].Num_Of_STAs
    
                # print(AP[i].NumRUs_6G)
                # print('i=',i)
                # print('test')
                # input()
    
            MaxRU_24G=0
            MaxRU_5G1 = 0
            MaxRU_5G2 = 0
            MaxRU_6G = 0
            MinRU_24G=inf
            MinRU_5G1 = inf
            MinRU_5G2 = inf
            MinRU_6G = inf
            Mini_24G=inf
            Mini_5G1 = inf
            Mini_5G2 = inf
            Mini_6G = inf
            Maxi_24G=inf
            Maxi_5G1 = inf
            Maxi_5G2 = inf
            Maxi_6G = inf
            for i in range(len(AP)):
    
                if AP[i].RUperUser_24G > MaxRU_24G:
                    MaxRU_24G=AP[i].RUperUser_24G
                    Maxi_24G=i
                if AP[i].RUperUser_5G1 > MaxRU_5G1:
                    MaxRU_5G1=AP[i].RUperUser_5G1
                    Maxi_5G1=i
                if AP[i].RUperUser_5G2 > MaxRU_5G2:
                    MaxRU_5G2=AP[i].RUperUser_5G2
                    Maxi_5G2=i
                if AP[i].RUperUser_6G > MaxRU_6G:
                    MaxRU_6G=AP[i].RUperUser_6G
                    Maxi_6G=i
    
                if AP[i].RUperUser_24G < MinRU_24G:
                    MinRU_24G=AP[i].RUperUser_24G
                    Mini_24G=i
                if AP[i].RUperUser_5G1 < MinRU_5G1:
                    MinRU_5G1=AP[i].RUperUser_5G1
                    Mini_5G1=i
                if AP[i].RUperUser_5G2 < MinRU_5G2:
                    MinRU_5G2=AP[i].RUperUser_5G2
                    Mini_5G2=i
                if AP[i].RUperUser_6G < MinRU_6G:
                    MinRU_6G=AP[i].RUperUser_6G
                    Mini_6G=i
            # print("Max24G",Maxi_24G)
            # print("Max5G1", Maxi_5G1)
            # print("Max5G2", Maxi_5G2)
            # print("Max6G", Maxi_6G)
            #
            # print("Min24G", Mini_24G)
            # print("Min5G1", Mini_5G1)
            # print("Min5G2", Mini_5G2)
            # print("Min6G", Mini_6G)
            #
            # print("Max24G_RU",AP[Maxi_24G].NumRUs_24G)
            # print("Max5G1_RU", AP[Maxi_5G1].NumRUs_5G1)
            # print("Max5G2_RU", AP[Maxi_5G2].NumRUs_5G2)
            # print("Max6G_RU", AP[Maxi_6G].NumRUs_6G)
            #
            # print("Min24G_RU", AP[Mini_24G].NumRUs_24G)
            # print("Min5G1_RU", AP[Mini_5G1].NumRUs_5G1)
            # print("Min5G2_RU", AP[Mini_5G2].NumRUs_5G2)
            # print("Min6G_RU", AP[Mini_6G].NumRUs_6G)
    
    
    
            if AP[Maxi_24G].NumRUs_24G-AP[Mini_24G].NumRUs_24G>=2:
                AP[Maxi_24G].NumRUs_24G=AP[Maxi_24G].NumRUs_24G-1
                AP[Mini_24G].NumRUs_24G=AP[Mini_24G].NumRUs_24G+1
            else:
                Flag24G=1
            if AP[Maxi_6G].NumRUs_6G-AP[Mini_6G].NumRUs_6G>=2:
                AP[Maxi_6G].NumRUs_6G=AP[Maxi_6G].NumRUs_6G-1
                AP[Mini_6G].NumRUs_6G=AP[Mini_6G].NumRUs_6G+1
            else:
                Flag6G=1
            if AP[Maxi_5G1].NumRUs_5G1-AP[Mini_5G1].NumRUs_5G1>=2:
                AP[Maxi_5G1].NumRUs_5G1=AP[Maxi_5G1].NumRUs_5G1-1
                AP[Mini_5G1].NumRUs_5G1=AP[Mini_5G1].NumRUs_5G1+1
            else:
                Flag5G1=1
            if AP[Maxi_5G2].NumRUs_5G2-AP[Mini_5G2].NumRUs_5G2>=2:
                AP[Maxi_5G2].NumRUs_5G2=AP[Maxi_5G2].NumRUs_5G2-1
                AP[Mini_5G2].NumRUs_5G2=AP[Mini_5G2].NumRUs_5G2+1
            else:
                Flag5G2=1
    
    
    
            # print('AP coordination:')
            #
            # print("Max6G_RU", AP[Maxi_6G].NumRUs_6G)
            # print("Min6G_RU", AP[Mini_6G].NumRUs_6G)
            # input()
            if Flag24G==1 and Flag5G1==1  and Flag5G2==1 and Flag6G==1:
                break
    
        RUlist24G = []
        RUlist5G1 = []
        RUlist5G2 = []
        RUlist6G = []
        for i in range(len(AP)):
            RUlist24G.append(AP[i].NumRUs_24G)
            RUlist5G1.append(AP[i].NumRUs_5G1)
            RUlist5G2.append(AP[i].NumRUs_5G2)
            RUlist6G.append(AP[i].NumRUs_6G)
            # print("Current RU:")
            # print('i=', i)
            # print(AP[i].NumRUs_6G)
        # print(RUlist24G)
        # print(RUlist5G1)
        # print(RUlist5G2)
        # print(RUlist6G)
        # input()








        #
        # AP[i].RUperUser_24G
        # AP[i].RUperUser_5G1
        # AP[i].RUperUser_5G2
        # AP[i].RUperUser_6G



    for i in range(len(AP)):#2.4G
        # print('number of AP:',i)
        for B in Band:
            if Band.index(B)==0:
                FF=[20,40]
                f=fc_2dot4G
            if Band.index(B)==1:
                FF=[20,40,80,160]
                f=fc_5GI
            if Band.index(B)==2:
                FF=[20,40,80]
                f=fc_5GII
            if Band.index(B)==3:
                FF=[20,40,80,160,320]
                f=fc_6G

            for F in FF:
                # Fre = []
                exec('Fre=Frequence{}_{}'.format(F,B))
                Freq=locals()['Fre']
                exec('chh=AP[i].C_{}'.format(B))
                ch=locals()['chh']
                # print(ch)
                # input()
                if ch in Freq:
                # if AP[i].C_24G in Freq:
                    if F==20:
                        #Maximum number of 106-tone RUs 即106RU的个数，原始带宽
                        max_num_of_RU=2
                    if F==40:
                        max_num_of_RU = 4
                    if F==80:
                        max_num_of_RU = 8
                    if F==160:
                        max_num_of_RU = 16
                    if F==320:
                        max_num_of_RU = 32

                    #Determine the Maximum number of 106-tone RUs
                    exec('max_num_of_RUU=AP[i].NumRUs_{}'.format(B))
                    max_num_of_RUuu = locals()['max_num_of_RUU']
                    # max_num_of_RUuu=15
                    max_num_of_RU=max_num_of_RUuu//4  #经过AP协调之后，每个AP的106RU的个数，可能是原始带宽+来自其他AP的RU
                    RU_rem=max_num_of_RUuu%4

                    # print(max_num_of_RUuu)
                    # print(max_num_of_RU)
                    # print(RU_rem)
                    # input()

                    RUset_Full=[]
                    RUset_Part =[]
                    for ru in range(0,max_num_of_RU):
                        RUset_Full.append(4*26+2)
                    if RU_rem !=0:
                        for j in range(0,RU_rem):
                            # print('j=',j)
                            # print('max_num_of_ru=',max_num_of_RU)
                            # print('AP_c=',ch)
                            ind=j%max_num_of_RU

                            RUset_Full[ind]=RUset_Full[ind]+26

                        # for rem_ru in range(0,RU_rem):
                        #     print('RUset_full')
                        #     print(RUset_Full)
                        #     RUset_Full[rem_ru]=RUset_Full[rem_ru] + 26
                        #     print('rem_ru', rem_ru)
                    # print('======')
                    # print('RUset_full',RUset_Full)
                    # print('max_num_of_RU',max_num_of_RU)
                    # print('======')
                    # input()


                    # print(B)
                    # print('max_num_of_RU',max_num_of_RU)
                    # print('max_num_of_RU', int(max_num_of_RU))
                    # input()
                    # b=AP[i].Num_Of_STAs//Max_transmissions #Quotient,
                    # a=AP[i].Num_Of_STAs%Max_transmissions #Remainder




#Beamforming revised
                    Max_transmissions=Num_of_antennas*max_num_of_RU
                    #Max_transmissions=max_num_of_RU #波束的个数
                    
                    
                        
                    
                    # print("Beamforming nmus: ",max_num_of_RU)
                    
                    # b=AP[i].Num_Of_STAs//Max_transmissions #Quotient,
                    # a=AP[i].Num_Of_STAs%Max_transmissions #Remainder
                    exec('numSTAs=len(AP[i].ID_Of_STAs_{})'.format(B))
                    NumSTAs=locals()['numSTAs']
                    # print(NumSTAs,len(AP[i].ID_Of_STAs_24G))
                    # input()
                    round=int(np.ceil(NumSTAs/Max_transmissions))
                    # print('i',i)
                    # print('B',B)
                    # print('round:',round)
                    exec('AP[i].Round_{}=round'.format(B))


                    for r in range(round):
                        exec('AP[i].Group_{}.append([])'.format(B))

                    # The stations should be ordered
                    for j in range(NumSTAs):
                        r = j % round
                        exec('AP[i].Group_{}[r].append(AP[i].ID_Of_STAs_{}[j])'.format(B,B))
                        # print(AP[i].Group_24G[r])
                        # input()
                    # input()

                    for g in range(round):
                        # Sort the stations according to the distances

                        # list = [34, 23, 77, 54, 99, 10]
                        # sorted_id = sorted(range(len(list)), key=lambda x: list[x], reverse=True)
                        # print(sorted_id)



                        exec('len1=len(AP[i].Group_{}[g])'.format(B))
                        len_temp=locals()['len1']
                        b=len_temp//Num_of_antennas#Quotient
                        a =len_temp%Num_of_antennas# Remainder
                        if a==0:
                            a_temp=0
                            # print('a=0')

                            # exec('listt=AP[i].Group_{}[g]'.format(B))
                            # list_temp = locals()['listt']
                            # # print('list_temp',list_temp)
                            # Diss_temp=[]
                            # for d in range(0,len(list_temp)):
                            #     diss=((AP[i].x-STA[list_temp[d]].x)**2+(AP[i].y-STA[list_temp[d]].y)**2)**(0.5)
                            #     # print(list_temp[d])
                            #     # input()
                            #     Diss_temp.append(diss)
                            # # print('dis_temp',Diss_temp)
                            #
                            #
                            #
                            #
                            # for jj in range(len_temp):
                            #     temp_rem=jj%max_num_of_RU
                            #     #exec('STA[AP[i].Group_{}[g][j]].NumOfRU=RU{}_{}[temp_rem]'.format(B,F,max_num_of_RU))
                            #     # exec('STA[AP[i].Group_{}[g][j]].NumOfRU_{}=RU{}_{}[temp_rem]'.format(B, B, F, max_num_of_RU))
                            #
                            #     # print('len_temp',len_temp)
                            #     # print('RUset_Full',RUset_Full)
                            #     # print('max_num_of_RU', max_num_of_RU)
                            #     # print('jj',jj)
                            #     # print('temp_rem',temp_rem)
                            #     #
                            #     # print('g',g)
                            #     #
                            #     # exec('group=AP[i].Group_{}[g][jj]'.format(B))
                            #     # group_temp = locals()['group']
                            #     # exec('groupsta=AP[i].Group_{}[g]'.format(B))
                            #     # group_temp_sta = locals()['groupsta']
                            #     # print('group-sta',group_temp)
                            #     # print('group',group_temp_sta)
                            #
                            #
                            #
                            #     # input()
                            #     exec('STA[AP[i].Group_{}[g][jj]].NumOfRU_{}=RUset_Full[temp_rem]'.format(B, B))
                        if a==0 or a >0:

                            # exec('listt=AP[i].Group_{}[g]'.format(B))
                            # list_temp = locals()['listt']
                            # print('list_temp', list_temp)
                            # Diss_temp = []
                            # for d in range(0, len(list_temp)):
                            #     diss = ((AP[i].x - STA[list_temp[d]].x) ** 2 + (
                            #                 AP[i].y - STA[list_temp[d]].y) ** 2) ** (0.5)
                            #     # print(list_temp[d])
                            #     # input()
                            #     Diss_temp.append(diss)
                            # print('dis_temp', Diss_temp)

                            SS=[]
                            # print('a!=0')
                            for k in range(Num_of_antennas):
                                SS.append([])
                            for j in range(len_temp):
                                temp_rem=j%Num_of_antennas
                                exec('SS[temp_rem].append(AP[i].Group_{}[g][j])'.format(B))
                                # SS[temp_rem].append(AP[i].Group_24G[g][j])
                                # print('temp_rem',temp_rem)
                                # print('SS[temp]',SS)
                                # input()
                            len_ss=np.zeros(Num_of_antennas)
                            for j in range(Num_of_antennas):
                                len_ss[j]=len(SS[j])
                            max_RU=max(len_ss)

                            # print(SS)




                            for Max_RU in range(1,max_num_of_RU+1):
                                # print('max_RU',max_RU)
                                # print('Max_RU',Max_RU)
                                # input()

                                if max_RU==Max_RU: # Why use this condition? Because we need to determine the RU set!
                                    # print('part_index')
                                    # print(max_RU)
                                    # input()

                                    for j in range(Num_of_antennas):

                                        # Sort the stations for each spatial stream. Bigger RU for farther station
                                        exec('listt=SS[j]'.format(B))
                                        list_temp = locals()['listt']
                                        # print('list_temp', list_temp)
                                        Diss_temp = []
                                        for d in range(0, len(list_temp)):
                                            
                                            # if STA[list_temp[d]].z==h_sta:
                                                
                                            
                                            #     diss = ((AP[i].x - STA[list_temp[d]].x) ** 2 + (
                                            #             AP[i].y - STA[list_temp[d]].y) ** 2) ** (0.5)
                                            #     # print(list_temp[d])
                                            #     # input()
    
                                            #     #3D distance
                                            #     diss=(diss**2+H_sta**2)**(0.5)
                                                
                                            # else:
                                            diss = ((AP[i].x - STA[list_temp[d]].x) ** 2 + (
                                                        AP[i].y - STA[list_temp[d]].y) ** 2+(AP[i].z - STA[list_temp[d]].z) ** 2) ** (0.5)

                                            Diss_temp.append(diss)
                                        # print('dis_temp', Diss_temp)


                                        sorted_id = sorted(range(len(Diss_temp)), key=lambda x: Diss_temp[x], reverse=True)
                                        # print(sorted_id)
                                        # input()



                                        if len(SS[j]) != 0:

                                            exec('max_num_of_RUUT=AP[i].NumRUs_{}'.format(B))
                                            max_num_of_RUuuT = locals()['max_num_of_RUUT']
                                            # max_num_of_RUuu=15
                                            # Max_RU here is the number of RU subsets
                                            # max_num_of_RU here is the number of 26-tone RUs
                                            max_num_of_RUt = max_num_of_RUuuT // len(SS[j])
                                            RU_remt = max_num_of_RUuuT % len(SS[j])

                                            # print('max_RU',Max_RU)
                                            # print('max_num_of_RUuuT',max_num_of_RUuuT)
                                            # print('max_num_of_RU',max_num_of_RU)
                                            # print('RU_rem',RU_rem)
                                            # print('number of AP i:',i)
                                            # input()

                                            RUset_Part = []
                                            for ru in range(0, len(SS[j])):
                                                RU_element = max_num_of_RUt * 26
                                                if max_num_of_RUt * 26 >= 16 * 4 and max_num_of_RUt * 26 < 16 * 8:
                                                    RU_element = max_num_of_RUt * 26 + 2
                                                if max_num_of_RUt * 26 >= 16 * 8:
                                                    RU_element = max_num_of_RUt * 26 + 4
                                                RUset_Part.append(RU_element)
                                            if RU_remt != 0:
                                                for jj in range(0, RU_remt):
                                                    ind = jj % len(SS[j])
                                                    RUset_Part[ind] = RUset_Part[ind] + 26

                                                # for rem_ru in range(0, RU_rem):
                                                #     RUset_Part[rem_ru] = RUset_Part[rem_ru] + 26

                                            # print('RUset_Part', RUset_Part)
                                            # input()

                                            for k in range(len(SS[j])):
                                                # exec('STA[AP[i].ID_Of_STAs[SS[j][k]]].NumOfRU = RU20_{}[k]'.format(Max_RU))
                                                # print(RUset_Part)
                                                # print(RUlist24G)
                                                # print(RUlist5G1)
                                                # print(RUlist5G2)
                                                # print(RUlist6G)
                                                # input()
                                                # exec('STA[SS[j][k]].NumOfRU_{} = RU{}_{}[k]'.format(B,F,Max_RU))
                                                # print('RUset_Part=',RUset_Part)
                                                # print('k=',k)
                                                # print('j=',j)
                                                # print('ss=',SS)
                                                # print('SS[j][k]=',SS[j][k])
                                                # exec('axx=STA[SS[j][k]].NumOfRU_{} = RUset_Part[k]'.format(B))

                                                # exec('STA[SS[j][k]].NumOfRU_{} = RUset_Part[k]'.format(B))
                                                exec('STA[list_temp[sorted_id[k]]].NumOfRU_{} = RUset_Part[k]'.format(B))
                                                # exec('testRU=STA[list_temp[sorted_id[k]]].NumOfRU_{}'.format(B))
                                                # testRUU= locals()['testRU']
                                                # exec('testSTA=list_temp[sorted_id[k]]')
                                                # testSTAa= locals()['testSTA']
                                                # print('testSTAa',testSTAa)
                                                # print('testRUU',testRUU)
                                                # input()

                            # AP[i].Group=AP[i].Group+SS
                           # input()
                        # Calculate the data rate
                    # if AP[i].C in C_2dot4G:
                    # input()
                    # f=fc_2dot4G
                    # if AP[i].C in C_5GI:
                    #     f=fc_5GI
                    #
                    # for j in range(AP[i].Num_Of_STAs):
                    #     P_RX=AP[i].P-PL(AP[i].Dis_Between_AP_and_STAs[j],f)
                    exec('lenSTAs=len(AP[i].ID_Of_STAs_{})'.format(B))
                    LenSTAs=locals()['lenSTAs']
                    for j in range(LenSTAs):
                        # print(AP[i].Dis_Between_AP_and_STAs)
                        # print(AP[i].ID_Of_STAs)
                        # print(len(AP[i].Dis_Between_AP_and_STAs))
                        # print(len(AP[i].ID_Of_STAs))

                        exec('IDSTA=AP[i].ID_Of_STAs_{}[j]'.format(B))
                        idSTA=locals()['IDSTA']
                        index_sta=AP[i].ID_Of_STAs.index(idSTA)

                        #dBi
                        #dBi_Rx
                        
                        exec('P_mww=10**(AP[i].P_{}/10)'.format(B))
                        P_mw=locals()['P_mww']
                        
                        P_beam_mw=P_mw/(Num_of_antennas) #allocate power for each beam
                        P_beam_dbm=10*np.log10(P_beam_mw)                        
                        
                        # print(P_mw,P_beam_dbm)
                        
                        exec('STA[AP[i].ID_Of_STAs_{}[j]].Rx_power_dBm_{}=dBi+dBi_Rx+G_arr_AP+P_beam_dbm+G_beam{}_AP-PL(AP[i].Dis_Between_AP_and_STAs[index_sta],f)'.format(B,B,B))
                        # exec('STA[AP[i].ID_Of_STAs_{}[j]].Rx_power_dBm_{}=dBi+dBi_Rx+AP[i].P_{}-PL(AP[i].Dis_Between_AP_and_STAs[index_sta],f)'.format(B,B,B))#in dBm

                        exec('STA[AP[i].ID_Of_STAs_{}[j]].Rx_power_W_{}=10**(STA[AP[i].ID_Of_STAs_{}[j]].Rx_power_dBm_{}/10)/1000'.format(B,B,B,B)) #in W

                        exec('temp=STA[AP[i].ID_Of_STAs_{}[j]].Rx_power_W_{}/(STA[AP[i].ID_Of_STAs_{}[j]].Total_Interference_{}+P_noise_W)'.format(B,B,B,B))
                        XX=locals()['temp']

                        exec('STA[AP[i].ID_Of_STAs_{}[j]].SINR_{}=10*np.log10(XX)'.format(B,B))
                        # STA[AP[i].ID_Of_STAs[j]].DataRate_DL=STA[AP[i].ID_Of_STAs[j]].NumOfRU*Wsc*np.log2(1+STA[AP[i].ID_Of_STAs[j]].SINR)/1000000
                        
                        
                        exec('STA[AP[i].ID_Of_STAs_{}[j]].DataRate_DL_{}=STA[AP[i].ID_Of_STAs_{}[j]].NumOfRU_{}*Wsc*Get_pbs_per_Hz(STA[AP[i].ID_Of_STAs_{}[j]].SINR_{})/1000/1000'.format(B,B,B,B,B,B))
                        # print('test---')
                        # print(STA[AP[i].ID_Of_STAs_24G[j]].NumOfRU_24G)
                        # input()
                    exec('len_g=len(AP[i].Group_{})'.format(B))
                    Len_G=locals()['len_g']
                    for g in range(Len_G):
                        # max_dis = 0
                        # for sta in range(len(AP[i].Group[g])):
                        #     if STA[AP[i].Group[g][sta]].Dis_to_AP>max_dis:
                        #         # print('a>b', STA[AP[i].Group[g][sta]].Dis_to_AP, max_dis)
                        #         max_dis=STA[AP[i].Group[g][sta]].Dis_to_AP
                                # print('max_dis=', max_dis)

                        exec('lengg=len(AP[i].Group_{}[g])'.format(B))
                        LenGG=locals()['lengg']
                        for sta in range(LenGG):
                            exec('POWER=AP[i].P_{}'.format(B))
                            Power=locals()['POWER']

                            exec('DIS=STA[AP[i].Group_{}[g][sta]].Dis_to_AP'.format(B))
                            Diss=locals()['DIS']

#add two items:G_arr_sta=10*np.log10(2), G_beam24G_sta, etc.
                            exec('Rx_power_dBmm = AP[i].P_{} +G_arr_sta+G_beam{}_sta- PL(STA[AP[i].Group_{}[g][sta]].Dis_to_AP, f)'.format(B,B,B))
                            Rx_power_dBm=locals()['Rx_power_dBmm']

                            # Rx_power_dBm = AP[i].P - PL(STA[AP[i].Group[g][sta]].Dis_to_AP, f)  # in dBm

                            Rx_power_W = 10 ** (Rx_power_dBm / 10) / 1000  # in W

                            exec('TOTAL_Int=AP[i].Total_Interference_{}'.format(B))
                            total_ints=locals()['TOTAL_Int']

                            exec('X_temp = Rx_power_W / (AP[i].Total_Interference_{} + P_noise_W)'.format(B))
                            X=locals()['X_temp']
                            # X = Rx_power_W / (AP[i].Total_Interference + P_noise_W)
                            # print()
                            # print('Total interference:', total_ints)
                            #
                            # print('X=',X)
                            SINR = 10 * np.log10(X)

                            if SINR <0:
                                print("SINR<0!!")
                                print('Debug error:')
                                print('SINR:',SINR)
                                print('Band:',B)
                                print('TX power dBm:', Power)
                                print('Rx_power_dBm:',Rx_power_dBm)
                                print('Rx_power_w=',Rx_power_W)
                                print('Total interference:',total_ints)
                                print('Distance:',Diss)


                                # print (SINR, B)
                                # input()

                            exec('AP[i].SINR_{}.append(SINR)'.format(B))
                            # AP[i].SINR.append(SINR)
                            exec('STA[AP[i].Group_{}[g][sta]].SINR_UP_{}=SINR'.format(B,B))
                            # STA[AP[i].Group[g][sta]].SINR_UP=SINR
                            # STA[AP[i].ID_Of_STAs[j]].DataRate_DL=STA[AP[i].ID_Of_STAs[j]].NumOfRU*Wsc*np.log2(1+STA[AP[i].ID_Of_STAs[j]].SINR)/1000000
                            
                            
                            exec('STA[AP[i].Group_{}[g][sta]].DataRate_UL_{} = STA[AP[i].Group_{}[g][sta]].NumOfRU_{} * Wsc *Get_pbs_per_Hz(SINR) / 1000 / 1000'.format(B,B,B,B))
                            # STA[AP[i].Group[g][sta]].DataRate_UL = STA[AP[i].Group[g][sta]].NumOfRU * Wsc * Get_pbs_per_Hz(SINR) / 1000 / 1000


    # input()

    for i in range(len(AP)):
        # print('i=',i)
        ttt=[]
        nnn=[]
        for B in Band:
            if B=='24G':
                T_SIFS = 10 * 10 ** (-6)
            if B=='5G1':
                T_SIFS = 16 * 10 ** (-6)
            if B=='5G2':
                T_SIFS = 16 * 10 ** (-6)
            if B=='6G':
                T_SIFS = 16 * 10 ** (-6)
            T_DL = T_DIFS + T_backoff + T_dl + T_SIFS + T_OFDMA
            T_UL = T_DIFS + T_backoff + T_TF + 2 * T_SIFS + T_ul + T_MBA

            exec('Round_time_temp = AP[i].Round_{}*(T_UL+T_DL)'.format(B))
            Round_time=locals()['Round_time_temp']
            # print('Round_time',Round_time)
            # ttt.append(Round_time)
            # input()
            # Round_time = AP[i].Round*(T_UL+T_DL)
            exec('lenNeighbor1=len(AP[i].NeiInt_{})'.format(B))
            lenNeighbor=locals()['lenNeighbor1']
            # nnn.append(lenNeighbor)
            # print('band=',B)
            # print('num of neighbor',lenNeighbor)
            # print('len=',len(AP[i].NeiInt_24G))
            # print('AP neighbor list=', AP[i].NeiInt_24G)
            if lenNeighbor>0:
                for j in range(lenNeighbor):
                    # print('j=',j)
                    # print('AP neighbor list=',AP[i].NeiInt_24G)
                    #
                    # print('round of neighbor=',AP[AP[i].NeiInt_24G[j]].Round_24G*(T_UL+T_DL))
                    # exec('Round_time = Round_time+AP[AP[i].NeiInt_{}[j]].Round_{}*(T_UL+T_DL)'.format(B,B))
                    # print(Round_time)
                    # input()

                    if B=='24G':
                        Round_time = Round_time + AP[AP[i].NeiInt_24G[j]].Round_24G * (T_UL + T_DL)
                    if B=='5G1':
                        Round_time = Round_time + AP[AP[i].NeiInt_5G1[j]].Round_5G1 * (T_UL + T_DL)
                    if B=='5G2':
                        Round_time = Round_time + AP[AP[i].NeiInt_5G2[j]].Round_5G2 * (T_UL + T_DL)
                    if B=='6G':
                        Round_time = Round_time + AP[AP[i].NeiInt_6G[j]].Round_6G * (T_UL + T_DL)
                    # print('non exec')
                    # print(Round_time)
                    # input()



                    # Round_time = Round_time+AP[AP[i].NeiInt[j]].Round*(T_UL+T_DL)
            # if lenNeighbor==0:
            #     exec('Round_time = Round_time+AP[AP[i].NeiInt_{}[j]].Round_{}*(T_UL+T_DL)'.format(B, B))
            exec('AP[i].Total_time_{}=Round_time'.format(B))


    # rou=0
    # Ro=[]
    # NuSTA=[]
    # sta=0
    # for i in range(len(AP)):
    #     Ro.append(AP[i].Round_24G)
    #     NuSTA.append(AP[i].Num_Of_STAs)
    #     rou=rou+AP[i].Round_24G
    #     sta=sta+AP[i].Num_Of_STAs
    # print('numOfAPs:',len(AP))
    # print('numOfrou',rou)
    # print('RoList',Ro)
    # print('NuSTA',NuSTA)
    # print('numOfsta',sta)
    # Plot_fig()
    #
    # input()
    global MaxThroughput
    MaxThroughput = 0
    global MinThroughput
    MinThroughput = inf
    for i in range(len(AP)):
        # print(i)
        for B in Band:

            exec('len_temp10=len(AP[i].ID_Of_STAs_{})'.format(B))
            numStas=locals()['len_temp10']
            for j in range(numStas):
                # count=count+1
                exec('idOfsta=AP[i].ID_Of_STAs_{}[j]'.format(B))
                idd_sta=locals()['idOfsta']

                # T_SIFS = 10 * 10 ** (-6)


                # print(AP[i].Total_time_24G)
                # input()

                exec('debug_t=AP[i].Total_time_{}'.format(B))
                debug_tt=locals()['debug_t']
                if debug_tt==0:
                    print("Total_time = 0!!")
                    print("B=",B)
                    print("AP No.:",i)
                    print('There is a error!')
                    input()
                    
                #++++++++++++++++++++++++++++++++++++++  for throughput verify    
                p=0.99
                NumOfPacket=1000
                print("Obtaining throughput in simulation...")
                exec('Throughput_{}=0'.format(B))
                for pak in range(NumOfPacket):
                    pUL=random.random()
                    pDL=random.random()
                    if pUL<=p and pDL<=p:
                        exec('STA[idd_sta].Throughput_{}=(STA[idd_sta].DataRate_DL_{}*T_dl+STA[idd_sta].DataRate_UL_{}*T_ul)/AP[i].Total_time_{}'.format(B,B,B,B))
                        exec('Throughput_{}=Throughput_{}+STA[idd_sta].Throughput_{}'.format(B,B,B))
            
                    if pUL>p and pDL>p:
                        exec('STA[idd_sta].Throughput_{}=0'.format(B))
                        exec('Throughput_{}=Throughput_{}+STA[idd_sta].Throughput_{}'.format(B,B,B))
            
                    if pUL<=p and pDL>p:
                        exec('STA[idd_sta].Throughput_{}=(0*T_dl+STA[idd_sta].DataRate_UL_{}*T_ul)/AP[i].Total_time_{}'.format(B,B,B))
                        exec('Throughput_{}=Throughput_{}+STA[idd_sta].Throughput_{}'.format(B,B,B))
            
                    if pUL>p and pDL<=p:
                        exec('STA[idd_sta].Throughput_{}=(STA[idd_sta].DataRate_DL_{}*T_dl+0*T_ul)/AP[i].Total_time_{}'.format(B,B,B))
                        exec('Throughput_{}=Throughput_{}+STA[idd_sta].Throughput_{}'.format(B,B,B))
            
                print("Simulation is doned!")    
                exec('STA[idd_sta].Throughput_sim_{}=Throughput_{}/NumOfPacket'.format(B, B))
                exec('STA[idd_sta].Throughput_{}=(STA[idd_sta].DataRate_DL_{}*T_dl+STA[idd_sta].DataRate_UL_{}*T_ul)/AP[i].Total_time_{}'.format(B, B, B, B))

               


    idd_sta=0
    for idd_sta in range(len(STA)):
        STA[idd_sta].Throughput=STA[idd_sta].Throughput_24G+STA[idd_sta].Throughput_5G1+STA[idd_sta].Throughput_5G2+STA[idd_sta].Throughput_6G
        
        
        
        #++++++++++++++++++++++++++++++++++++++  for throughput verify
        STA[idd_sta].Throughput_sim = STA[idd_sta].Throughput_sim_24G + STA[idd_sta].Throughput_sim_5G1 + STA[idd_sta].Throughput_sim_5G2 + STA[idd_sta].Throughput_sim_6G
        
        if STA[idd_sta].Throughput==0:
            print("Throughput=0!!")
            print('STA =',idd_sta)
            print('AP =',STA[idd_sta].IDOfAP)
            print('DL SINR =',STA[idd_sta].SINR)
            print('UL SINR =', STA[idd_sta].SINR_UP)
            print('Throughput error!')
            # Plot_fig()
            # input()
        # else:
        #     print(count)
        Throughput_of_STAs.append(STA[idd_sta].Throughput)

        # print(Throughput_of_STAs)
        # input()

        if STA[idd_sta].Throughput>MaxThroughput:
            MaxThroughput=STA[idd_sta].Throughput
            APNo=STA[idd_sta].IDOfAP
            idofsta=idd_sta

            # print('STA[idd_sta].IDOfAP',STA[idd_sta].IDOfAP)
            # input()


            ChannelNo24G = AP[STA[idd_sta].IDOfAP].C_24G
            ChannelNo5G1 = AP[STA[idd_sta].IDOfAP].C_5G1
            ChannelNo5G2 = AP[STA[idd_sta].IDOfAP].C_5G2
            ChannelNo6G = AP[STA[idd_sta].IDOfAP].C_6G
            IDOfstas_24G=AP[STA[idd_sta].IDOfAP].ID_Of_STAs_24G
            IDOfstas_5G1 = AP[STA[idd_sta].IDOfAP].ID_Of_STAs_5G1
            IDOfstas_5G2 = AP[STA[idd_sta].IDOfAP].ID_Of_STAs_5G2
            IDOfstas_6G = AP[STA[idd_sta].IDOfAP].ID_Of_STAs_6G

        if STA[idd_sta].Throughput<MinThroughput:

            MinThroughput=STA[idd_sta].Throughput
            APNo1 = STA[idd_sta].IDOfAP
            idofsta1 = idd_sta
            ChannelNo24G1 = AP[STA[idd_sta].IDOfAP].C_24G
            ChannelNo5G11= AP[STA[idd_sta].IDOfAP].C_5G1
            ChannelNo5G21 = AP[STA[idd_sta].IDOfAP].C_5G2
            ChannelNo6G1 = AP[STA[idd_sta].IDOfAP].C_6G
            IDOfstas_24G1 = AP[STA[idd_sta].IDOfAP].ID_Of_STAs_24G
            IDOfstas_5G11 = AP[STA[idd_sta].IDOfAP].ID_Of_STAs_5G1
            IDOfstas_5G21 = AP[STA[idd_sta].IDOfAP].ID_Of_STAs_5G2
            IDOfstas_6G1 = AP[STA[idd_sta].IDOfAP].ID_Of_STAs_6G

    # print('MaxThroughput--',MaxThroughput)
    # print('MinThroughput--',MinThroughput)
    
    # input()
    # print('APNo',APNo)
    # print('ChannelNo24G',ChannelNo24G)
    # print('ChannelNo5G1',ChannelNo5G1)
    # print('ChannelNo5G2',ChannelNo5G2)
    # print('ChannelNo6G',ChannelNo6G)
    # print('IDOfstas_24G',IDOfstas_24G)
    # print('IDOfstas_5G1', IDOfstas_5G1)
    # print('IDOfstas_5G2', IDOfstas_5G2)
    # print('IDOfstas_6G', IDOfstas_6G)
    # print('idofsta',idofsta)
    # print('---------')

    # For savety margin test
    if SafetyMargin==1:
        print('MinThroughput',MinThroughput)
        print('For safety margin test has done!')
        input()

    # print('APNo1',APNo1)
    # print('ChannelNo24G1',ChannelNo24G1)
    # print('ChannelNo5G11',ChannelNo5G11)
    # print('ChannelNo5G21',ChannelNo5G21)
    # print('ChannelNo6G1',ChannelNo6G1)
    # print('IDOfstas_24G1',IDOfstas_24G1)
    # print('IDOfstas_5G11', IDOfstas_5G11)
    # print('IDOfstas_5G21', IDOfstas_5G21)
    # print('IDOfstas_6G1', IDOfstas_6G1)
    # print('idofsta1',idofsta1)

                        #Calculate the throughput of STAs
                        # Ti=(T_DL+T_UL)*len(AP[i].Group)
                        # STA[AP[i].Group[g][sta]].Throughput=(STA[AP[i].Group[g][sta]].DataRate_UL*T_ul+STA[AP[i].Group[g][sta]].DataRate_DL*T_dl)/(Ti*)

    #-------------------------------------------------------
    # atest=0
    # for i in range(len(AP)):
    #     print("2.4GC",AP[i].C_24G)
    #     print("5G1C", AP[i].C_5G1)
    #     print("5G2C", AP[i].C_5G2)
    #     print("6GC", AP[i].C_6G)
    #
    #     print("2.4GRU",AP[i].NumRUs_24G)
    #     print("5G1RU", AP[i].NumRUs_5G1)
    #     print("5G2RU", AP[i].NumRUs_5G2)
    #     print("6GRU", AP[i].NumRUs_6G)
    #
    #     print("2.4Ggroup",len(AP[i].Group_24G))
    #     print("5G1group", len(AP[i].Group_5G1))
    #     print("5G2group", len(AP[i].Group_5G2))
    #     print("6Ggroup", len(AP[i].Group_6G))
    #
    #     print('NumOfstations=',AP[i].Num_Of_STAs)
    #     input()

    return

def RUassignment_APCoordination_modelVerify_withoutbeamforming():
    #Determine the channel width
    # print('begin RU assignment')
    # input()

    # for i in range(len(AP)):
    #     print('AP_24G_C=',AP[i].C_24G)
    #     print('AP_5G1_C=', AP[i].C_5G1)
    #     print('AP_5G2_C=', AP[i].C_5G2)
    #     print('AP_6G_C=', AP[i].C_6G)
    #     input()

    for i in range(len(AP)):
        if AP[i].C_24G in C20M:
            AP[i].NumRUs_24G = 9
            # AP[i].RUperUser_24G = AP[i].NumRUs_24G / AP[i].Num_Of_STAs
        if AP[i].C_24G in C40M:
            AP[i].NumRUs_24G = 18
            # AP[i].RUperUser_24G = AP[i].NumRUs_24G / AP[i].Num_Of_STAs
        if AP[i].C_24G in C80M:
            AP[i].NumRUs_24G = 36
            # AP[i].RUperUser_24G = AP[i].NumRUs_24G / AP[i].Num_Of_STAs
        if AP[i].C_24G in C160M:
            AP[i].NumRUs_24G = 72
            # AP[i].RUperUser_24G = AP[i].NumRUs_24G / AP[i].Num_Of_STAs
        if AP[i].C_24G in C320M:
            AP[i].NumRUs_24G = 144
            # AP[i].RUperUser_24G = AP[i].NumRUs_24G / AP[i].Num_Of_STAs

        if AP[i].C_5G1 in C20M:
            AP[i].NumRUs_5G1 = 9
            # AP[i].RUperUser_5G1 = AP[i].NumRUs_5G1 / AP[i].Num_Of_STAs
        if AP[i].C_5G1 in C40M:
            AP[i].NumRUs_5G1 = 18
            # AP[i].RUperUser_5G1 = AP[i].NumRUs_5G1 / AP[i].Num_Of_STAs
        if AP[i].C_5G1 in C80M:
            AP[i].NumRUs_5G1 = 36
            # AP[i].RUperUser_5G1 = AP[i].NumRUs_5G1 / AP[i].Num_Of_STAs
        if AP[i].C_5G1 in C160M:
            AP[i].NumRUs_5G1 = 72
            # AP[i].RUperUser_24G = AP[i].NumRUs_24G / AP[i].Num_Of_STAs
        if AP[i].C_5G1 in C320M:
            AP[i].NumRUs_5G1 = 144
            # AP[i].RUperUser_24G = AP[i].NumRUs_24G / AP[i].Num_Of_STAs

        if AP[i].C_5G2 in C20M:
            AP[i].NumRUs_5G2 = 9
            # AP[i].RUperUser_5G2 = AP[i].NumRUs_5G2 / AP[i].Num_Of_STAs
        if AP[i].C_5G2 in C40M:
            AP[i].NumRUs_5G2 = 18
            # AP[i].RUperUser_5G2 = AP[i].NumRUs_5G2 / AP[i].Num_Of_STAs
        if AP[i].C_5G2 in C80M:
            AP[i].NumRUs_5G2 = 36
            # AP[i].RUperUser_5G2 = AP[i].NumRUs_5G2 / AP[i].Num_Of_STAs
        if AP[i].C_5G2 in C160M:
            AP[i].NumRUs_5G2 = 72
            # AP[i].RUperUser_24G = AP[i].NumRUs_24G / AP[i].Num_Of_STAs
        if AP[i].C_5G2 in C320M:
            AP[i].NumRUs_5G2 = 144
            # AP[i].RUperUser_24G = AP[i].NumRUs_24G / AP[i].Num_Of_STAs

        if AP[i].C_6G in C20M:
            AP[i].NumRUs_6G = 9
            # AP[i].RUperUser_6G = AP[i].NumRUs_6G / AP[i].Num_Of_STAs
        if AP[i].C_6G in C40M:
            AP[i].NumRUs_6G = 18
            # AP[i].RUperUser_6G = AP[i].NumRUs_6G / AP[i].Num_Of_STAs
        if AP[i].C_6G in C80M:
            AP[i].NumRUs_6G = 36
            # AP[i].RUperUser_6G = AP[i].NumRUs_6G / AP[i].Num_Of_STAs
        if AP[i].C_6G in C160M:
            AP[i].NumRUs_6G = 72
            # AP[i].RUperUser_24G = AP[i].NumRUs_24G / AP[i].Num_Of_STAs
        if AP[i].C_6G in C320M:
            AP[i].NumRUs_6G = 144
            # AP[i].RUperUser_24G = AP[i].NumRUs_24G / AP[i].Num_Of_STAs

    RUlist24G=[]
    RUlist5G1 = []
    RUlist5G2 = []
    RUlist6G = []
    for i in range(len(AP)):
        RUlist24G.append(AP[i].NumRUs_24G)
        RUlist5G1.append(AP[i].NumRUs_5G1)
        RUlist5G2.append(AP[i].NumRUs_5G2)
        RUlist6G.append(AP[i].NumRUs_6G)
        # print("Current RU:")
        # print('i=', i)
        # print(AP[i].NumRUs_6G)
    # print(RUlist24G)
    # print(RUlist5G1)
    # print(RUlist5G2)
    # print(RUlist6G)
    # input()

    Flag24G=0
    Flag5G1=0
    Flag5G2 = 0
    Flag6G = 0
    while(1):
        for i in range(len(AP)):
            # if AP[i].C_24G in C20M:
            #     AP[i].NumRUs_24G=9
            AP[i].RUperUser_24G=AP[i].NumRUs_24G/AP[i].Num_Of_STAs
            # if AP[i].C_24G in C40M:
            #     AP[i].NumRUs_24G=18
            # AP[i].RUperUser_24G = AP[i].NumRUs_24G / AP[i].Num_Of_STAs
            # if AP[i].C_24G in C80M:
            #     AP[i].NumRUs_24G=36
            # AP[i].RUperUser_24G = AP[i].NumRUs_24G / AP[i].Num_Of_STAs

            # if AP[i].C_5G1 in C20M:
            #     AP[i].NumRUs_5G1=9
            AP[i].RUperUser_5G1 = AP[i].NumRUs_5G1 / AP[i].Num_Of_STAs
            # if AP[i].C_5G1 in C40M:
            #     AP[i].NumRUs_5G1=18
            # AP[i].RUperUser_5G1 = AP[i].NumRUs_5G1 / AP[i].Num_Of_STAs
            # if AP[i].C_5G1 in C80M:
            #     AP[i].NumRUs_5G1=36
            # AP[i].RUperUser_5G1 = AP[i].NumRUs_5G1 / AP[i].Num_Of_STAs

            # if AP[i].C_5G2 in C20M:
            #     AP[i].NumRUs_5G2=9
            AP[i].RUperUser_5G2 = AP[i].NumRUs_5G2 / AP[i].Num_Of_STAs
            # if AP[i].C_5G2 in C40M:
            #     AP[i].NumRUs_5G2=18
            # AP[i].RUperUser_5G2 = AP[i].NumRUs_5G2 / AP[i].Num_Of_STAs
            # if AP[i].C_5G2 in C80M:
            #     AP[i].NumRUs_5G2=36
            # AP[i].RUperUser_5G2 = AP[i].NumRUs_5G2 / AP[i].Num_Of_STAs

            # if AP[i].C_6G in C20M:
            #     AP[i].NumRUs_6G=9
            AP[i].RUperUser_6G = AP[i].NumRUs_6G / AP[i].Num_Of_STAs
            # if AP[i].C_6G in C40M:
            #     AP[i].NumRUs_6G=18
            # AP[i].RUperUser_6G = AP[i].NumRUs_6G / AP[i].Num_Of_STAs
            # if AP[i].C_6G in C80M:
            #     AP[i].NumRUs_6G=36
            # AP[i].RUperUser_6G = AP[i].NumRUs_6G / AP[i].Num_Of_STAs

            # print(AP[i].NumRUs_6G)
            # print('i=',i)
            # print('test')
            # input()

        MaxRU_24G=0
        MaxRU_5G1 = 0
        MaxRU_5G2 = 0
        MaxRU_6G = 0
        MinRU_24G=inf
        MinRU_5G1 = inf
        MinRU_5G2 = inf
        MinRU_6G = inf
        Mini_24G=inf
        Mini_5G1 = inf
        Mini_5G2 = inf
        Mini_6G = inf
        Maxi_24G=inf
        Maxi_5G1 = inf
        Maxi_5G2 = inf
        Maxi_6G = inf
        for i in range(len(AP)):

            if AP[i].RUperUser_24G > MaxRU_24G:
                MaxRU_24G=AP[i].RUperUser_24G
                Maxi_24G=i
            if AP[i].RUperUser_5G1 > MaxRU_5G1:
                MaxRU_5G1=AP[i].RUperUser_5G1
                Maxi_5G1=i
            if AP[i].RUperUser_5G2 > MaxRU_5G2:
                MaxRU_5G2=AP[i].RUperUser_5G2
                Maxi_5G2=i
            if AP[i].RUperUser_6G > MaxRU_6G:
                MaxRU_6G=AP[i].RUperUser_6G
                Maxi_6G=i

            if AP[i].RUperUser_24G < MinRU_24G:
                MinRU_24G=AP[i].RUperUser_24G
                Mini_24G=i
            if AP[i].RUperUser_5G1 < MinRU_5G1:
                MinRU_5G1=AP[i].RUperUser_5G1
                Mini_5G1=i
            if AP[i].RUperUser_5G2 < MinRU_5G2:
                MinRU_5G2=AP[i].RUperUser_5G2
                Mini_5G2=i
            if AP[i].RUperUser_6G < MinRU_6G:
                MinRU_6G=AP[i].RUperUser_6G
                Mini_6G=i
        # print("Max24G",Maxi_24G)
        # print("Max5G1", Maxi_5G1)
        # print("Max5G2", Maxi_5G2)
        # print("Max6G", Maxi_6G)
        #
        # print("Min24G", Mini_24G)
        # print("Min5G1", Mini_5G1)
        # print("Min5G2", Mini_5G2)
        # print("Min6G", Mini_6G)
        #
        # print("Max24G_RU",AP[Maxi_24G].NumRUs_24G)
        # print("Max5G1_RU", AP[Maxi_5G1].NumRUs_5G1)
        # print("Max5G2_RU", AP[Maxi_5G2].NumRUs_5G2)
        # print("Max6G_RU", AP[Maxi_6G].NumRUs_6G)
        #
        # print("Min24G_RU", AP[Mini_24G].NumRUs_24G)
        # print("Min5G1_RU", AP[Mini_5G1].NumRUs_5G1)
        # print("Min5G2_RU", AP[Mini_5G2].NumRUs_5G2)
        # print("Min6G_RU", AP[Mini_6G].NumRUs_6G)

        if AP[Maxi_24G].NumRUs_24G-AP[Mini_24G].NumRUs_24G>=2:
            AP[Maxi_24G].NumRUs_24G=AP[Maxi_24G].NumRUs_24G-1
            AP[Mini_24G].NumRUs_24G=AP[Mini_24G].NumRUs_24G+1
        else:
            Flag24G=1
        if AP[Maxi_6G].NumRUs_6G-AP[Mini_6G].NumRUs_6G>=2:
            AP[Maxi_6G].NumRUs_6G=AP[Maxi_6G].NumRUs_6G-1
            AP[Mini_6G].NumRUs_6G=AP[Mini_6G].NumRUs_6G+1
        else:
            Flag6G=1
        if AP[Maxi_5G1].NumRUs_5G1-AP[Mini_5G1].NumRUs_5G1>=2:
            AP[Maxi_5G1].NumRUs_5G1=AP[Maxi_5G1].NumRUs_5G1-1
            AP[Mini_5G1].NumRUs_5G1=AP[Mini_5G1].NumRUs_5G1+1
        else:
            Flag5G1=1
        if AP[Maxi_5G2].NumRUs_5G2-AP[Mini_5G2].NumRUs_5G2>=2:
            AP[Maxi_5G2].NumRUs_5G2=AP[Maxi_5G2].NumRUs_5G2-1
            AP[Mini_5G2].NumRUs_5G2=AP[Mini_5G2].NumRUs_5G2+1
        else:
            Flag5G2=1

        # print('AP coordination:')
        #
        # print("Max6G_RU", AP[Maxi_6G].NumRUs_6G)
        # print("Min6G_RU", AP[Mini_6G].NumRUs_6G)
        # input()
        if Flag24G==1 and Flag5G1==1  and Flag5G2==1 and Flag6G==1:
            break

    RUlist24G = []
    RUlist5G1 = []
    RUlist5G2 = []
    RUlist6G = []
    for i in range(len(AP)):
        RUlist24G.append(AP[i].NumRUs_24G)
        RUlist5G1.append(AP[i].NumRUs_5G1)
        RUlist5G2.append(AP[i].NumRUs_5G2)
        RUlist6G.append(AP[i].NumRUs_6G)
        # print("Current RU:")
        # print('i=', i)
        # print(AP[i].NumRUs_6G)
    # print(RUlist24G)
    # print(RUlist5G1)
    # print(RUlist5G2)
    # print(RUlist6G)
    # input()








        #
        # AP[i].RUperUser_24G
        # AP[i].RUperUser_5G1
        # AP[i].RUperUser_5G2
        # AP[i].RUperUser_6G



    for i in range(len(AP)):#2.4G
        # print('number of AP:',i)
        for B in Band:
            if Band.index(B)==0:
                FF=[20,40]
                f=fc_2dot4G
            if Band.index(B)==1:
                FF=[20,40,80,160]
                f=fc_5GI
            if Band.index(B)==2:
                FF=[20,40,80]
                f=fc_5GII
            if Band.index(B)==3:
                FF=[20,40,80,160,320]
                f=fc_6G

            for F in FF:
                # Fre = []
                exec('Fre=Frequence{}_{}'.format(F,B))
                Freq=locals()['Fre']
                exec('chh=AP[i].C_{}'.format(B))
                ch=locals()['chh']
                # print(ch)
                # input()
                if ch in Freq:
                # if AP[i].C_24G in Freq:
                    if F==20:
                        #Maximum number of 106-tone RUs
                        max_num_of_RU=2
                    if F==40:
                        max_num_of_RU = 4
                    if F==80:
                        max_num_of_RU = 8
                    if F==160:
                        max_num_of_RU = 16
                    if F==320:
                        max_num_of_RU = 32

                    #Determine the Maximum number of 106-tone RUs
                    exec('max_num_of_RUU=AP[i].NumRUs_{}'.format(B))
                    max_num_of_RUuu = locals()['max_num_of_RUU']
                    # max_num_of_RUuu=15
                    max_num_of_RU=max_num_of_RUuu//4
                    
                    # print(max_num_of_RU)
                    # input()
                    
                    RU_rem=max_num_of_RUuu%4

                    # print(max_num_of_RUuu)
                    # print(max_num_of_RU)
                    # print(RU_rem)
                    # input()

                    RUset_Full=[]
                    RUset_Part =[]
                    for ru in range(0,max_num_of_RU):
                        RUset_Full.append(4*26+2)
                    if RU_rem !=0:
                        for j in range(0,RU_rem):
                            # print('j=',j)
                            # print('max_num_of_ru=',max_num_of_RU)
                            # print('AP_c=',ch)
                            ind=j%max_num_of_RU

                            RUset_Full[ind]=RUset_Full[ind]+26

                        # for rem_ru in range(0,RU_rem):
                        #     print('RUset_full')
                        #     print(RUset_Full)
                        #     RUset_Full[rem_ru]=RUset_Full[rem_ru] + 26
                        #     print('rem_ru', rem_ru)
                    # print('======')
                    # print('RUset_full',RUset_Full)
                    # print('max_num_of_RU',max_num_of_RU)
                    # print('======')
                    # input()


                    # print(B)
                    # print('max_num_of_RU',max_num_of_RU)
                    # print('max_num_of_RU', int(max_num_of_RU))
                    # input()
                    # b=AP[i].Num_Of_STAs//Max_transmissions #Quotient,
                    # a=AP[i].Num_Of_STAs%Max_transmissions #Remainder





                    Max_transmissions=Num_of_antennas*max_num_of_RU
                    # b=AP[i].Num_Of_STAs//Max_transmissions #Quotient,
                    # a=AP[i].Num_Of_STAs%Max_transmissions #Remainder
                    exec('numSTAs=len(AP[i].ID_Of_STAs_{})'.format(B))
                    NumSTAs=locals()['numSTAs']
                    # print(NumSTAs,len(AP[i].ID_Of_STAs_24G))
                    # input()
                    round=int(np.ceil(NumSTAs/Max_transmissions))
                    # print('i',i)
                    # print('B',B)
                    # print('round:',round)
                    exec('AP[i].Round_{}=round'.format(B))


                    for r in range(round):
                        exec('AP[i].Group_{}.append([])'.format(B))

                    # The stations should be ordered
                    for j in range(NumSTAs):
                        r = j % round
                        exec('AP[i].Group_{}[r].append(AP[i].ID_Of_STAs_{}[j])'.format(B,B))
                        # print(AP[i].Group_24G[r])
                        # input()
                    # input()

                    for g in range(round):
                        # Sort the stations according to the distances

                        # list = [34, 23, 77, 54, 99, 10]
                        # sorted_id = sorted(range(len(list)), key=lambda x: list[x], reverse=True)
                        # print(sorted_id)



                        exec('len1=len(AP[i].Group_{}[g])'.format(B))
                        len_temp=locals()['len1']
                        b=len_temp//Num_of_antennas#Quotient
                        a =len_temp%Num_of_antennas# Remainder
                        if a==0:
                            a_temp=0
                            # print('a=0')

                            # exec('listt=AP[i].Group_{}[g]'.format(B))
                            # list_temp = locals()['listt']
                            # # print('list_temp',list_temp)
                            # Diss_temp=[]
                            # for d in range(0,len(list_temp)):
                            #     diss=((AP[i].x-STA[list_temp[d]].x)**2+(AP[i].y-STA[list_temp[d]].y)**2)**(0.5)
                            #     # print(list_temp[d])
                            #     # input()
                            #     Diss_temp.append(diss)
                            # # print('dis_temp',Diss_temp)
                            #
                            #
                            #
                            #
                            # for jj in range(len_temp):
                            #     temp_rem=jj%max_num_of_RU
                            #     #exec('STA[AP[i].Group_{}[g][j]].NumOfRU=RU{}_{}[temp_rem]'.format(B,F,max_num_of_RU))
                            #     # exec('STA[AP[i].Group_{}[g][j]].NumOfRU_{}=RU{}_{}[temp_rem]'.format(B, B, F, max_num_of_RU))
                            #
                            #     # print('len_temp',len_temp)
                            #     # print('RUset_Full',RUset_Full)
                            #     # print('max_num_of_RU', max_num_of_RU)
                            #     # print('jj',jj)
                            #     # print('temp_rem',temp_rem)
                            #     #
                            #     # print('g',g)
                            #     #
                            #     # exec('group=AP[i].Group_{}[g][jj]'.format(B))
                            #     # group_temp = locals()['group']
                            #     # exec('groupsta=AP[i].Group_{}[g]'.format(B))
                            #     # group_temp_sta = locals()['groupsta']
                            #     # print('group-sta',group_temp)
                            #     # print('group',group_temp_sta)
                            #
                            #
                            #
                            #     # input()
                            #     exec('STA[AP[i].Group_{}[g][jj]].NumOfRU_{}=RUset_Full[temp_rem]'.format(B, B))
                        if a==0 or a >0:

                            # exec('listt=AP[i].Group_{}[g]'.format(B))
                            # list_temp = locals()['listt']
                            # print('list_temp', list_temp)
                            # Diss_temp = []
                            # for d in range(0, len(list_temp)):
                            #     diss = ((AP[i].x - STA[list_temp[d]].x) ** 2 + (
                            #                 AP[i].y - STA[list_temp[d]].y) ** 2) ** (0.5)
                            #     # print(list_temp[d])
                            #     # input()
                            #     Diss_temp.append(diss)
                            # print('dis_temp', Diss_temp)

                            SS=[]
                            # print('a!=0')
                            for k in range(Num_of_antennas):
                                SS.append([])
                            for j in range(len_temp):
                                temp_rem=j%Num_of_antennas
                                exec('SS[temp_rem].append(AP[i].Group_{}[g][j])'.format(B))
                                # SS[temp_rem].append(AP[i].Group_24G[g][j])
                                # print('temp_rem',temp_rem)
                                # print('SS[temp]',SS)
                                # input()
                            len_ss=np.zeros(Num_of_antennas)
                            for j in range(Num_of_antennas):
                                len_ss[j]=len(SS[j])
                            max_RU=max(len_ss)

                            # print(SS)




                            for Max_RU in range(1,max_num_of_RU+1):
                                # print('max_RU',max_RU)
                                # print('Max_RU',Max_RU)
                                # input()

                                if max_RU==Max_RU: # Why use this condition? Because we need to determine the RU set!
                                    # print('part_index')
                                    # print(max_RU)
                                    # input()

                                    for j in range(Num_of_antennas):

                                        # Sort the stations for each spatial stream. Big RU for farther station
                                        exec('listt=SS[j]'.format(B))
                                        list_temp = locals()['listt']
                                        # print('list_temp', list_temp)
                                        Diss_temp = []
                                        for d in range(0, len(list_temp)):
                                            diss = ((AP[i].x - STA[list_temp[d]].x) ** 2 + (
                                                    AP[i].y - STA[list_temp[d]].y) ** 2) ** (0.5)
                                            # print(list_temp[d])
                                            # input()
                                            Diss_temp.append(diss)
                                        # print('dis_temp', Diss_temp)


                                        sorted_id = sorted(range(len(Diss_temp)), key=lambda x: Diss_temp[x], reverse=True)
                                        # print(sorted_id)
                                        # input()



                                        if len(SS[j]) != 0:

                                            exec('max_num_of_RUUT=AP[i].NumRUs_{}'.format(B))
                                            max_num_of_RUuuT = locals()['max_num_of_RUUT']
                                            # max_num_of_RUuu=15
                                            # Max_RU here is the number of RU subsets
                                            # max_num_of_RU here is the number of 26-tone RUs
                                            max_num_of_RUt = max_num_of_RUuuT // len(SS[j])
                                            RU_remt = max_num_of_RUuuT % len(SS[j])

                                            # print('max_RU',Max_RU)
                                            # print('max_num_of_RUuuT',max_num_of_RUuuT)
                                            # print('max_num_of_RU',max_num_of_RU)
                                            # print('RU_rem',RU_rem)
                                            # print('number of AP i:',i)
                                            # input()

                                            RUset_Part = []
                                            for ru in range(0, len(SS[j])):
                                                RU_element = max_num_of_RUt * 26
                                                if max_num_of_RUt * 26 >= 16 * 4 and max_num_of_RUt * 26 < 16 * 8:
                                                    RU_element = max_num_of_RUt * 26 + 2
                                                if max_num_of_RUt * 26 >= 16 * 8:
                                                    RU_element = max_num_of_RUt * 26 + 4
                                                RUset_Part.append(RU_element)
                                            if RU_remt != 0:
                                                for jj in range(0, RU_remt):
                                                    ind = jj % len(SS[j])
                                                    RUset_Part[ind] = RUset_Part[ind] + 26

                                                # for rem_ru in range(0, RU_rem):
                                                #     RUset_Part[rem_ru] = RUset_Part[rem_ru] + 26

                                            # print('RUset_Part', RUset_Part)
                                            # input()

                                            for k in range(len(SS[j])):
                                                # exec('STA[AP[i].ID_Of_STAs[SS[j][k]]].NumOfRU = RU20_{}[k]'.format(Max_RU))
                                                # print(RUset_Part)
                                                # print(RUlist24G)
                                                # print(RUlist5G1)
                                                # print(RUlist5G2)
                                                # print(RUlist6G)
                                                # input()
                                                # exec('STA[SS[j][k]].NumOfRU_{} = RU{}_{}[k]'.format(B,F,Max_RU))
                                                # print('RUset_Part=',RUset_Part)
                                                # print('k=',k)
                                                # print('j=',j)
                                                # print('ss=',SS)
                                                # print('SS[j][k]=',SS[j][k])
                                                # exec('axx=STA[SS[j][k]].NumOfRU_{} = RUset_Part[k]'.format(B))

                                                # exec('STA[SS[j][k]].NumOfRU_{} = RUset_Part[k]'.format(B))
                                                exec('STA[list_temp[sorted_id[k]]].NumOfRU_{} = RUset_Part[k]'.format(B))
                                                # exec('testRU=STA[list_temp[sorted_id[k]]].NumOfRU_{}'.format(B))
                                                # testRUU= locals()['testRU']
                                                # exec('testSTA=list_temp[sorted_id[k]]')
                                                # testSTAa= locals()['testSTA']
                                                # print('testSTAa',testSTAa)
                                                # print('testRUU',testRUU)
                                                # input()

                            # AP[i].Group=AP[i].Group+SS
                           # input()
                        # Calculate the data rate
                    # if AP[i].C in C_2dot4G:
                    # input()
                    # f=fc_2dot4G
                    # if AP[i].C in C_5GI:
                    #     f=fc_5GI
                    #
                    # for j in range(AP[i].Num_Of_STAs):
                    #     P_RX=AP[i].P-PL(AP[i].Dis_Between_AP_and_STAs[j],f)
                    exec('lenSTAs=len(AP[i].ID_Of_STAs_{})'.format(B))
                    LenSTAs=locals()['lenSTAs']
                    for j in range(LenSTAs):
                        # print(AP[i].Dis_Between_AP_and_STAs)
                        # print(AP[i].ID_Of_STAs)
                        # print(len(AP[i].Dis_Between_AP_and_STAs))
                        # print(len(AP[i].ID_Of_STAs))

                        exec('IDSTA=AP[i].ID_Of_STAs_{}[j]'.format(B))
                        idSTA=locals()['IDSTA']
                        index_sta=AP[i].ID_Of_STAs.index(idSTA)

                        exec('STA[AP[i].ID_Of_STAs_{}[j]].Rx_power_dBm_{}=AP[i].P_{}-PL(AP[i].Dis_Between_AP_and_STAs[index_sta],f)'.format(B,B,B))#in dBm

                        exec('STA[AP[i].ID_Of_STAs_{}[j]].Rx_power_W_{}=10**(STA[AP[i].ID_Of_STAs_{}[j]].Rx_power_dBm_{}/10)/1000'.format(B,B,B,B)) #in W

                        exec('temp=STA[AP[i].ID_Of_STAs_{}[j]].Rx_power_W_{}/(STA[AP[i].ID_Of_STAs_{}[j]].Total_Interference_{}+P_noise_W)'.format(B,B,B,B))
                        XX=locals()['temp']

                        exec('STA[AP[i].ID_Of_STAs_{}[j]].SINR_{}=10*np.log10(XX)'.format(B,B))
                        # STA[AP[i].ID_Of_STAs[j]].DataRate_DL=STA[AP[i].ID_Of_STAs[j]].NumOfRU*Wsc*np.log2(1+STA[AP[i].ID_Of_STAs[j]].SINR)/1000000
                        exec('STA[AP[i].ID_Of_STAs_{}[j]].DataRate_DL_{}=STA[AP[i].ID_Of_STAs_{}[j]].NumOfRU_{}*Wsc*Get_pbs_per_Hz(STA[AP[i].ID_Of_STAs_{}[j]].SINR_{})/1000/1000'.format(B,B,B,B,B,B))
                        # print('test---')
                        # input()
                    exec('len_g=len(AP[i].Group_{})'.format(B))
                    Len_G=locals()['len_g']
                    for g in range(Len_G):
                        # max_dis = 0
                        # for sta in range(len(AP[i].Group[g])):
                        #     if STA[AP[i].Group[g][sta]].Dis_to_AP>max_dis:
                        #         # print('a>b', STA[AP[i].Group[g][sta]].Dis_to_AP, max_dis)
                        #         max_dis=STA[AP[i].Group[g][sta]].Dis_to_AP
                                # print('max_dis=', max_dis)

                        exec('lengg=len(AP[i].Group_{}[g])'.format(B))
                        LenGG=locals()['lengg']
                        for sta in range(LenGG):
                            exec('POWER=AP[i].P_{}'.format(B))
                            Power=locals()['POWER']

                            exec('DIS=STA[AP[i].Group_{}[g][sta]].Dis_to_AP'.format(B))
                            Diss=locals()['DIS']

                            exec('Rx_power_dBmm = AP[i].P_{} - PL(STA[AP[i].Group_{}[g][sta]].Dis_to_AP, f)'.format(B,B))
                            Rx_power_dBm=locals()['Rx_power_dBmm']

                            # Rx_power_dBm = AP[i].P - PL(STA[AP[i].Group[g][sta]].Dis_to_AP, f)  # in dBm

                            Rx_power_W = 10 ** (Rx_power_dBm / 10) / 1000  # in W

                            exec('TOTAL_Int=AP[i].Total_Interference_{}'.format(B))
                            total_ints=locals()['TOTAL_Int']

                            exec('X_temp = Rx_power_W / (AP[i].Total_Interference_{} + P_noise_W)'.format(B))
                            X=locals()['X_temp']
                            # X = Rx_power_W / (AP[i].Total_Interference + P_noise_W)
                            # print()
                            # print('Total interference:', total_ints)
                            #
                            # print('X=',X)
                            SINR = 10 * np.log10(X)

                            if SINR <0:
                                print("SINR<0!!")
                                print('Debug error:')
                                print('SINR:',SINR)
                                print('Band:',B)
                                print('TX power dBm:', Power)
                                print('Rx_power_dBm:',Rx_power_dBm)
                                print('Rx_power_w=',Rx_power_W)
                                print('Total interference:',total_ints)
                                print('Distance:',Diss)


                                # print (SINR, B)
                                # input()

                            exec('AP[i].SINR_{}.append(SINR)'.format(B))
                            # AP[i].SINR.append(SINR)
                            exec('STA[AP[i].Group_{}[g][sta]].SINR_UP_{}=SINR'.format(B,B))
                            # STA[AP[i].Group[g][sta]].SINR_UP=SINR
                            # STA[AP[i].ID_Of_STAs[j]].DataRate_DL=STA[AP[i].ID_Of_STAs[j]].NumOfRU*Wsc*np.log2(1+STA[AP[i].ID_Of_STAs[j]].SINR)/1000000
                            exec('STA[AP[i].Group_{}[g][sta]].DataRate_UL_{} = STA[AP[i].Group_{}[g][sta]].NumOfRU_{} * Wsc * Get_pbs_per_Hz(SINR) / 1000 / 1000'.format(B,B,B,B))
                            # STA[AP[i].Group[g][sta]].DataRate_UL = STA[AP[i].Group[g][sta]].NumOfRU * Wsc * Get_pbs_per_Hz(SINR) / 1000 / 1000


    # input()

    for i in range(len(AP)):
        # print('i=',i)
        ttt=[]
        nnn=[]
        for B in Band:
            if B=='24G':
                T_SIFS = 10 * 10 ** (-6)
            if B=='5G1':
                T_SIFS = 16 * 10 ** (-6)
            if B=='5G2':
                T_SIFS = 16 * 10 ** (-6)
            if B=='6G':
                T_SIFS = 16 * 10 ** (-6)
            T_DL = T_DIFS + T_backoff + T_dl + T_SIFS + T_OFDMA
            T_UL = T_DIFS + T_backoff + T_TF + 2 * T_SIFS + T_ul + T_MBA

            exec('Round_time_temp = AP[i].Round_{}*(T_UL+T_DL)'.format(B))
            Round_time=locals()['Round_time_temp']
            # print('Round_time',Round_time)
            # ttt.append(Round_time)
            # input()
            # Round_time = AP[i].Round*(T_UL+T_DL)
            exec('lenNeighbor1=len(AP[i].NeiInt_{})'.format(B))
            lenNeighbor=locals()['lenNeighbor1']
            # nnn.append(lenNeighbor)
            # print('band=',B)
            # print('num of neighbor',lenNeighbor)
            # print('len=',len(AP[i].NeiInt_24G))
            # print('AP neighbor list=', AP[i].NeiInt_24G)
            if lenNeighbor>0:
                for j in range(lenNeighbor):
                    # print('j=',j)
                    # print('AP neighbor list=',AP[i].NeiInt_24G)
                    #
                    # print('round of neighbor=',AP[AP[i].NeiInt_24G[j]].Round_24G*(T_UL+T_DL))
                    # exec('Round_time = Round_time+AP[AP[i].NeiInt_{}[j]].Round_{}*(T_UL+T_DL)'.format(B,B))
                    # print(Round_time)
                    # input()

                    if B=='24G':
                        Round_time = Round_time + AP[AP[i].NeiInt_24G[j]].Round_24G * (T_UL + T_DL)
                    if B=='5G1':
                        Round_time = Round_time + AP[AP[i].NeiInt_5G1[j]].Round_5G1 * (T_UL + T_DL)
                    if B=='5G2':
                        Round_time = Round_time + AP[AP[i].NeiInt_5G2[j]].Round_5G2 * (T_UL + T_DL)
                    if B=='6G':
                        Round_time = Round_time + AP[AP[i].NeiInt_6G[j]].Round_6G * (T_UL + T_DL)
                    # print('non exec')
                    # print(Round_time)
                    # input()



                    # Round_time = Round_time+AP[AP[i].NeiInt[j]].Round*(T_UL+T_DL)
            # if lenNeighbor==0:
            #     exec('Round_time = Round_time+AP[AP[i].NeiInt_{}[j]].Round_{}*(T_UL+T_DL)'.format(B, B))
            exec('AP[i].Total_time_{}=Round_time'.format(B))


    # rou=0
    # Ro=[]
    # NuSTA=[]
    # sta=0
    # for i in range(len(AP)):
    #     Ro.append(AP[i].Round_24G)
    #     NuSTA.append(AP[i].Num_Of_STAs)
    #     rou=rou+AP[i].Round_24G
    #     sta=sta+AP[i].Num_Of_STAs
    # print('numOfAPs:',len(AP))
    # print('numOfrou',rou)
    # print('RoList',Ro)
    # print('NuSTA',NuSTA)
    # print('numOfsta',sta)
    # Plot_fig()
    #
    # input()
    global MaxThroughput
    MaxThroughput = 0
    global MinThroughput
    MinThroughput = inf
    for i in range(len(AP)):
        # print(i)
        for B in Band:

            exec('len_temp10=len(AP[i].ID_Of_STAs_{})'.format(B))
            numStas=locals()['len_temp10']
            for j in range(numStas):
                # count=count+1
                exec('idOfsta=AP[i].ID_Of_STAs_{}[j]'.format(B))
                idd_sta=locals()['idOfsta']

                # T_SIFS = 10 * 10 ** (-6)


                # print(AP[i].Total_time_24G)
                # input()

                exec('debug_t=AP[i].Total_time_{}'.format(B))
                debug_tt=locals()['debug_t']
                if debug_tt==0:
                    print("Total_time = 0!!")
                    print("B=",B)
                    print("AP No.:",i)
                    print('There is a error!')
                    input()


                p=0.99
                NumOfPacket=1000
                exec('Throughput_{}=0'.format(B))
                for pak in range(NumOfPacket):
                    pUL=random.random()
                    pDL=random.random()
                    if pUL<=p and pDL<=p:
                        exec('STA[idd_sta].Throughput_{}=(STA[idd_sta].DataRate_DL_{}*T_dl+STA[idd_sta].DataRate_UL_{}*T_ul)/AP[i].Total_time_{}'.format(B,B,B,B))
                        exec('Throughput_{}=Throughput_{}+STA[idd_sta].Throughput_{}'.format(B,B,B))

                    if pUL>p and pDL>p:
                        exec('STA[idd_sta].Throughput_{}=0'.format(B))
                        exec('Throughput_{}=Throughput_{}+STA[idd_sta].Throughput_{}'.format(B,B,B))

                    if pUL<=p and pDL>p:
                        exec('STA[idd_sta].Throughput_{}=(0*T_dl+STA[idd_sta].DataRate_UL_{}*T_ul)/AP[i].Total_time_{}'.format(B,B,B))
                        exec('Throughput_{}=Throughput_{}+STA[idd_sta].Throughput_{}'.format(B,B,B))

                    if pUL>p and pDL<=p:
                        exec('STA[idd_sta].Throughput_{}=(STA[idd_sta].DataRate_DL_{}*T_dl+0*T_ul)/AP[i].Total_time_{}'.format(B,B,B))
                        exec('Throughput_{}=Throughput_{}+STA[idd_sta].Throughput_{}'.format(B,B,B))

                exec('STA[idd_sta].Throughput_sim_{}=Throughput_{}/NumOfPacket'.format(B, B))
                exec('STA[idd_sta].Throughput_{}=(STA[idd_sta].DataRate_DL_{}*T_dl+STA[idd_sta].DataRate_UL_{}*T_ul)/AP[i].Total_time_{}'.format(B, B, B, B))



    idd_sta=0
    for idd_sta in range(len(STA)):
        STA[idd_sta].Throughput=STA[idd_sta].Throughput_24G+STA[idd_sta].Throughput_5G1+STA[idd_sta].Throughput_5G2+STA[idd_sta].Throughput_6G
        STA[idd_sta].Throughput_sim = STA[idd_sta].Throughput_sim_24G + STA[idd_sta].Throughput_sim_5G1 + STA[idd_sta].Throughput_sim_5G2 + STA[idd_sta].Throughput_sim_6G
        if STA[idd_sta].Throughput==0:
            print("Throughput=0!!")
            print('STA =',idd_sta)
            print('AP =',STA[idd_sta].IDOfAP)
            print('DL SINR =',STA[idd_sta].SINR)
            print('UL SINR =', STA[idd_sta].SINR_UP)
            print('Throughput error!')
            # Plot_fig()
            # input()
        # else:
        #     print(count)
        Throughput_of_STAs.append(STA[idd_sta].Throughput)



        if STA[idd_sta].Throughput>MaxThroughput:
            MaxThroughput=STA[idd_sta].Throughput
            APNo=STA[idd_sta].IDOfAP
            idofsta=idd_sta

            # print('STA[idd_sta].IDOfAP',STA[idd_sta].IDOfAP)
            # input()


            ChannelNo24G = AP[STA[idd_sta].IDOfAP].C_24G
            ChannelNo5G1 = AP[STA[idd_sta].IDOfAP].C_5G1
            ChannelNo5G2 = AP[STA[idd_sta].IDOfAP].C_5G2
            ChannelNo6G = AP[STA[idd_sta].IDOfAP].C_6G
            IDOfstas_24G=AP[STA[idd_sta].IDOfAP].ID_Of_STAs_24G
            IDOfstas_5G1 = AP[STA[idd_sta].IDOfAP].ID_Of_STAs_5G1
            IDOfstas_5G2 = AP[STA[idd_sta].IDOfAP].ID_Of_STAs_5G2
            IDOfstas_6G = AP[STA[idd_sta].IDOfAP].ID_Of_STAs_6G

        if STA[idd_sta].Throughput<MinThroughput:

            MinThroughput=STA[idd_sta].Throughput
            APNo1 = STA[idd_sta].IDOfAP
            idofsta1 = idd_sta
            ChannelNo24G1 = AP[STA[idd_sta].IDOfAP].C_24G
            ChannelNo5G11= AP[STA[idd_sta].IDOfAP].C_5G1
            ChannelNo5G21 = AP[STA[idd_sta].IDOfAP].C_5G2
            ChannelNo6G1 = AP[STA[idd_sta].IDOfAP].C_6G
            IDOfstas_24G1 = AP[STA[idd_sta].IDOfAP].ID_Of_STAs_24G
            IDOfstas_5G11 = AP[STA[idd_sta].IDOfAP].ID_Of_STAs_5G1
            IDOfstas_5G21 = AP[STA[idd_sta].IDOfAP].ID_Of_STAs_5G2
            IDOfstas_6G1 = AP[STA[idd_sta].IDOfAP].ID_Of_STAs_6G

    # print('MaxThroughput',MaxThroughput)
    # print('APNo',APNo)
    # print('ChannelNo24G',ChannelNo24G)
    # print('ChannelNo5G1',ChannelNo5G1)
    # print('ChannelNo5G2',ChannelNo5G2)
    # print('ChannelNo6G',ChannelNo6G)
    # print('IDOfstas_24G',IDOfstas_24G)
    # print('IDOfstas_5G1', IDOfstas_5G1)
    # print('IDOfstas_5G2', IDOfstas_5G2)
    # print('IDOfstas_6G', IDOfstas_6G)
    # print('idofsta',idofsta)
    # print('---------')

    # For savety margin test
    if SafetyMargin==1:
        print('MinThroughput',MinThroughput)
        print('For safety margin test has done!')
        input()

    # print('APNo1',APNo1)
    # print('ChannelNo24G1',ChannelNo24G1)
    # print('ChannelNo5G11',ChannelNo5G11)
    # print('ChannelNo5G21',ChannelNo5G21)
    # print('ChannelNo6G1',ChannelNo6G1)
    # print('IDOfstas_24G1',IDOfstas_24G1)
    # print('IDOfstas_5G11', IDOfstas_5G11)
    # print('IDOfstas_5G21', IDOfstas_5G21)
    # print('IDOfstas_6G1', IDOfstas_6G1)
    # print('idofsta1',idofsta1)

                        #Calculate the throughput of STAs
                        # Ti=(T_DL+T_UL)*len(AP[i].Group)
                        # STA[AP[i].Group[g][sta]].Throughput=(STA[AP[i].Group[g][sta]].DataRate_UL*T_ul+STA[AP[i].Group[g][sta]].DataRate_DL*T_dl)/(Ti*)

    #-------------------------------------------------------
    # atest=0
    # for i in range(len(AP)):
    #     print("2.4GC",AP[i].C_24G)
    #     print("5G1C", AP[i].C_5G1)
    #     print("5G2C", AP[i].C_5G2)
    #     print("6GC", AP[i].C_6G)
    #
    #     print("2.4GRU",AP[i].NumRUs_24G)
    #     print("5G1RU", AP[i].NumRUs_5G1)
    #     print("5G2RU", AP[i].NumRUs_5G2)
    #     print("6GRU", AP[i].NumRUs_6G)
    #
    #     print("2.4Ggroup",len(AP[i].Group_24G))
    #     print("5G1group", len(AP[i].Group_5G1))
    #     print("5G2group", len(AP[i].Group_5G2))
    #     print("6Ggroup", len(AP[i].Group_6G))
    #
    #     print('NumOfstations=',AP[i].Num_Of_STAs)
    #     input()

    return
#Step 7
# RUassignment()
# print('------')
# input()

def Plot_Throughput():
    plt.plot(Throughput_of_STAs, '-bo')
    plt.pause(0.1)

# for i in range(NumOfSTAs):
#     print('interference=',STA[i].Total_Interference)
#     input()
#
# Plot_Throughput()
# input()

# def Get_Throughput(AP,STA):
def ReSet_Initial_StateOf_APs_STAs():
    Reset_all_APsettings()
    Reset_STAsetting()

    # AP = []
    # # print(NumOfAPs)
    # # print('test')
    # # input()
    # for i in range(NumOfAPs):
    #
    #     AP.append(AccessPoint())
    # # print(AP[6].x,AP[6].y)
    # # input()
    # STA = []
    # # for i in range(NumOfSTAs+Num_Of_Static_STAs):????
    # for i in range(NumOfSTAs):
    #     STA.append(Station())


############################### AP placement ##############################
def Reset_all_APsettings(): #Exclude x,y
    for i in range(len(AP)):
        AP[i].ID_Of_STAs = []  # The ID of STAs that associate with this AP

        AP[i].ID_Of_STAs_24G = []
        AP[i].ID_Of_STAs_5G1 = []
        AP[i].ID_Of_STAs_5G2 = []
        AP[i].ID_Of_STAs_6G = []

        AP[i].Num_Of_STAs = 0  # The number of STAs that have been associated with this AP

        AP[i].Dis_Between_AP_and_STAs = []  # The distance between AP and STAs that associated with this AP
        AP[i].Dis_Between_AP_and_STAs_24G = []
        AP[i].Dis_Between_AP_and_STAs_5G1 = []
        AP[i].Dis_Between_AP_and_STAs_5G2 = []
        AP[i].Dis_Between_AP_and_STAs_6G = []

        AP[i].P = 0  # power level
        AP[i].P_24G=27
        AP[i].P_5G1=28
        AP[i].P_5G2=28
        AP[i].P_6G=30
        # AP[i].P_24G = 0
        # AP[i].P_5G1 = 0
        # AP[i].P_5G2 = 0
        # AP[i].P_6G = 0

        AP[i].C = 0  # channel no.
        AP[i].C_24G = 0
        AP[i].C_5G1 = 0
        AP[i].C_5G2 = 0
        AP[i].C_6G = 0

        AP[i].NumRUs_24G=0
        AP[i].NumRUs_5G1=0
        AP[i].NumRUs_5G2=0
        AP[i].NumRUs_6G=0

        AP[i].RUperUser_24G=0
        AP[i].RUperUser_5G1=0
        AP[i].RUperUser_5G2=0
        AP[i].RUperUser_6G=0

        AP[i].CandidateChannelsList_24G = []
        AP[i].CandidateChannelsList_5G1 = []
        AP[i].CandidateChannelsList_5G2 = []
        AP[i].CandidateChannelsList_6G = []

        AP[i].NumOfInterference_24G = 0
        AP[i].NumOfInterference_5G1 = 0
        AP[i].NumOfInterference_5G2 = 0
        AP[i].NumOfInterference_6G = 0

        AP[i].r_24G = 0
        AP[i].r_5G1 = 0
        AP[i].r_5G2 = 0
        AP[i].r_6G = 0

        AP[i].Total_Interference = 0
        AP[i].Total_Interference_24G = 0
        AP[i].Total_Interference_5G1 = 0
        AP[i].Total_Interference_5G2 = 0
        AP[i].Total_Interference_6G = 0

        AP[i].SINR = []
        AP[i].SINR_24G = []
        AP[i].SINR_5G1 = []
        AP[i].SINR_5G2 = []
        AP[i].SINR_6G = []

        AP[i].Rx_power = 0
        AP[i].Rx_power_24G = 0
        AP[i].Rx_power_5G1 = 0
        AP[i].Rx_power_5G2 = 0
        AP[i].Rx_power_6G = 0

        AP[i].Rx_power_dBm = 0
        AP[i].Rx_power_dBm_24G = 0
        AP[i].Rx_power_dBm_5G1 = 0
        AP[i].Rx_power_dBm_5G2 = 0
        AP[i].Rx_power_dBm_6G = 0

        AP[i].Rx_power_W = 0
        AP[i].Rx_power_W_24G = 0
        AP[i].Rx_power_W_5G1 = 0
        AP[i].Rx_power_W_5G2 = 0
        AP[i].Rx_power_W_6G = 0

        AP[i].Group = []  # Maximum transmission rounds
        AP[i].Group_24G = []
        AP[i].Group_5G1 = []
        AP[i].Group_5G2 = []
        AP[i].Group_6G = []

        AP[i].NeighborAPList = []
        AP[i].NeighborAPList_24G = []
        AP[i].NeighborAPList_5G1 = []
        AP[i].NeighborAPList_5G2 = []
        AP[i].NeighborAPList_6G = []

        AP[i].NeiInt_24G = []
        AP[i].NeiInt_5G1 = []
        AP[i].NeiInt_5G2 = []
        AP[i].NeiInt_6G = []

        AP[i].Round = 0
        AP[i].Round_24G = 0
        AP[i].Round_5G1 = 0
        AP[i].Round_5G2 = 0
        AP[i].Round_6G = 0

        AP[i].Total_time = 0
        AP[i].Total_time_24G = 0
        AP[i].Total_time_5G1 = 0
        AP[i].Total_time_5G2 = 0
        AP[i].Total_time_6G = 0

        AP[i].Dis_to_IS = []


def Func_CalculateNumOfAPsAndIDOfAPsCouldBeAssociated(fulltest):
    global CannotBeCovered
    for i in range(len(STA)):
        # print("len(STA)",len(STA))
        # print("i=",i)
        for j in range(len(AP)):
            # print("j=",j)
            # print("len(AP)",len(AP))
            #3D distance
            d_sta_ap=((STA[i].x-AP[j].x)**2+(STA[i].y-AP[j].y)**2+(STA[i].z-AP[j].z)**2)**(0.5)

            # print("d_sta_ap-1:",d_sta_ap)
            #
            #3D distance [in this case, we do not need 3D distance]
            # d_sta_ap=(d_sta_ap**2+H_sta**2)**(0.5)
            # print("d_sta_ap-21:", d_sta_ap)
            # print("H_sta:",H_sta)
            # input()

            # print("d_sta_ap",d_sta_ap)
            # print("D",D)
            # input()

            if d_sta_ap<=D:
                # print(D)
                # input()
                STA[i].NumOfAPs=STA[i].NumOfAPs+1
                STA[i].IDOfAPs.append(j)

                # print("cover")

                STA[i].DistancesFromAPs.append(d_sta_ap)
                # print("fulltest",fulltest)
                # print("STA[i].IDOfAPs",STA[i].IDOfAPs)

                # print("STA[i].DistancesFromAPs:",STA[i].DistancesFromAPs)
                # input()


        if fulltest==1:
            if len(STA[i].IDOfAPs)==0:
                CannotBeCovered=1
                break
            else:
                if len(STA[i].IDOfAPs)<KK+1:
                    CannotBeCovered=1
                    break
                else:
                    CannotBeCovered=0
        else:
            CannotBeCovered=0

        # print("len(STA[i].IDOfAPs)",len(STA[i].IDOfAPs))

    # for i in range(len(STA)):
    #     print("i=",i)
    #     print("STA[i].IDOfAPs",STA[i].IDOfAPs)
    #     print("STA[i].x",STA[i].x)
    #     print("STA[i].x", STA[i].y)
    #
    # for j in range(len(AP)):
    #     print("j=",j)
    #     print("AP[j].x",AP[j].x)
    #     print("AP[j].y", AP[j].y)
    # input()

    # print("CannotBeCovered--inside",CannotBeCovered)
    # print("======")
    # input()
    return CannotBeCovered

def Func_GetThroughputOfEachSTAs():
    Flag=1
    for i in range(len(STA)):
        if STA[i].Throughput<rhoH:
            Flag=0
            break
    # if Flag==1:
    #     Plot_Throughput()

    return Flag



def Func_TestTheSolution():
    global STA
    global AP,CannotBeCovered

    # print("test:")
    # input()

    #If SafetyMargin==1, import apinfo for safety margin test
    if SafetyMargin==1:
        
        ap = np.loadtxt('ap_info.csv')
        AP = []
        for i in range(len(ap)):
            AP.append(AccessPoint())
            AP[i].x = ap[i][0]
            AP[i].y = ap[i][1]


    faInitial_temp=0
    STA_temp=STA

    IsASolution=1
    if len(AP)==0:
        IsASolution=0;

    AP_temp2=AP
    if len(AP)>0:
        Reset_all_APsettings()
        STA=STA_temp
        # Reset_STAsetting()
        #Step 1.
        fulltest=1 #What is the meaning of fulltest=1?
        CannotBeCovered = Func_CalculateNumOfAPsAndIDOfAPsCouldBeAssociated(fulltest)

        # print("CannotBeCovered-outside:",CannotBeCovered)
        # input()

        if CannotBeCovered==0:
            if KK==0:
                AP=AP_temp2
                Reset_all_APsettings()
                STA=STA_temp
                Reset_STAsetting()
                fulltest=0
                CannotBeCovered = Func_CalculateNumOfAPsAndIDOfAPsCouldBeAssociated(fulltest)

                #Step 2 Association
                Association()
                
                
                # for xx in range(len(AP)):
                #     print(AP[xx].Num_Of_STAs)
                #     input()
                
                
                
                Func_PowerAdjustment()
                Func_GetNeighbors_All()
                Func_ChannelAssignment()


                #Show the channels
                # C24G=[]
                # C5G1=[]
                # C5G2=[]
                # C6G=[]
                #
                # for i in range(len(AP)):
                #     C24G.append(AP[i].C_24G)
                #     C5G1.append(AP[i].C_5G1)
                #     C5G2.append(AP[i].C_5G2)
                #     C6G.append(AP[i].C_6G)
                # print(C24G)
                # print(C5G1)
                # print(C5G2)
                # print(C6G)
                # input()


                Func_PowerReAdjustment()
                # print(AP[0].P_24G)
                # print(AP[0].P_5G1)
                # print(AP[0].P_5G2)
                # print(AP[0].P_6G)
                # input()
                # print('call2')
                Func_GetNeiInt()  # Get interference AP list of each AP
                Get_Interference_Of_APs_and_STAs()
                # RUassignment()
                # print('testRU_line_4928')
                # input()
                RUassignment_APCoordination()
                Flag=Func_GetThroughputOfEachSTAs()
                if Flag==0:
                    IsASolution=0
                    AP=AP_temp2
        else:
            IsASolution=0
            AP=AP_temp2
    AP=AP_temp2
    return IsASolution



                #Step 3

def Func_TestTheSolution_modelVerify():
    global STA
    global AP

    #If SafetyMargin==1, import apinfo for safety margin test
    if SafetyMargin==1:
        ap = np.loadtxt('ap_info.csv')
        AP = []
        for i in range(len(ap)):
            AP.append(AccessPoint())
            AP[i].x = ap[i][0]
            AP[i].y = ap[i][1]


    faInitial_temp=0
    STA_temp=STA

    IsASolution=1
    if len(AP)==0:
        IsASolution=0;

    AP_temp2=AP
    if len(AP)>0:
        Reset_all_APsettings()
        STA=STA_temp
        # Reset_STAsetting()
        #Step 1.
        fulltest=1
        CannotBeCovered = Func_CalculateNumOfAPsAndIDOfAPsCouldBeAssociated(fulltest)
        if CannotBeCovered==0:
            if KK==0:
                AP=AP_temp2
                Reset_all_APsettings()
                STA=STA_temp
                Reset_STAsetting()
                fulltest=0
                CannotBeCovered = Func_CalculateNumOfAPsAndIDOfAPsCouldBeAssociated(fulltest)

                #Step 2 Association
                Association()
                Func_PowerAdjustment()
                Func_GetNeighbors_All()
                Func_ChannelAssignment()


                #Show the channels
                # C24G=[]
                # C5G1=[]
                # C5G2=[]
                # C6G=[]
                #
                # for i in range(len(AP)):
                #     C24G.append(AP[i].C_24G)
                #     C5G1.append(AP[i].C_5G1)
                #     C5G2.append(AP[i].C_5G2)
                #     C6G.append(AP[i].C_6G)
                # print(C24G)
                # print(C5G1)
                # print(C5G2)
                # print(C6G)
                # input()


                Func_PowerReAdjustment()
                # print(AP[0].P_24G)
                # print(AP[0].P_5G1)
                # print(AP[0].P_5G2)
                # print(AP[0].P_6G)
                # input()
                # print('call2')
                Func_GetNeiInt()  # Get interference AP list of each AP
                Get_Interference_Of_APs_and_STAs()
                # RUassignment()
                # print('testRU_line_4928')
                # input()
                RUassignment_APCoordination_modelVerify()
                
                
                
                Flag=Func_GetThroughputOfEachSTAs()
                if Flag==0:
                    IsASolution=0
                    AP=AP_temp2
        else:
            IsASolution=0
            AP=AP_temp2
    AP=AP_temp2
    return IsASolution






                #Step 3


def Random():
    #Generate locations with Greedy
    global STA
    STA_temp=STA



    # Place APs to the network with Greedy
    global APNode
    global AP
    AP=APNode
    # print(len(APP))
    # print(len(AP))
    # input()
    count=0
    faInitial=1
    while True:
        STA=STA_temp
        IsASolution = Func_TestTheSolution()
        if IsASolution==0:

            # if index_Greedy==0:
            #     index_Greedy=len(APP)
            AP.append(AccessPoint())
            List_loc=np.arange(0,len(CandidateCites),1)
            index_random=np.random.choice(List_loc,1)
            Index=index_random[0]

            # print(List_loc)
            # print(CandidateCites)
            # print(Index)
            # print(CandidateCites[Index][0])
            # print(CandidateCites[Index][1])
            # input()

            AP[count].x=CandidateCites[Index][0]
            AP[count].y=CandidateCites[Index][1]
            Reset_all_APsettings()
            print('STAGE 1: It has added {} APs to the network.'.format(len(AP)))
            print('MaxThroughput',MaxThroughput)
            print('MinThroughput',MinThroughput)

            if MinThroughput>=8:
                print("################################This is solution for 8Mbps")
            if MinThroughput>=9:
                print("################################This is solution for 9Mbps")
            if MinThroughput >= 9.9:
                print("################################This is solution for 10Mbps")



            if (len(AP))>APsthreshold:
                print(f'The number of APs is more than {APsthreshold}. There is no suitable solution!')
                input()
            count=count+1
            # print(count)
            continue
        else:
            print('The Random has done! The number of APs is: {}'.format(len(AP)))
            print('MaxThroughput',MaxThroughput)
            print('MinThroughput',MinThroughput)
            break



    return AP

####Stage 1: Greedy  2024-12-1更新：每次迭代中，只选能让吞吐率比现有提升的位置以放置一个新AP。
def Func_Greedy():
    #Generate locations with Greedy
    global STA
    STA_temp=STA
    Dis=np.zeros([len(CandidateCites),NumOfSTAs+Num_Of_Static_STAs])

    # print(size(Dis))

    # print(CandidateCites)
    # print(NumOfSTAs+Num_Of_Static_STAs)
    # input()

    for i in range(len(CandidateCites)):
        for j in range(NumOfSTAs+Num_Of_Static_STAs):
            
            #3D distance
            d=((STA[j].x-CandidateCites[i][0])**2+(STA[j].y-CandidateCites[i][1])**2+(STA[j].z-height)**2)**(0.5)
            # print(d)
            # print(D)
            # input()

            #3D distance
            # Here, we do not need 3D distance? right?

            if d<=D:
                Dis[i,j]=1
    APP=[]
    while np.sum(Dis)!=0:
        Bx=np.sum(Dis,axis=1)
        # print(Bx)
        # input()
        NumOfSTAx=max(Bx)
        indx=np.where(Bx==NumOfSTAx)
        Indx=indx[0]
        # print('llllen',len(Indx))
        # input()
        if len(Indx)>1:
            iddxx=np.random.choice(Indx,1)
            id = iddxx[0]
        else:
            # idddd=np.argmax(Bx)
            id=Indx[0]
            # print('indx',idxxx)
            # print('id',idddd)
            # input()



        indicator=id
        APP.append(AccessPoint())
        APindicator=len(APP)-1
        APP[APindicator].x=CandidateCites[indicator][0]
        APP[APindicator].y = CandidateCites[indicator][1]
        APP[APindicator].Num_Of_STAs=NumOfSTAx
        # print('NumX',NumX)
        # input()
        #
        # print(Bx)
        # print(NumOfSTAx)
        # print(id)
        # print(Bx[id])
        # input()
        #Set those stas as 0
        # print(Dis[id])
        # print('total', len(Dis[id]))
        indexx=np.where(Dis[id]==1)
        # print(len(indexx[0]))
        Dis[:,indexx]=0


        # print(Dis)
        # print(Dis.shape)
        # # print('total',len(Dis[id]))
        #
        #
        # print(len(indexx[0]))
        # print(indexx)
    #     input()
    #
    # print(Dis.shape)
    # print(Dis)
    # print(sum(Dis))
    # input()


    # Place APs to the network with Greedy
    global APNode,BestThroughput,CannotBeCovered
    global AP,MinThroughput,MaxThroughput
    AP=APNode
    # print(len(APP))
    # print(len(AP))
    # input()
    count=0
    a_count=0
    faInitial=1
    
    MinThroughput=0
    BestThroughput=0
    temp_minThr=0
    MaxThroughput=0
    while True:
        STA=STA_temp
        
        
        
        
        
        IsASolution = Func_TestTheSolution()
        # print("CannotBeCovered",CannotBeCovered)
        
        # input()
        
        # print("ttttt")
        print('MaxThroughput',MaxThroughput)
        print('MinThroughput', MinThroughput)
        
        if MinThroughput<temp_minThr:     
            
            print("===== smaller then ====")
            
            print("temp_minthr: ",temp_minThr)        
        
            print('MaxThroughput',MaxThroughput)
            print('MinThroughput', MinThroughput)       
            
        
            # print(AP)
            # input()       
            
            print("remove--AP--test")
            _=AP.pop()
            # print(AP)
            # input()
        
        
        # IsASolution = Func_TestTheSolution()
        if IsASolution==0 and CannotBeCovered==1: #
            index_Greedy=np.mod(count,len(APP))
            # if index_Greedy==0:
            #     index_Greedy=len(APP)
            AP.append(AccessPoint())
            
            print("Index_greedy: ",index_Greedy)
            # print(AP)
            # input()            
            # AP[count].x=APP[index_Greedy].x
            # AP[count].y=APP[index_Greedy].y
            AP[len(AP)-1].x=APP[index_Greedy].x
            AP[len(AP)-1].y=APP[index_Greedy].y
            Reset_all_APsettings()
            
            # temp_minThr=BestThroughput            
            # if MinThroughput>BestThroughput:
            #     BestThroughput=MinThroughput
                
            #     a_count=0 #找到一个更好的位置，并放一个AP下去后，重头开始搜索。
            #     print("===larger than====")
            
            #     print('STAGE 1: It has added {} APs to the network.'.format(len(AP)))
            #     print('MaxThroughput',MaxThroughput)
            #     print('MinThroughput', BestThroughput)
            
            # print('STAGE 1: It has added {} APs to the network.'.format(len(AP)))
            # print('MaxThroughput',MaxThroughput)
            # print('MinThroughput', BestThroughput)
            
            # if (len(AP))>APsthreshold:
            #     print(f'The number of APs is more than {APsthreshold}. There is no suitable solution!')
            #     input()
                
                
                
                
            # print('Test range of AP.')
            # print(D)
            # print(AP[count].r_24G)
            # Plot_fig()

            count=count+1
            continue
        
        
        
        # else:
        #     print('The Greedy has done! The number of APs is: {}'.format(len(AP)))
        #     print('MaxThroughput',MaxThroughput)
        #     print('MinThroughput', MinThroughput)
        #     break
        
        if IsASolution==0 and CannotBeCovered==0:
            
            
            AP.append(AccessPoint())
            
            print("Index_a: ",a_count)
            print("total candidate locs:",len(CandidateCites))
            # print(AP)
            # input()            
            # AP[count].x=APP[index_Greedy].x
            # AP[count].y=APP[index_Greedy].y
            AP[len(AP)-1].x=CandidateCites[a_count][0]
            AP[len(AP)-1].y=CandidateCites[a_count][1]
            Reset_all_APsettings()
            
            temp_minThr=BestThroughput
            
            if MinThroughput>BestThroughput:
                BestThroughput=MinThroughput
                a_count=-1 #找到一个更好的位置，并放一个AP下去后，a_count=a_count+1=0重头开始搜索。
                print("===larger than====")
            
                print('STAGE 1: It has added {} APs to the network.'.format(len(AP)))
                print('MaxThroughput',MaxThroughput)
                print('MinThroughput', BestThroughput)
            
            print("===========")
            print('STAGE 1: It has added {} APs to the network.'.format(len(AP)))
            print('MaxThroughput',MaxThroughput)
            print('MinThroughput', BestThroughput)
            
            
            if(a_count==len(CandidateCites)-1):
                print(f'The number of APs is more than {len(CandidateCites)}. There is no suitable solution!')
                input()
            
            if (len(AP))>APsthreshold:
                print(f'The number of APs is more than {APsthreshold}. There is no suitable solution!')
                input()
            # print('Test range of AP.')
            # print(D)
            # print(AP[count].r_24G)
            # Plot_fig()

            a_count=a_count+1
            continue
        
        if IsASolution==1:
            print('The Greedy has done! The number of APs is: {}'.format(len(AP)))
            print('MaxThroughput',MaxThroughput)
            print('MinThroughput', MinThroughput)
            break
            
                
            
            
            
            



    return AP


def LocalOptimizer():
    #Generate locations with Greedy
    global STA
    STA_temp=STA
    Dis=np.zeros([len(CandidateCites),NumOfSTAs+Num_Of_Static_STAs])

    # print(size(Dis))

    # print(CandidateCites)
    # print(NumOfSTAs+Num_Of_Static_STAs)
    # input()

    for i in range(len(CandidateCites)):
        for j in range(NumOfSTAs+Num_Of_Static_STAs):
            d=((STA[j].x-CandidateCites[i][0])**2+(STA[j].y-CandidateCites[i][1])**2+(STA[j].z-height)**2)**(0.5)
            # print(d)
            # print(D)
            # input()

            #3D distance
            # Here, we do not need 3D distance? right?

            if d<=D:
                Dis[i,j]=1
    APP=[]
    while np.sum(Dis)!=0:
        Bx=np.sum(Dis,axis=1)
        # print(Bx)
        # input()
        NumOfSTAx=max(Bx)
        indx=np.where(Bx==NumOfSTAx)
        Indx=indx[0]
        # print('llllen',len(Indx))
        # input()
        if len(Indx)>1:
            iddxx=np.random.choice(Indx,1)
            id = iddxx[0]
        else:
            # idddd=np.argmax(Bx)
            id=Indx[0]
            # print('indx',idxxx)
            # print('id',idddd)
            # input()



        indicator=id
        APP.append(AccessPoint())
        APindicator=len(APP)-1
        APP[APindicator].x=CandidateCites[indicator][0]
        APP[APindicator].y = CandidateCites[indicator][1]
        # APP[APindicator].y = CandidateCites[indicator][1]
        APP[APindicator].Num_Of_STAs=NumOfSTAx
        # print('NumX',NumX)
        # input()
        #
        # print(Bx)
        # print(NumOfSTAx)
        # print(id)
        # print(Bx[id])
        # input()
        #Set those stas as 0
        # print(Dis[id])
        # print('total', len(Dis[id]))
        indexx=np.where(Dis[id]==1)
        # print(len(indexx[0]))
        Dis[:,indexx]=0


        # print(Dis)
        # print(Dis.shape)
        # # print('total',len(Dis[id]))
        #
        #
        # print(len(indexx[0]))
        # print(indexx)
    #     input()
    #
    # print(Dis.shape)
    # print(Dis)
    # print(sum(Dis))
    # input()


    # Place APs to the network with Greedy
    global APNode
    global AP
    AP=APNode
    # print(len(APP))
    # print(len(AP))
    # input()
    count=0
    faInitial=1
    while True:
        STA=STA_temp
        IsASolution = Func_TestTheSolution()
        if IsASolution==0:
            index_Greedy=np.mod(count,len(APP))
            # if index_Greedy==0:
            #     index_Greedy=len(APP)
            AP.append(AccessPoint())
            AP[count].x=APP[index_Greedy].x
            AP[count].y=APP[index_Greedy].y
            Reset_all_APsettings()
            print('STAGE 1: It has added {} APs to the network.'.format(len(AP)))
            print('MaxThroughput',MaxThroughput)
            print('MinThroughput', MinThroughput)
            if (len(AP))>APsthreshold:
                print(f'The number of APs is more than {APsthreshold}. There is no suitable solution!')
                input()
            # print('Test range of AP.')
            # print(D)
            # print(AP[count].r_24G)
            # Plot_fig()

            count=count+1
            continue
        else:
            print('The Greedy has done! The number of APs is: {}'.format(len(AP)))
            print('MaxThroughput',MaxThroughput)
            print('MinThroughput', MinThroughput)
            break



    return AP




def Func_Uniformly():
    #Generate locations with Greedy
    global MaxThroughput
    global MinThroughput
    global STA
    STA_temp=STA
    # Dis=np.zeros([len(CandidateCites),NumOfSTAs+Num_Of_Static_STAs])
    # for i in range(len(CandidateCites)):
    #     for j in range(NumOfSTAs+Num_Of_Static_STAs):
    #         d=((STA[j].x-CandidateCites[i][0])**2+(STA[j].y-CandidateCites[i][1])**2)**(0.5)
    #         if d<=D:
    #             Dis[i,j]=1
    APP=[]
    # while np.sum(Dis)!=0:
    #     Bx=np.sum(Dis,axis=1)
    #     # print(Bx)
    #     # input()
    #     NumOfSTAx=max(Bx)
    #     indx=np.where(Bx==NumOfSTAx)
    #     Indx=indx[0]
    #     # print('llllen',len(Indx))
    #     if len(Indx)>1:
    #         iddxx=np.random.choice(Indx,1)
    #         id = iddxx[0]
    #     else:
    #         # idddd=np.argmax(Bx)
    #         id=Indx[0]
    #         # print('indx',idxxx)
    #         # print('id',idddd)
    #         # input()
    #
    #
    #
    #     indicator=id
    #     APP.append(AccessPoint())
    #     APindicator=len(APP)-1
    #     APP[APindicator].x=CandidateCites[indicator][0]
    #     APP[APindicator].y = CandidateCites[indicator][1]
    #     APP[APindicator].Num_Of_STAs=NumOfSTAx
    #     # print('NumX',NumX)
    #     # input()
    #     #
    #     # print(Bx)
    #     # print(NumOfSTAx)
    #     # print(id)
    #     # print(Bx[id])
    #     # input()
    #     #Set those stas as 0
    #     # print(Dis[id])
    #     # print('total', len(Dis[id]))
    #     indexx=np.where(Dis[id]==1)
    #     # print(len(indexx[0]))
    #     Dis[:,indexx]=0


        # print(Dis)
        # print(Dis.shape)
        # # print('total',len(Dis[id]))
        #
        #
        # print(len(indexx[0]))
        # print(indexx)
    #     input()
    #
    # print(Dis.shape)
    # print(Dis)
    # print(sum(Dis))
    # input()
    # x=[]
    # xx=[]



    if APLoc48==1:
        for index_X in np.arange(5,76,10): # 8*6=48 APs for uniformly distributed
            for index_Y in np.arange(5,56,10):
                APP.append(AccessPoint())
                APindicator=len(APP)-1
                APP[APindicator].x=index_X
                APP[APindicator].y = index_Y
    else:
        # 12 APs for uniformly distributed
        for index_X in np.arange(10, 71, 20):
            for index_Y in np.arange(10, 51, 20):
                APP.append(AccessPoint())
                APindicator = len(APP) - 1
                APP[APindicator].x = index_X
                APP[APindicator].y = index_Y
    # print(len(APP))
    # input()

    # #20 APs for uniformly distributed
    # for index_X in np.arange(13.33,67,13.33):
    #     for index_Y in np.arange(12,49,12):
    #         APP.append(AccessPoint())
    #         APindicator=len(APP)-1
    #         APP[APindicator].x=index_X
    #         APP[APindicator].y = index_Y

    # #16 APs for uniformly distributed
    # for index_X in np.arange(10,71,20):
    #     for index_Y in np.arange(12,49,12):
    #         APP.append(AccessPoint())
    #         APindicator=len(APP)-1
    #         APP[APindicator].x=index_X
    #         APP[APindicator].y = index_Y
    #35 APs for uniformly distributed

    # for index_X in np.arange(5,76,10):
    #     for index_Y in np.arange(5,56,10):
    #         APP.append(AccessPoint())
    #         APindicator=len(APP)-1
    #         APP[APindicator].x=index_X
    #         APP[APindicator].y = index_Y


    # x=np.arange(10,71,10)
    # y=np.arange(10,51,10)
    # print(x)
    # print(y)
    # input()
    # print(len(APP))
    # for i in range(len(APP)):
    #     print(APP[i].x,APP[i].y)
    # input()




    # Place APs to the network with Greedy
    global APNode
    global AP
    AP=APNode
    # print(len(APP))
    # print(len(AP))
    # input()
    count=0
    faInitial=1
    while True:
        STA=STA_temp
        IsASolution = Func_TestTheSolution()
        if IsASolution==0:
            index_Greedy=np.mod(count,len(APP))
            # print("index_Greedy",index_Greedy)
            # print("len(AP)",len(AP))
            # input()
            # if index_Greedy==0:
            #     index_Greedy=len(APP)
            AP.append(AccessPoint())
            AP[count].x=APP[index_Greedy].x
            AP[count].y=APP[index_Greedy].y
            Reset_all_APsettings()
            print('STAGE 1: It has added {} APs to the network.'.format(len(AP)))
            print('MaxThroughput', MaxThroughput)
            print('MinThroughput', MinThroughput)
            if (len(AP))>APsthreshold:
                print(f'The number of APs is more than {APsthreshold}. There is no suitable solution!')
                input()
            # print('Test range of AP.')
            # print(D)
            # print(AP[count].r_24G)
            # Plot_fig()
            # input()

            count=count+1
            continue
        else:
            print('The Uniformly has done! The number of APs is: {}'.format(len(AP)))
            print('MaxThroughput', MaxThroughput)
            print('MinThroughput', MinThroughput)
            break



    return AP
# AP=Func_Greedy()
# NumOfAPs_1=len(AP)
# Plot_fig()
# input()
# Func_Uniformly()
#### Stage 2: Remove APs one by one
def Func_RemoveAPs():
    global STA
    global AP
    n = 0
    STA_temp=STA
    while True:
        Reset_all_APsettings()
        Reset_STAsetting()
        fulltest=0 #why fulltest equals to 0?
        CannotBeCovered = Func_CalculateNumOfAPsAndIDOfAPsCouldBeAssociated(fulltest)

        Association()

        #Sort AP in ascending order according to the number of STAs
        AP = sorted(AP, key=lambda x: x.Num_Of_STAs)
        removetimes=len(AP)

        for i in range(removetimes):
            print('STAGE 2: The total attempts is {}. The progress is {}.'.format(removetimes,i+1))
            buffer=AP[0]
            del AP[0]
            Reset_all_APsettings()
            Reset_STAsetting()
            IsASolution = Func_TestTheSolution()
            if IsASolution==0:
                AP.append(buffer)
            else:
                n=n+1
                print('It has removed {} APs. The number of APs is {}.'.format(n,len(AP)))
                print('MaxThroughput', MaxThroughput)
                print('MinThroughput', MinThroughput)
                break
        if i == (removetimes-1):
            break
    return AP

# AP=Func_RemoveAPs()
# NumOfAPs_2=len(AP)
# Plot_fig()
# input()

import math
from copy import deepcopy
from itertools import combinations
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import time

from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import traceback

# ---------------------------
# 辅助的子进程工作函数（必须在模块顶层以便被 multiprocessing pickler 导入）
# ---------------------------
def _worker_try_edge(args):
    (edge_pair, AP_temp, CandidateCites) = args
    try:
        # 简单打印，帮助观察子进程是否启动（可根据需要注释）
        # print(f"[Worker] start edge={edge_pair}, AP_count={len(AP_temp)}, candidates={len(CandidateCites)}")

        AP_local = deepcopy(AP_temp)

        i0, i1 = edge_pair
        if i0 > i1:
            i0, i1 = i1, i0

        if i0 < 0 or i1 >= len(AP_local):
            return (False, None, {"edge": edge_pair, "tried": 0})

        # 删除较大索引先
        del AP_local[i1]
        del AP_local[i0]

        tried = 0
        for newlocation in range(len(CandidateCites)):
            tried += 1

            # append new AP placeholder / copy
            AP_local.append(deepcopy(AP_local[0]) if len(AP_local) > 0 else None)
            try:
                # 尝试从模块顶层构造 AccessPoint
                from __main__ import AccessPoint
                if AP_local[-1] is None:
                    AP_local[-1] = AccessPoint()
            except Exception:
                try:
                    AccessPoint = globals().get('AccessPoint', None)
                    if AP_local[-1] is None and AccessPoint is not None:
                        AP_local[-1] = AccessPoint()
                except Exception:
                    return (False, None, {"edge": edge_pair, "tried": tried})

            AP_local[-1].x = CandidateCites[newlocation][0]
            AP_local[-1].y = CandidateCites[newlocation][1]

            try:
                # 在子进程内设置全局 AP（子进程独立内存）
                globals()['AP'] = AP_local
                # 如果需要，也可设置 STA（此处保持原样）
                if 'Reset_all_APsettings' in globals():
                    globals()['Reset_all_APsettings']()
                if 'Reset_STAsetting' in globals():
                    globals()['Reset_STAsetting']()

                IsASolution = globals()['Func_TestTheSolution']()
            except Exception as e:
                # 打印 traceback 帮助排查子进程错误（不会抛到主进程）
                # traceback.print_exc()
                IsASolution = 0

            if IsASolution:
                return (True, deepcopy(AP_local), {"edge": edge_pair, "tried": tried})
            else:
                AP_local.pop()

        return (False, None, {"edge": edge_pair, "tried": tried})
    except Exception as e:
        # 捕获任意子进程异常并返回错误信息，主进程可打印
        tb = traceback.format_exc()
        return (False, None, {"edge": edge_pair, "error": str(e), "traceback": tb})


# ---------------------------
# 主函数并行化改写（替换版）
# ---------------------------
def Func_MergeTwoByOne_parallel(parallel_workers=None, chunk_size=16):
    global STA
    global AP

    if parallel_workers is None:
        parallel_workers = max(1, multiprocessing.cpu_count() - 1)
        parallel_workers=24

    STA_temp = deepcopy(STA)
    AP_temp = deepcopy(AP)

    count = 0
    flag_Tri = 1
    Flag_NchoosK = 0

    while True:
        if flag_Tri == 1:
            AP = deepcopy(AP_temp)
            pointNumber = len(AP)

            points = np.zeros((pointNumber, 2))
            X_temp = []
            Y_temp = []

            for i in range(pointNumber):
                X_temp.append(AP[i].x)
                Y_temp.append(AP[i].y)

            if pointNumber > 0:
                points[:, 0] = X_temp
                points[:, 1] = Y_temp

            Points_temp = deepcopy(points)
            PP = Points_temp.tolist()
            AP_INF = deepcopy(PP)

            for Lx in PP[:]:
                if PP.count(Lx) > 1:
                    PP.remove(Lx)

            if len(PP) < 3:
                Flag_NchoosK = 1
            else:
                from scipy.spatial import Delaunay
                tri = Delaunay(points)

                edges = set()
                for simplex in tri.simplices:
                    simplex_edges = [
                        tuple(sorted((simplex[0], simplex[1]))),
                        tuple(sorted((simplex[1], simplex[2]))),
                        tuple(sorted((simplex[2], simplex[0]))),
                    ]
                    edges.update(simplex_edges)

                edges = list(edges)
                print(f"\n边总数: {len(edges)}")

                XYIndex_to_list = []
                for e in edges:
                    i0, i1 = e
                    # dis_two_aps = math.hypot(PP[i0][0] - PP[i1][0], PP[i0][1] - PP[i1][1])
                    if i0 >= len(PP) or i1 >= len(PP):
                        continue  # 跳过越界索引
                    dis_two_aps = math.hypot(PP[i0][0] - PP[i1][0], PP[i0][1] - PP[i1][1])

                    XYIndex_to_list.append([i0, i1, dis_two_aps])

                XYIndex_to_list.sort(key=lambda x: x[2])

                APIndex_to_list = []
                for item in XYIndex_to_list:
                    a = AP_INF.index(PP[item[0]])
                    b = AP_INF.index(PP[item[1]])
                    APIndex_to_list.append([a, b, item[2]])

                total_attempts = len(APIndex_to_list)
                print(f"STAGE 3: 将并行尝试 {total_attempts} 条边（按距离从小到大）")

                task_args = []
                for pair in APIndex_to_list:
                    task_args.append(((pair[0], pair[1]), deepcopy(AP_temp), deepcopy(CandidateCites)))

                found_solution = False
                found_result_AP = None
                found_stats = None

                # 滚动提交 + 等待第一个完成（避免 as_completed 固定初始集合的问题）
                with ProcessPoolExecutor(max_workers=parallel_workers) as exe:
                    total_tasks = len(task_args)
                    submitted = 0
                    running = set()

                    # 提交初始 chunk
                    while submitted < total_tasks and len(running) < chunk_size:
                        args = task_args[submitted]
                        running.add(exe.submit(_worker_try_edge, args))
                        submitted += 1

                    processed = 0
                    last_print_time = time.time()

                    # 循环直到 running 空（没有任务在运行）或找到解
                    while running:
                        done, not_done = wait(running, return_when=FIRST_COMPLETED, timeout=300)

                        if not done:
                            # 超时诊断输出，继续等待
                            print("STAGE 3: wait timeout (no worker finished in 300s). continuing...")
                            continue

                        for fut in done:
                            processed += 1
                            try:
                                ok, newAP, stats = fut.result()
                            except Exception as e:
                                ok = False
                                newAP = None
                                stats = {"error": repr(e)}
                                print(f"[Main] exception getting result: {e}")
                                traceback.print_exc()

                            now = time.time()
                            if now - last_print_time > 0.5 or processed % 10 == 0:
                                print(f"STAGE 3: progress {processed}/{total_tasks} attempts processed.")
                                last_print_time = now

                            if ok:
                                found_solution = True
                                found_result_AP = newAP
                                found_stats = stats
                                print(f"找到可行合并（edge={stats.get('edge')}），尝试次数：{stats.get('tried')}")
                                break

                        if found_solution:
                            # 尝试取消未开始的任务（不能强制停止已运行的进程）
                            for fut in not_done:
                                try:
                                    fut.cancel()
                                except Exception:
                                    pass
                            running.clear()
                            break

                        # 从 running 中移除已完成的 futures
                        running.difference_update(done)

                        # 补充新的任务以保持 chunk_size
                        while submitted < total_tasks and len(running) < chunk_size:
                            args = task_args[submitted]
                            running.add(exe.submit(_worker_try_edge, args))
                            submitted += 1

                    # with 结束，executor 会等待所有已提交任务收尾或取消

                if found_solution:
                    AP_temp = deepcopy(found_result_AP)
                    count += 1
                    AP = deepcopy(AP_temp)
                    print('It has merged {} edges. The number of APs is {}.'.format(count, len(AP)))
                    try:
                        Reset_all_APsettings()
                        Reset_STAsetting()
                        _ = Func_TestTheSolution()
                    except Exception:
                        pass
                    continue
                else:
                    AP = deepcopy(AP_temp)
                    break  # 没找到解，退出 while True

        if Flag_NchoosK == 1:
            AP = deepcopy(AP_temp)
            APIndex = np.arange(0, len(AP), 1)
            APIndex_com = combinations(APIndex, 2)
            APIndex_to_list = []
            for i in APIndex_com:
                APIndex_to_list.append(list(i))

            for i in range(len(APIndex_to_list)):
                dis_two_aps = math.hypot(AP[APIndex_to_list[i][0]].x - AP[APIndex_to_list[i][1]].x,
                                        AP[APIndex_to_list[i][0]].y - AP[APIndex_to_list[i][1]].y)
                APIndex_to_list[i].append(dis_two_aps)
            APIndex_to_list.sort(key=lambda x: x[2])

            task_args = []
            for pair in APIndex_to_list:
                task_args.append(((pair[0], pair[1]), deepcopy(AP_temp), deepcopy(CandidateCites)))

            found_solution = False
            found_result_AP = None

            total_tasks = len(task_args)
            print(f"STAGE 3 (NchoosK): total attempts {total_tasks}")

            with ProcessPoolExecutor(max_workers=parallel_workers) as exe:
                submitted = 0
                running = set()

                while submitted < total_tasks and len(running) < chunk_size:
                    args = task_args[submitted]
                    running.add(exe.submit(_worker_try_edge, args))
                    submitted += 1

                processed = 0
                last_print_time = time.time()

                while running:
                    done, not_done = wait(running, return_when=FIRST_COMPLETED, timeout=300)

                    if not done:
                        print("STAGE 3 (NchoosK): wait timeout (no worker finished in 300s). continuing...")
                        continue

                    for fut in done:
                        processed += 1
                        try:
                            ok, newAP, stats = fut.result()
                        except Exception as e:
                            ok = False
                            newAP = None
                            stats = {"error": repr(e)}
                            print(f"[Main NchoosK] exception getting result: {e}")
                            traceback.print_exc()

                        now = time.time()
                        if now - last_print_time > 0.5 or processed % 10 == 0:
                            print(f"STAGE 3 (NchoosK): progress {processed}/{total_tasks} attempts processed.")
                            last_print_time = now

                        if ok:
                            found_solution = True
                            found_result_AP = newAP
                            print(f"NchoosK found solution edge={stats.get('edge')}, tried {stats.get('tried')}")
                            break

                    if found_solution:
                        for fut in not_done:
                            try:
                                fut.cancel()
                            except Exception:
                                pass
                        running.clear()
                        break

                    running.difference_update(done)
                    while submitted < total_tasks and len(running) < chunk_size:
                        args = task_args[submitted]
                        running.add(exe.submit(_worker_try_edge, args))
                        submitted += 1

            if found_solution:
                AP_temp = deepcopy(found_result_AP)
                count += 1
                AP = deepcopy(AP_temp)
                try:
                    Reset_all_APsettings()
                    Reset_STAsetting()
                    _ = Func_TestTheSolution()
                except Exception:
                    pass
                continue
            else:
                AP = deepcopy(AP_temp)
                break

        break

    return AP










#### Stage 3: Merge two APs by one
def Func_MergeTwoByOne():
    global STA
    global AP

    STA_temp=deepcopy(STA)
    AP_temp=deepcopy(AP)
    # print(id(AP_tem))
    # print(id(AP))
    # print(len(AP_temp))
    #
    # del AP[0]
    # print(len(AP_temp))
    #
    # input()

    count=0
    flag_Tri=1
    Flag_NchoosK=0
    while True:
        # AP=AP_temp
        # x=[]
        # y=[]
        # for i in range(len(AP)):
        #     x.append(AP[i].x)
        #     y.append(AP[i].y)
        if flag_Tri==1:
            AP = deepcopy(AP_temp)
            pointNumber=len(AP)

            points=np.zeros((pointNumber,2))
            X_temp=[]
            Y_temp=[]


            for i in range(pointNumber):
                X_temp.append(AP[i].x)
                Y_temp.append(AP[i].y)

            points[:,0]=X_temp
            points[:,1]=Y_temp

            

            Points_temp=deepcopy(points) #points is a array
            PP=Points_temp.tolist()  #PP is a list
            AP_INF=deepcopy(PP)

            # print(PP)
            # print(AP_INF)
            # input()

            for Lx in PP[:]:
                if PP.count(Lx)>1:
                    PP.remove(Lx)

            if len(PP)<3:
                Flag_NchoosK=1
            else:
                tri=Delaunay(points)
                # print(tri.simplices) #三角形编号
                # # [[ 3 16 19]...
                # #  [11  5  2]
                # #  [ 3 14  2]]
                # # print(tri.simplices.shape())
                # # print(len(tri.simplices))
                
                # print(points[tri.simplices]) #具体坐标
                # # [[[ 75. 115.]
                # #   [ 45. 105.]
                # #   [ 75.  65.]]...

                # #  [[135.  75.]
                # #   [155.  65.]
                # #   [155. 105.]]]
                # input()
                
                # print('test')
                
                # 生成所有三角形的边（无重复）
                edges = set()
                for simplex in tri.simplices:
                    simplex_edges = [
                        tuple(sorted((simplex[0], simplex[1]))),
                        tuple(sorted((simplex[1], simplex[2]))),
                        tuple(sorted((simplex[2], simplex[0]))),
                    ]
                    edges.update(simplex_edges)
                
                # 转为列表并输出
                edges = list(edges)
                # print("\n所有三角形的边（点编号对）:")
                # for e in edges:
                #     print(e)
                
                print(f"\n边总数: {len(edges)}")
                
                # input()



                # print(points)
                # print(points.shape)
                
                
                # print(PP)
                # plt.figure(figsize=(6,6))
                # plt.triplot(points[:,0], points[:,1], tri.simplices, color='blue', linewidth=1)
                # plt.plot(points[:,0], points[:,1], 'ro')  # 绘制点
            
                # # 标注点编号
                # for i, p in enumerate(points):
                #     plt.text(p[0]+0.02, p[1]+0.02, str(i), color='red')
            
                # plt.title("Delaunay 三角剖分示意图", fontsize=14)
                # plt.xlabel("X 坐标")
                # plt.ylabel("Y 坐标")
                # plt.grid(True)
                # plt.axis('equal')
                # plt.show()
                # input()

                XYIndex=np.arange(0,len(PP),1)
                XYIndex_com=combinations(XYIndex,2)
                
                # print("XYIndex",XYIndex)
                # print("XYIndex_com",XYIndex_com)
                
                # input()
                
                XYIndex_com=edges
                # print("XYIndex_com",XYIndex_com)
                # input()
                
                XYIndex_to_list = []
                for i in XYIndex_com:
                    XYIndex_to_list.append(list(i))
                # print('ComLocsList:', XYIndex_to_list)
                
                # input()

                for i in range(len(XYIndex_to_list)):
                    dis_two_aps=((PP[XYIndex_to_list[i][0]][0]-PP[XYIndex_to_list[i][1]][0])**2+(PP[XYIndex_to_list[i][0]][1]-PP[XYIndex_to_list[i][1]][1])**2)**(0.5)
                    XYIndex_to_list[i].append(dis_two_aps)
                # print("XYIndex_to_list",XYIndex_to_list)
                # input()

                XYIndex_to_list.sort(key=lambda XYIndex_to_list:XYIndex_to_list[2])

                APIndex_to_list=[]
                for i in range(len(XYIndex_to_list)):
                    IInndex=[]

                    IInndex.append(AP_INF.index(PP[XYIndex_to_list[i][0]]))
                    IInndex.append(AP_INF.index(PP[XYIndex_to_list[i][1]]))
                    APIndex_to_list.append(IInndex)

                # print(AP_INF)
                # print(PP)
                # print(XYIndex_to_list)
                # print(APIndex_to_list)
                # input()







                for i in range(len( APIndex_to_list)): # test all edges to see if they can be merged
                    print('STAGE 3: The total attempts is {}. The progress is {}. The number of APs: {}'.format(len( APIndex_to_list),i+1,len(AP)))
                    # print('STAGE 3:')
                    flag=0
                    AP=deepcopy(AP_temp)
                    # print('len_AP', len(AP))
                    # input()

                    # originalAP1_x=AP[APIndex_to_list[i][0]].x
                    # originalAP1_y = AP[APIndex_to_list[i][0]].y
                    # originalAP2_x=AP[APIndex_to_list[i][1]].x
                    # originalAP2_y = AP[APIndex_to_list[i][1]].y
                    # print(APIndex_to_list)
                    # print(APIndex_to_list[i][0])
                    # print(APIndex_to_list[i][1]-1)
                    # print(len(AP))
                    # print('progree',i)
                    # print('testapp')
                    # input()
                    # print('glo?',len(AP_temp))

                    del AP[APIndex_to_list[i][0]]
                    del AP[APIndex_to_list[i][1]-1]

                    # print('loc?',len(AP_temp))

                    for newlocation in range(len(CandidateCites)):
                        # print('STAGE 3: The total attempts is {}. The progress is {}.Sub prosess:{}'.format(len(APIndex_to_list),
                        #                                                                       i + 1,newlocation))
                        AP.append(AccessPoint())
                        AP[-1].x=CandidateCites[newlocation][0]
                        AP[-1].y = CandidateCites[newlocation][1]

                        # print('tx',AP[0].r_24G)
                        # input()


                        Reset_all_APsettings()
                        Reset_STAsetting()

                        # print('t1',AP[0].r_24G)
                        # input()

                        IsASolution = Func_TestTheSolution()

                        # print('t2',AP[0].r_24G)
                        # input()

                        if IsASolution==0:
                            del AP[-1]
                        else:
                            count=count+1
                            print('It has merged {} edges.The number of APs is {}.'.format(count,len(AP)))
                            print('MaxThroughput', MaxThroughput)
                            print('MinThroughput', MinThroughput)
                            AP_temp=deepcopy(AP)
                            flag=1
                            break

                    # print('tk',AP[0].r_24G)
                    # input()

                    if flag==1:

                        break
                if flag==0:
                    AP=deepcopy(AP_temp)
                    break


                # print('Original:',points)
                # print('Reduce:',PP)
                # print('CombLocs:',XYIndex_com)
                #
                # print('XYIndex_to_list.sort',XYIndex_to_list)
                #
                #
                # input()


        if Flag_NchoosK==1:
            AP = deepcopy(AP_temp)
            APIndex=np.arange(0,len(AP),1)
            APIndex_com=combinations(APIndex,2)
            APIndex_to_list=[]
            for i in APIndex_com:
                APIndex_to_list.append(list(i))

            for i in range(len(APIndex_to_list)):
                dis_two_aps=((AP[APIndex_to_list[i][0]].x-AP[APIndex_to_list[i][1]].x)**2+(AP[APIndex_to_list[i][0]].y-AP[APIndex_to_list[i][1]].y)**2)**(0.5)
                APIndex_to_list[i].append(dis_two_aps)
            # print(APIndex_to_list)
            APIndex_to_list.sort(key=lambda APIndex_to_list:APIndex_to_list[2])

            for i in range(len( APIndex_to_list)): # test all edges to see if they can be merged
                print('STAGE 3: The total attempts is {}. The progress is {}.'.format(len( APIndex_to_list),i+1))
                flag=0
                AP=deepcopy(AP_temp)
                # print('len_AP', len(AP))
                # input()

                # originalAP1_x=AP[APIndex_to_list[i][0]].x
                # originalAP1_y = AP[APIndex_to_list[i][0]].y
                # originalAP2_x=AP[APIndex_to_list[i][1]].x
                # originalAP2_y = AP[APIndex_to_list[i][1]].y
                # print(APIndex_to_list)
                # print(APIndex_to_list[i][0])
                # print(APIndex_to_list[i][1]-1)
                # print(len(AP))
                # print('progree',i)
                # print('testapp')
                # input()
                # print('glo?',len(AP_temp))

                del AP[APIndex_to_list[i][0]]
                del AP[APIndex_to_list[i][1]-1]

                # print('loc?',len(AP_temp))

                for newlocation in range(len(CandidateCites)):
                    AP.append(AccessPoint())
                    AP[-1].x=CandidateCites[newlocation][0]
                    AP[-1].y = CandidateCites[newlocation][1]
                    Reset_all_APsettings()
                    Reset_STAsetting()

                    # print('1',AP[0].r_24G)
                    # input()

                    IsASolution = Func_TestTheSolution()

                    # print('2',AP[0].r_24G)
                    # input()

                    if IsASolution==0:
                        del AP[-1]
                    else:
                        count=count+1
                        print('It has merged {} edges.The number of APs is {}.'.format(count,len(AP)))
                        print('MaxThroughput', MaxThroughput)
                        print('MinThroughput', MinThroughput)
                        AP_temp=deepcopy(AP)
                        flag=1
                        break
                if flag==1:

                    break
            if flag==0:
                AP=deepcopy(AP_temp)
                break

    return AP




        # # print(APIndex_to_list.shape())
        # print(APIndex_to_list)
        # print(len(APIndex_to_list))
        #
        # input()

# AP=Func_MergeTwoByOne()
# NumOfAPs_3=len(AP)
# # print(NumOfAPs)
# Plot_fig()
# input()
# Solutions=[str(NumOfAPs_1),str(NumOfAPs_2),str(NumOfAPs_3)]
# filename='Sols_{}Mbps_{}dBm_{}P_{}NumI_{}.csv'.format(rhoH,P_InterferenceTop_dBm,P_interference,Num_of_interferences,3)
# with open(filename,'a+') as fp:
#     fp.write(",".join(Solutions)+"\n")
def evaluate_two_locations(idx_pair, CandidateCites, AP_template):
    """
    评估在 CandidateCites 中添加一对新AP位置的效果
    """
    from copy import deepcopy
    AP_local = deepcopy(AP_template)

    # 添加两处候选位置
    AP_local.append(AccessPoint())
    AP_local.append(AccessPoint())
    AP_local[-1].x = CandidateCites[idx_pair[0]][0]
    AP_local[-1].y = CandidateCites[idx_pair[0]][1]
    AP_local[-2].x = CandidateCites[idx_pair[1]][0]
    AP_local[-2].y = CandidateCites[idx_pair[1]][1]

    # 重置网络状态并测试方案
    Reset_all_APsettings()
    Reset_STAsetting()
    IsASolution = Func_TestTheSolution()

    if IsASolution == 0:
        return None  # 不可行方案
    else:
        # 返回结果指标
        return {
            "AP": deepcopy(AP_local),
            "MaxThroughput": MaxThroughput,
            "MinThroughput": MinThroughput
        }




# from joblib import Parallel, delayed, cpu_count
# from tqdm import tqdm
# import numpy as np
# from copy import deepcopy
# from itertools import combinations

def Func_ReplaceThreeByTwo_parallel():#parallel version2
    global STA
    global AP

    STA_temp = deepcopy(STA)
    AP_temp = deepcopy(AP)
    count = 0
    flag_tri = 1
    Flag_NchoosK = 0

    while True:
        if flag_tri == 1:
            AP = deepcopy(AP_temp)
            pointNumber = len(AP)
        
            # 构建坐标数组
            points = np.zeros((pointNumber, 2))
            for i in range(pointNumber):
                points[i, 0] = AP[i].x
                points[i, 1] = AP[i].y
        
            Points_temp = deepcopy(points)  # 原始坐标，用于索引映射
        
            # 原始点列表
            PP = Points_temp.tolist()
            AP_INF = deepcopy(PP)  # 保留原始索引来源
        
            # 去重并构造映射表
            unique_PP = []
            orig_to_unique = {}  # key: tuple(x,y), value: 去重后索引
            for i, p in enumerate(PP):
                pt_tuple = (p[0], p[1])
                if pt_tuple not in orig_to_unique:
                    orig_to_unique[pt_tuple] = len(unique_PP)
                    unique_PP.append(p)
        
            PP = unique_PP  # 去重后的点列表
        
            if len(PP) < 3:
                Flag_NchoosK = 1
            else:
                # Delaunay 三角剖分
                from scipy.spatial import Delaunay
                tri = Delaunay(points)
        
                # 计算每个三角形及边长总和
                XYIndex_to_list = []
                for simplex in tri.simplices:
                    d1 = np.linalg.norm(points[simplex[0]] - points[simplex[1]])
                    d2 = np.linalg.norm(points[simplex[1]] - points[simplex[2]])
                    d3 = np.linalg.norm(points[simplex[2]] - points[simplex[0]])
                    XYIndex_to_list.append([*simplex, d1 + d2 + d3])
                XYIndex_to_list.sort(key=lambda v: v[3])
        
                # 构建 AP 原始索引列表（使用 AP_INF 保证不越界）
                APIndex_to_list = []
                for tri_pts in XYIndex_to_list:
                    idx = [
                        AP_INF.index(Points_temp[tri_pts[0]].tolist()),
                        AP_INF.index(Points_temp[tri_pts[1]].tolist()),
                        AP_INF.index(Points_temp[tri_pts[2]].tolist())
                    ]
                    APIndex_to_list.append(idx)

        # if flag_tri == 1:
        #     AP = deepcopy(AP_temp)
        #     pointNumber = len(AP)

        #     points = np.zeros((pointNumber, 2))
        #     for i in range(pointNumber):
        #         points[i, 0] = AP[i].x
        #         points[i, 1] = AP[i].y

        #     Points_temp = deepcopy(points)
            
        #     #original
        #     PP = Points_temp.tolist()
        #     AP_INF = deepcopy(PP)

        #     for Lx in PP[:]:
        #         if PP.count(Lx) > 1:
        #             PP.remove(Lx)
                    
                    
                    

        #     if len(PP) < 3:
        #         Flag_NchoosK = 1
        #     else:
        #         tri = Delaunay(points)
        #         XYIndex_to_list = []
        #         for simplex in tri.simplices:
        #             d1 = np.linalg.norm(points[simplex[0]] - points[simplex[1]])
        #             d2 = np.linalg.norm(points[simplex[1]] - points[simplex[2]])
        #             d3 = np.linalg.norm(points[simplex[2]] - points[simplex[0]])
        #             XYIndex_to_list.append([*simplex, d1 + d2 + d3])
        #         XYIndex_to_list.sort(key=lambda v: v[3])

        #         APIndex_to_list = []
        #         for tri_pts in XYIndex_to_list:
        #             idx = [
        #                 AP_INF.index(PP[tri_pts[0]]),
        #                 AP_INF.index(PP[tri_pts[1]]),
        #                 AP_INF.index(PP[tri_pts[2]])
        #             ]
        #             APIndex_to_list.append(idx)
        else:
            test = 0

        if Flag_NchoosK == 1:
            AP = deepcopy(AP_temp)
            APIndex = np.arange(0, len(AP), 1)
            APIndex_com = combinations(APIndex, 3)
            APIndex_to_list = []
            for i in APIndex_com:
                APIndex_to_list.append(list(i))

            for i in range(len(APIndex_to_list)):
                dis_two_aps1 = np.linalg.norm(np.array([AP[APIndex_to_list[i][0]].x, AP[APIndex_to_list[i][0]].y]) -
                                              np.array([AP[APIndex_to_list[i][1]].x, AP[APIndex_to_list[i][1]].y]))
                dis_two_aps2 = np.linalg.norm(np.array([AP[APIndex_to_list[i][2]].x, AP[APIndex_to_list[i][2]].y]) -
                                              np.array([AP[APIndex_to_list[i][1]].x, AP[APIndex_to_list[i][1]].y]))
                dis_two_aps3 = np.linalg.norm(np.array([AP[APIndex_to_list[i][0]].x, AP[APIndex_to_list[i][0]].y]) -
                                              np.array([AP[APIndex_to_list[i][2]].x, AP[APIndex_to_list[i][2]].y]))
                dis_three_aps = dis_two_aps1 + dis_two_aps2 + dis_two_aps3
                APIndex_to_list[i].append(dis_three_aps)
            APIndex_to_list.sort(key=lambda APIndex_to_list: APIndex_to_list[3])

        # ==============================
        # 主循环：测试三角形组合
        # ==============================
        for i in range(len(APIndex_to_list)):
            print(f'STAGE 4: 总尝试 {len(APIndex_to_list)}，进度 {i + 1}/{len(APIndex_to_list)}，当前AP数 {len(AP)}')
            flag = 0
            AP = deepcopy(AP_temp)

            # 删除原三点
            del AP[APIndex_to_list[i][0]]
            del AP[APIndex_to_list[i][1] - 1]
            del AP[APIndex_to_list[i][2] - 2]

            APIndex_loc = np.arange(0, len(CandidateCites), 1)
            APIndex_com_loc = combinations(APIndex_loc, 2)
            APIndex_loc_to_list = [list(i) for i in APIndex_com_loc]

            # ===============================================================
            # ✅ 并行处理两点组合——块并行优化
            n_jobs = min(cpu_count(), 48)
            n_jobs=24
            print(f"并行评估 {len(APIndex_loc_to_list)} 个两点组合，CPU核数：{n_jobs}")

            # 分块处理，减少多进程开销 chunk_size=50, 100, 150, 哪个更快？
            chunk_size = 100
            blocks = [APIndex_loc_to_list[j:j+chunk_size] for j in range(0, len(APIndex_loc_to_list), chunk_size)]
            # n_jobs = min(cpu_count(), 48)
            # n_jobs=24

            def evaluate_block(block, CandidateCites, AP_template):
                best_local = None
                for idx_pair in block:
                    AP_local = deepcopy(AP_template)
                    AP_local.append(AccessPoint())
                    AP_local.append(AccessPoint())
                    AP_local[-1].x = CandidateCites[idx_pair[0]][0]
                    AP_local[-1].y = CandidateCites[idx_pair[0]][1]
                    AP_local[-2].x = CandidateCites[idx_pair[1]][0]
                    AP_local[-2].y = CandidateCites[idx_pair[1]][1]

                    Reset_all_APsettings()
                    Reset_STAsetting()
                    IsASolution = Func_TestTheSolution()
                    if IsASolution != 0:
                        if best_local is None or MaxThroughput > best_local["MaxThroughput"]:
                            best_local = {
                                "AP": deepcopy(AP_local),
                                "MaxThroughput": MaxThroughput,
                                "MinThroughput": MinThroughput
                            }
                return best_local

            # 并行计算块
            results = Parallel(n_jobs=n_jobs, backend='loky')(
                delayed(evaluate_block)(block, CandidateCites, AP) for block in tqdm(blocks, desc="并行处理任务块", ncols=80, mininterval=10)
            )

            # 筛选有效结果
            valid_results = [r for r in results if r is not None]

            if len(valid_results) > 0:
                best = max(valid_results, key=lambda r: r["MaxThroughput"])
                AP = deepcopy(best["AP"])
                count += 1
                print(f'✅ 已合并 {count} 个三角形，当前AP数量：{len(AP)}')
                print(f'MaxThroughput={best["MaxThroughput"]}, MinThroughput={best["MinThroughput"]}')
                AP_temp = deepcopy(AP)
                flag = 1
                break
            # ===============================================================

        if flag == 0:
            AP = deepcopy(AP_temp)
            break

    return AP







def Func_ReplaceThreeByTwo():#parallel version1
    global STA
    global AP

    STA_temp = deepcopy(STA)
    AP_temp = deepcopy(AP)
    count = 0
    flag_tri = 1
    Flag_NchoosK = 0

    while True:
        if flag_tri == 1:
            AP = deepcopy(AP_temp)
            pointNumber = len(AP)

            points = np.zeros((pointNumber, 2))
            for i in range(pointNumber):
                points[i, 0] = AP[i].x
                points[i, 1] = AP[i].y

            Points_temp = deepcopy(points)
            PP = Points_temp.tolist()
            AP_INF = deepcopy(PP)

            for Lx in PP[:]:
                if PP.count(Lx) > 1:
                    PP.remove(Lx)

            if len(PP) < 3:
                Flag_NchoosK = 1
            else:
                tri = Delaunay(points)
                XYIndex_to_list = []
                for simplex in tri.simplices:
                    d1 = np.linalg.norm(points[simplex[0]] - points[simplex[1]])
                    d2 = np.linalg.norm(points[simplex[1]] - points[simplex[2]])
                    d3 = np.linalg.norm(points[simplex[2]] - points[simplex[0]])
                    XYIndex_to_list.append([*simplex, d1 + d2 + d3])
                XYIndex_to_list.sort(key=lambda v: v[3])

                APIndex_to_list = []
                for tri_pts in XYIndex_to_list:
                    idx = [
                        AP_INF.index(PP[tri_pts[0]]),
                        AP_INF.index(PP[tri_pts[1]]),
                        AP_INF.index(PP[tri_pts[2]])
                    ]
                    APIndex_to_list.append(idx)
        else:
            test = 0

        if Flag_NchoosK == 1:
            AP = deepcopy(AP_temp)
            APIndex = np.arange(0, len(AP), 1)
            APIndex_com = combinations(APIndex, 3)
            APIndex_to_list = []
            for i in APIndex_com:
                APIndex_to_list.append(list(i))

            for i in range(len(APIndex_to_list)):
                dis_two_aps1 = np.linalg.norm(np.array([AP[APIndex_to_list[i][0]].x, AP[APIndex_to_list[i][0]].y]) -
                                              np.array([AP[APIndex_to_list[i][1]].x, AP[APIndex_to_list[i][1]].y]))
                dis_two_aps2 = np.linalg.norm(np.array([AP[APIndex_to_list[i][2]].x, AP[APIndex_to_list[i][2]].y]) -
                                              np.array([AP[APIndex_to_list[i][1]].x, AP[APIndex_to_list[i][1]].y]))
                dis_two_aps3 = np.linalg.norm(np.array([AP[APIndex_to_list[i][0]].x, AP[APIndex_to_list[i][0]].y]) -
                                              np.array([AP[APIndex_to_list[i][2]].x, AP[APIndex_to_list[i][2]].y]))
                dis_three_aps = dis_two_aps1 + dis_two_aps2 + dis_two_aps3
                APIndex_to_list[i].append(dis_three_aps)
            APIndex_to_list.sort(key=lambda APIndex_to_list: APIndex_to_list[3])

        # ==============================
        # 主循环：测试三角形组合
        # ==============================
        for i in range(len(APIndex_to_list)):
            print(f'STAGE 4: 总尝试 {len(APIndex_to_list)}，进度 {i + 1}/{len(APIndex_to_list)}，当前AP数 {len(AP)}')
            flag = 0
            AP = deepcopy(AP_temp)

            # 删除原三点
            del AP[APIndex_to_list[i][0]]
            del AP[APIndex_to_list[i][1] - 1]
            del AP[APIndex_to_list[i][2] - 2]

            APIndex_loc = np.arange(0, len(CandidateCites), 1)
            APIndex_com_loc = combinations(APIndex_loc, 2)
            APIndex_loc_to_list = [list(i) for i in APIndex_com_loc]

            # ===============================================================
            # ✅ 【修改部分开始】——使用 joblib 并行加速两点组合评估
            print(f"并行评估 {len(APIndex_loc_to_list)} 个两点组合，CPU核数：{cpu_count()}")

            n_jobs = min(cpu_count(), 48)
            results = Parallel(n_jobs=n_jobs, backend='loky')(
                delayed(evaluate_two_locations)(idx_pair, CandidateCites, AP)
                for idx_pair in tqdm(APIndex_loc_to_list, desc="并行评估中", ncols=80)
            )

            valid_results = [r for r in results if r is not None]

            if len(valid_results) > 0:
                best = max(valid_results, key=lambda r: r["MaxThroughput"])
                AP = deepcopy(best["AP"])
                count += 1
                print(f'✅ 已合并 {count} 个三角形，当前AP数量：{len(AP)}')
                print(f'MaxThroughput={best["MaxThroughput"]}, MinThroughput={best["MinThroughput"]}')
                AP_temp = deepcopy(AP)
                flag = 1
                break
            # ✅ 【修改部分结束】
            # ===============================================================

        if flag == 0:
            AP = deepcopy(AP_temp)
            break

    return AP



#### Stage 4: Replace three APs by two
def Func_ReplaceThreeByTwo__original():
    global STA
    global AP

    STA_temp = deepcopy(STA)
    AP_temp = deepcopy(AP)
    # print(id(AP_tem))
    # print(id(AP))
    # print(len(AP_temp))
    #
    # del AP[0]
    # print(len(AP_temp))
    #
    # input()

    count = 0
    flag_tri=1
    Flag_NchoosK = 0

    while True:
        # AP=AP_temp
        # x=[]
        # y=[]
        # for i in range(len(AP)):
        #     x.append(AP[i].x)
        #     y.append(AP[i].y)
        if flag_tri==1:
            AP = deepcopy(AP_temp)
            pointNumber = len(AP)

            points = np.zeros((pointNumber, 2))
            X_temp = []
            Y_temp = []

            for i in range(pointNumber):
                X_temp.append(AP[i].x)
                Y_temp.append(AP[i].y)

            points[:, 0] = X_temp
            points[:, 1] = Y_temp

            Points_temp = deepcopy(points)
            PP = Points_temp.tolist()
            AP_INF = deepcopy(PP)

            # print(PP)
            # print(AP_INF)
            # input()

            for Lx in PP[:]:
                if PP.count(Lx) > 1:
                    PP.remove(Lx)

            if len(PP) < 3:
                Flag_NchoosK = 1
            else:
                tri = Delaunay(points)
                # print(tri.simplices)
                # # print(tri.simplices.shape())
                # print(len(tri.simplices))
                #
                # print(points[tri.simplices])
                #
                # print('test')

                # print(points)
                # print(points.shape)
                #
                #
                # print(PP)
                # input()
                
                # #画三角剖分
                # plt.figure(figsize=(6,6))
                # plt.triplot(points[:,0], points[:,1], tri.simplices, color='blue', linewidth=1)
                # plt.plot(points[:,0], points[:,1], 'ro')  # 绘制点
            
                # # 标注点编号
                # for i, p in enumerate(points):
                #     plt.text(p[0]+0.02, p[1]+0.02, str(i), color='red')
            
                # plt.title("Delaunay 三角剖分示意图", fontsize=14)
                # plt.xlabel("X 坐标")
                # plt.ylabel("Y 坐标")
                # plt.grid(True)
                # plt.axis('equal')
                # plt.show()
                # input()

                XYIndex = np.arange(0, len(PP), 1)
                XYIndex_com = combinations(XYIndex, 3)  #all combinations of the points
                
                XYIndex_com=tri.simplices #all triangles
                
                XYIndex_to_list = []
                for i in XYIndex_com:
                    XYIndex_to_list.append(list(i))
                # print('ComLocsList:', XYIndex_to_list)

                for i in range(len(XYIndex_to_list)):
                    dis_two_aps1 = ((PP[XYIndex_to_list[i][0]][0] - PP[XYIndex_to_list[i][1]][0]) ** 2 + (
                                PP[XYIndex_to_list[i][0]][1] - PP[XYIndex_to_list[i][1]][1]) ** 2) ** (0.5)
                    dis_two_aps2 = ((PP[XYIndex_to_list[i][2]][0] - PP[XYIndex_to_list[i][1]][0]) ** 2 + (
                                PP[XYIndex_to_list[i][2]][1] - PP[XYIndex_to_list[i][1]][1]) ** 2) ** (0.5)
                    dis_two_aps3 = ((PP[XYIndex_to_list[i][0]][0] - PP[XYIndex_to_list[i][2]][0]) ** 2 + (
                                PP[XYIndex_to_list[i][0]][1] - PP[XYIndex_to_list[i][2]][1]) ** 2) ** (0.5)
                    dis_three_aps=dis_two_aps1+dis_two_aps2+dis_two_aps3
                    XYIndex_to_list[i].append(dis_three_aps)
                # print(XYIndex_to_list)

                XYIndex_to_list.sort(key=lambda XYIndex_to_list: XYIndex_to_list[3])

                APIndex_to_list = []
                for i in range(len(XYIndex_to_list)):
                    IInndex = []

                    IInndex.append(AP_INF.index(PP[XYIndex_to_list[i][0]]))
                    IInndex.append(AP_INF.index(PP[XYIndex_to_list[i][1]]))
                    IInndex.append(AP_INF.index(PP[XYIndex_to_list[i][2]]))
                    
                    APIndex_to_list.append(IInndex)
                # print(APIndex_to_list)
                # input()
                    
                

            ##############
            # AP = deepcopy(AP_temp)
            # 
            # APIndex = np.arange(0, len(AP), 1)
            # APIndex_com = combinations(APIndex, 3)
            # APIndex_to_list = []
            # for i in APIndex_com:
            #     APIndex_to_list.append(list(i))
            # 
            # for i in range(len(APIndex_to_list)):
            #     dis_two_aps1 = ((AP[APIndex_to_list[i][0]].x - AP[APIndex_to_list[i][1]].x) ** 2 + (
            #             AP[APIndex_to_list[i][0]].y - AP[APIndex_to_list[i][1]].y) ** 2) ** (0.5)
            #     dis_two_aps2 = ((AP[APIndex_to_list[i][2]].x - AP[APIndex_to_list[i][1]].x) ** 2 + (
            #             AP[APIndex_to_list[i][2]].y - AP[APIndex_to_list[i][1]].y) ** 2) ** (0.5)
            #     dis_two_aps3 = ((AP[APIndex_to_list[i][0]].x - AP[APIndex_to_list[i][2]].x) ** 2 + (
            #             AP[APIndex_to_list[i][0]].y - AP[APIndex_to_list[i][2]].y) ** 2) ** (0.5)
            #     dis_three_aps = dis_two_aps1 + dis_two_aps2 + dis_two_aps3
            #     APIndex_to_list[i].append(dis_three_aps)
            # # print(APIndex_to_list)
            # APIndex_to_list.sort(key=lambda APIndex_to_list: APIndex_to_list[3])

        else:
            test=0

        if Flag_NchoosK==1:



            #================================================================================================================
            AP = deepcopy(AP_temp)

            APIndex = np.arange(0, len(AP), 1)
            APIndex_com = combinations(APIndex, 3)
            APIndex_to_list = []
            for i in APIndex_com:
                APIndex_to_list.append(list(i))

            for i in range(len(APIndex_to_list)):
                dis_two_aps1 = ((AP[APIndex_to_list[i][0]].x - AP[APIndex_to_list[i][1]].x) ** 2 + (
                            AP[APIndex_to_list[i][0]].y - AP[APIndex_to_list[i][1]].y) ** 2) ** (0.5)
                dis_two_aps2 = ((AP[APIndex_to_list[i][2]].x - AP[APIndex_to_list[i][1]].x) ** 2 + (
                            AP[APIndex_to_list[i][2]].y - AP[APIndex_to_list[i][1]].y) ** 2) ** (0.5)
                dis_two_aps3 = ((AP[APIndex_to_list[i][0]].x - AP[APIndex_to_list[i][2]].x) ** 2 + (
                            AP[APIndex_to_list[i][0]].y - AP[APIndex_to_list[i][2]].y) ** 2) ** (0.5)
                dis_three_aps=dis_two_aps1+dis_two_aps2+dis_two_aps3
                APIndex_to_list[i].append(dis_three_aps)
            # print(APIndex_to_list)
            APIndex_to_list.sort(key=lambda APIndex_to_list: APIndex_to_list[3])
            #===============================================================================================================


        #

        for i in range(len(APIndex_to_list)):  # test all edges to see if they can be merged
            i_x=i+1
            print('STAGE 4: The total attempts is {}. The progress is {}.The number of APs: {}.'.format(len(APIndex_to_list), i + 1,len(AP)))
            flag = 0
            AP = deepcopy(AP_temp)
            # print('len_AP', len(AP))
            # input()

            # originalAP1_x=AP[APIndex_to_list[i][0]].x
            # originalAP1_y = AP[APIndex_to_list[i][0]].y
            # originalAP2_x=AP[APIndex_to_list[i][1]].x
            # originalAP2_y = AP[APIndex_to_list[i][1]].y
            # print(APIndex_to_list)
            # print(APIndex_to_list[i][0])
            # print(APIndex_to_list[i][1]-1)
            # print(len(AP))
            # print('progree',i)
            # print('testapp')
            # input()
            # print('glo?',len(AP_temp))

            del AP[APIndex_to_list[i][0]]
            del AP[APIndex_to_list[i][1] - 1]
            del AP[APIndex_to_list[i][2] - 2]

            # print('loc?',len(AP_temp))
            # print(CandidateCites)

            APIndex_loc = np.arange(0, len(CandidateCites), 1)
            # print(APIndex_loc)
            APIndex_com_loc = combinations(APIndex_loc, 2)
            APIndex_loc_to_list = []
            for i in APIndex_com_loc:
                APIndex_loc_to_list.append(list(i))
            # print(APIndex_loc_to_list)
            # input()

            # for Two_newlocation in range(len(APIndex_loc_to_list)):
            for Two_newlocation in tqdm(range(len(APIndex_loc_to_list)), 
                    desc="测试两点候选组合", ncols=80):
                # # print('STAGE 4: The total attempts is {}. The progress is {}.'.format(len(APIndex_to_list), i + 1))
                # print('progress:',i+1)
                # print(len(APIndex_loc_to_list),Two_newlocation)
                # print('STAGE 4: The total attempts is {}. The progress is {}. Subprogress:{}.'.format(len(APIndex_to_list), i_x,Two_newlocation))
                AP.append(AccessPoint())
                AP.append(AccessPoint())
                AP[-1].x = CandidateCites[APIndex_loc_to_list[Two_newlocation][0]][0]
                AP[-1].y = CandidateCites[APIndex_loc_to_list[Two_newlocation][0]][1]
                AP[-2].x = CandidateCites[APIndex_loc_to_list[Two_newlocation][1]][0]
                AP[-2].y = CandidateCites[APIndex_loc_to_list[Two_newlocation][1]][1]

                # print(AP[-1].x,AP[-1].y,AP[-2].x,AP[-2].y)
                # input()



                Reset_all_APsettings()
                Reset_STAsetting()
                IsASolution = Func_TestTheSolution()
                if IsASolution == 0:
                    # print(len(AP))
                    #
                    # if len(AP)==2:
                    #     Plot_fig()
                    #
                    #
                    #     input()
                    del AP[-1]
                    # del AP[-2]
                    del AP[-1]
                else:
                    count = count + 1
                    print('It has merged {} triangles.The number of APs is {}.'.format(count, len(AP)))
                    print('MaxThroughput', MaxThroughput)
                    print('MinThroughput', MinThroughput)





                    AP_temp = deepcopy(AP)
                    flag = 1
                    break
            if flag == 1:
                break
        if flag == 0:
            AP = deepcopy(AP_temp)
            break
    return AP

# AP=Func_ReplaceThreeByTwo()
# NumOfAPs_4=len(AP)
# # # print(NumOfAPs)
# print('The solution is: {},{},{},{}.'.format(NumOfAPs_1,NumOfAPs_2,NumOfAPs_3,NumOfAPs_4))
#
# time_end=time.time()
# time_run=time_end-time_start
# print('time_run=',time_run)
#
# finame='RunTime_{}Mbps_{}dBm_{}P_{}NumI.csv'.format(rhoH,P_InterferenceTop_dBm,P_interference,Num_of_interferences)
# with open(finame,'a+') as fp:
#     fp.write(str(time_run)+"\n")
#
# Solutions=[str(NumOfAPs_1),str(NumOfAPs_2),str(NumOfAPs_3),str(NumOfAPs_4)]
# filename='Solutions_{}Mbps_{}dBm_{}P_{}NumI.csv'.format(rhoH,P_InterferenceTop_dBm,P_interference,Num_of_interferences)
# with open(filename,'a+') as fp:
#     fp.write(",".join(Solutions)+"\n")

def Func_ExhaustiveSearch(NumOfAPs_4):
    global STA
    global AP
    time_start_ES = time.time()
    N=NumOfAPs_4-1 #to see if ES method has a better solution
    # N = NumOfAPs_4   # to see if ES method has a better solution

    # STA_temp = deepcopy(STA)
    # AP_temp = deepcopy(AP)
    # print(id(AP_tem))
    # print(id(AP))
    # print(len(AP_temp))
    #
    # del AP[0]
    # print(len(AP_temp))
    #
    # input()

    APIndex_loc = np.arange(0, len(CandidateCites), 1)
    # print(APIndex_loc)
    APIndex_com_loc = combinations(APIndex_loc, N)
    APIndex_loc_to_list = [] # the combinations of the N locations in len(CandidateCites) locations
    for i in APIndex_com_loc:
        APIndex_loc_to_list.append(list(i))

    for i in range(len(APIndex_loc_to_list)):
        if i%500==0:
            print(f"The total trials are {len(APIndex_loc_to_list)}, the progress is {i}.")
        AP = []
        for j in range(N):

            AP.append(AccessPoint())
            # print("CandidateCites:",CandidateCites)
            # print("APIndex_loc_to_list",APIndex_loc_to_list)
            # print("i and j:",i,j)
            # print("APIndex_loc_to_list[i][j]",APIndex_loc_to_list[i][j])
            # print("AP[j].x=CandidateCites[APIndex_loc_to_list[i][j]][0]",CandidateCites[APIndex_loc_to_list[i][j]][0])
            # print("AP[j].x=CandidateCites[APIndex_loc_to_list[i][j]][0]", CandidateCites[APIndex_loc_to_list[i][j]][1])
            AP[j].x=CandidateCites[APIndex_loc_to_list[i][j]][0]
            AP[j].y = CandidateCites[APIndex_loc_to_list[i][j]][1]



                # print(AP[-1].x,AP[-1].y,AP[-2].x,AP[-2].y)
                # input()

        Reset_all_APsettings()
        Reset_STAsetting()
        IsASolution = Func_TestTheSolution()
        # print("IsASolution=",IsASolution)
        if IsASolution == 1:
            print("The number of APs obtained by the fourth stage: {}".format(NumOfAPs_4))

            print("The number of APs obtained by ES is better than four-stage: {}".format(len(AP)))
            os._exit(1)



    print("=====The number of APs obtained by ES is equal the four-stage: {}".format(len(AP)))
    time_end_ES = time.time()
    time_ES=time_end_ES-time_start_ES
    finame = 'RunTime_{}Mbps_{}dBm_{}P_{}NumI_ESMethod.csv'.format(rhoH, P_InterferenceTop_dBm, P_interference,
                                                          Num_of_interferences)
    with open(finame, 'a+') as fp:
        fp.write(str(time_ES) + "\n")
    print("The number of APs obtained by ES is equal the four-stage: {}".format(NumOfAPs_4))
    print("The execution time is: {}".format(time_ES))

    print("ttttt")
    os._exit(1)

    return AP


def PlotAPs3D():
    global X_StaticSTAs
    global Y_StaticSTAs
    global Z_StaticSTAs
    X_StaticSTAs = np.atleast_1d(X_StaticSTAs).ravel()
    Y_StaticSTAs = np.atleast_1d(Y_StaticSTAs).ravel()
    Z_StaticSTAs = np.atleast_1d(Z_StaticSTAs).ravel()
    
    Lx, Ly, Lz = RegionLength,RegionWidth, 9.0   # m, AP 高度为 Lz
    # 环境尺寸
    Lx, Ly, Lz = RegionLength,RegionWidth, 9.0   # m, AP 高度为 Lz
      
    # 网格布局 (天花板)，48 个候选 AP
    global GridLength
    
    nx, ny = int(Lx/GridLength), int(Ly/GridLength)                  # nx * ny == 48
    n_ap = nx * ny
      
    # 用户
    n_user = NumOfSTAs+Num_Of_Static_STAs
      
 
      
    
      
    # -------------------- 生成候选 AP 与用户位置 --------------------
    xs = np.linspace(Lx/(2*nx), Lx - Lx/(2*nx), nx) #[ 5. 15. 25. 35. 45. 55. 65. 75.]
      
    # print(xs)
      
    ys = np.linspace(Ly/(2*ny), Ly - Ly/(2*ny), ny) #[ 5. 15. 25. 35. 45. 55.]
    # print(ys)
      
    # input()
      
    aps_xy = np.array([[x,y] for y in ys for x in xs])  # shape (48,2)  48个AP的候选位置
    #[[ 5.  5.]
     # [15.  5.]
     # ....
     # [25.  5.]]
      
    # print(aps_xy)
    # input()
      
    aps = np.hstack((aps_xy, np.full((n_ap,1), Lz)))    # shape (48,3), z=Lz
    
    # 合并成 (N,3) 的 NumPy 数组
    users = np.column_stack((X_StaticSTAs, Y_StaticSTAs, Z_StaticSTAs))
    
    global CandidateCites_STA
    # 构造 (NumOfSTAs, 3) 的数组,numofstas are grid points
    CandidateCites_STA = np.array(CandidateCites_STA)  # 转换成 (800,2) 的 NumPy 数组
    
    # print(len(CandidateCites_STA))
    # input()
    
    stas = np.column_stack((
        CandidateCites_STA[:,0],      # x
        CandidateCites_STA[:,1],      # y
        np.full(NumOfSTAs, h_sta)     # z = h_sta
    ))
    
    # 拼接到 users 后面
    users = np.vstack((users, stas))
    
    
    # ===================== 第三个 3D 可视化图（真实高度显示） =====================
    fig2 = plt.figure(figsize=(10, 7))
    ax2 = fig2.add_subplot(111, projection="3d")
    
    # -------------------- 立方体边界 --------------------
    r = [0, int(Lx)]
    s = [0, int(Ly)]
    t = [0, int(Lz)]
    verts = [
        [(r[0], s[0], t[0]), (r[1], s[0], t[0]), (r[1], s[1], t[0]), (r[0], s[1], t[0])],
        [(r[0], s[0], t[1]), (r[1], s[0], t[1]), (r[1], s[1], t[1]), (r[0], s[1], t[1])],
        [(r[0], s[0], t[0]), (r[0], s[0], t[1]), (r[0], s[1], t[1]), (r[0], s[1], t[0])],
        [(r[1], s[0], t[0]), (r[1], s[0], t[1]), (r[1], s[1], t[1]), (r[1], s[1], t[0])],
        [(r[0], s[0], t[0]), (r[1], s[0], t[0]), (r[1], s[0], t[1]), (r[0], s[0], t[1])],
        [(r[0], s[1], t[0]), (r[1], s[1], t[0]), (r[1], s[1], t[1]), (r[0], s[1], t[1])]
    ]
    ax2.add_collection3d(Poly3DCollection(verts, alpha=0.1, facecolor="cyan"))
    
    # -------------------- 用户位置 --------------------
    ax2.scatter(users[:, 0], users[:, 1], users[:, 2],
                c="blue", s=5, label="Reference stations")
    
    
    # import numpy as np
    
    # 假设CSV文件路径是 'ap.csv'
    data = np.genfromtxt(f'ap_info_4S_{rhoH}M_{P_InterferenceTop_dBm}dBm_run{run}.csv', delimiter=' ')
    
    # 添加 z = 9 作为第三列
    z_column = np.full((data.shape[0], 1), 9)
    sel_ap_coords = np.hstack([data, z_column])
    
    # 打印生成的结果
    # print(aps)

    
    
    # -------------------- 候选 AP --------------------
    ax2.scatter(aps[:, 0], aps[:, 1], aps[:, 2],
                c="gray", marker="^", s=40, alpha=0.3, label="Candidate AP locations")
    
    # -------------------- 被选中 AP --------------------
    ax2.scatter(sel_ap_coords[:, 0], sel_ap_coords[:, 1], sel_ap_coords[:, 2],
                c="red", marker="^", s=80, label="Selected AP locations")
    
    # -------------------- 坐标轴比例修正 --------------------
    def set_axes_xy_equal(ax):
        """让 X、Y 等比例，Z 高度保持真实"""
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()
        
        x_range = x_limits[1] - x_limits[0]
        y_range = y_limits[1] - y_limits[0]
        max_range = max(x_range, y_range)
        
        x_middle = np.mean(x_limits)
        y_middle = np.mean(y_limits)
        
        ax.set_xlim3d([x_middle - max_range/2, x_middle + max_range/2])
        ax.set_ylim3d([y_middle - max_range/2, y_middle + max_range/2])
        # Z 高度保持原始
        ax.set_zlim3d(z_limits)
    
    set_axes_xy_equal(ax2)
    
    # -------------------- 视角 --------------------
    ax2.view_init(elev=30, azim=45)  # 可调角度观察立体感
    
    # -------------------- 坐标和图例 --------------------
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")
    ax2.set_zlabel("Z (m)")
    
    
    
    # -------------------- 视角 --------------------
    # ax2.view_init(elev=60, azim=45)  # 更改视角的俯视角度
    
    # -------------------- Z轴标签 --------------------
    # ax2.set_zlabel("Z (m)", labelpad=15)  # 设置Z轴标签并添加适当的间距
    
    # # -------------------- 坐标轴范围 --------------------
    # ax2.set_zlim3d(0, Lz)  # 手动设置Z轴的范围
    
    # -------------------- 坐标和图例 --------------------



    
    
    
    
    ax2.legend()
    # plt.title("3D Layout of Stations and APs")
    plt.savefig(f"FourS_3D_layout_{rhoH}_{P_InterferenceTop_dBm}.png", dpi=300, bbox_inches="tight")
    plt.show()
    input()
    # os._exit(1)



#nloptMethod
if NloptMethod==1:
    # nlopt_ap_sinr_3d.py
    import numpy as np
    import nlopt
    import math
    import random
    import time

    # === 新增全局日志 ===
    feasible_solutions_log = []   # 用于记录 (出现顺序, 可行解 #AP 数量, 约束值, 目标值)
    
    # === 新增：迭代历史记录 ===
    iteration_history_all_processes = []  # 记录所有进程的迭代历史
    
    # -------------------- 可调参数 --------------------
    np.random.seed(0)
    random.seed(0)

    # 环境尺寸
    Lx, Ly, Lz = RegionLength,RegionWidth, 9.0   # m, AP 高度为 Lz

    # 网格布局 (天花板)，48 个候选 AP
    nx, ny = int(Lx/10), int(Ly/10)                  # nx * ny == 48
    n_ap = nx * ny

    # 用户
    n_user = NumOfSTAs+Num_Of_Static_STAs

    # AP 参数
    # coverage = 60.0                # m, 最大有效通信半径.
    coverage =D
    
    
    
    
    a=rhoH

    # NLopt 参数
    n_restarts = 30
    maxeval = 1000

    

    # -------------------- 生成候选 AP 与用户位置 --------------------
    xs = np.linspace(Lx/(2*nx), Lx - Lx/(2*nx), nx) #[ 5. 15. 25. 35. 45. 55. 65. 75.]

    # print(xs)

    ys = np.linspace(Ly/(2*ny), Ly - Ly/(2*ny), ny) #[ 5. 15. 25. 35. 45. 55.]
    # print(ys)

    # input()

    aps_xy = np.array([[x,y] for y in ys for x in xs])  # shape (48,2)  48个AP的候选位置
    #[[ 5.  5.]
     # [15.  5.]
     # ....
     # [25.  5.]]

    # print(aps_xy)
    # input()

    aps = np.hstack((aps_xy, np.full((n_ap,1), Lz)))    # shape (48,3), z=Lz
    #np.full((n_ap,1), Lz)生成一个形状为 (n_ap, 1) 的数组，每个元素都是 Lz
    #np.hstack(( ... ))  水平拼接，把 (x, y) 和 z=Lz 合并成 (x, y, z)。
    # [[ 5.  5.  9.]
    #  [15.  5.  9.]
    #  ...
    #  [25.  5.  9.]]
     
    # print(aps[0][0])
    # input()



    #把用户位置从文件导入，并生成与users同结构的数组。users 是一个 NumPy 数组，而且是二维数组（矩阵形式）。
    # 地板上随机用户
    # users_xy = np.column_stack((np.random.rand(n_user) * Lx, np.random.rand(n_user) * Ly))
    # users = np.hstack((users_xy, np.zeros((n_user,1))))  # z = 0
    #[[12.5, 33.7, 0],
     # [48.2,  5.9, 0],
     #...
     # [70.1, 50.3, 0]]

# 确保它们是一维数组
    X_StaticSTAs = np.atleast_1d(X_StaticSTAs).ravel()
    Y_StaticSTAs = np.atleast_1d(Y_StaticSTAs).ravel()
    Z_StaticSTAs = np.atleast_1d(Z_StaticSTAs).ravel()
    
    # 合并成 (N,3) 的 NumPy 数组
    users = np.column_stack((X_StaticSTAs, Y_StaticSTAs, Z_StaticSTAs))
    
    # print("users.shape =", users.shape)
    # print(users[:5])  # 打印前5个用户坐标
    
    
        # CandidateCites_STA 是 (NumOfSTAs, 2)，取 x,y
    # h_sta 是一个常数，例如 1.5 (米)
    
    # 构造 (NumOfSTAs, 3) 的数组,numofstas are grid points
    CandidateCites_STA = np.array(CandidateCites_STA)  # 转换成 (800,2) 的 NumPy 数组
    
    # print(len(CandidateCites_STA))
    # input()
    
    stas = np.column_stack((
        CandidateCites_STA[:,0],      # x
        CandidateCites_STA[:,1],      # y
        np.full(NumOfSTAs, h_sta)     # z = h_sta
    ))
    
    # 拼接到 users 后面
    users = np.vstack((users, stas))
    
    # print("users.shape =", users.shape)
    # print(users[:5])  # 打印最后 5 个用户看看
    # print(users[-5:])  # 打印最后 5 个用户看看

    def compute_rates_given_active_set(active_idx_set):
        if len(active_idx_set) == 0:
            print("[compute_rates] ⚠️ 空集合，返回零速率")
            return np.zeros(n_user)
    
        active_idx = np.array(sorted(list(active_idx_set)), dtype=int)
        m = len(active_idx)
    
        # 计算距离矩阵并判断覆盖
        dists = np.linalg.norm(users[:, None, :] - aps[active_idx][None, :, :], axis=2)
        in_cov = dists <= coverage
    
        # -------------------- ✅ 新逻辑：软惩罚式覆盖检查 --------------------
        uncovered = np.where(~np.any(in_cov, axis=1))[0]
        # if len(uncovered) > 0:  #do not need to print it
        #     print(f"[compute_rates] ⚠️ 有 {len(uncovered)} 个 STA 未被覆盖，将其速率记为 0。")
    
        # 初始化 AP 对象
        global AP
        AP = []
        for ap_i in range(m):
            ap_ind = active_idx[ap_i]
            AP.append(AccessPoint())
            AP[ap_i].x = aps[ap_ind][0]
            AP[ap_i].y = aps[ap_ind][1]
            AP[ap_i].z = aps[ap_ind][2]
    
        # 调用你的系统函数链
        Reset_all_APsettings()
        Association()
        Func_PowerAdjustment()
        Func_GetNeighbors_All()
        Func_ChannelAssignment()
        Func_PowerReAdjustment()
        Func_GetNeiInt()
        Get_Interference_Of_APs_and_STAs()
        RUassignment_APCoordination()
    
        # 计算吞吐率
        rates = np.array([sta.Throughput for sta in STA])
    
        # -------------------- ✅ 新增：未覆盖用户的速率清零 --------------------
        if len(uncovered) > 0:
            rates[uncovered] = 0.0
    
        min_rate = np.min(rates)
        mean_rate = np.mean(rates)
    
        return rates


    def objective(x, grad):
        """
        目标：最小化 AP 数量，同时对低速 STA 数量进行轻微惩罚
        """
        active_idx = set(np.where(x > 0.5)[0].tolist())
        rates = compute_rates_given_active_set(active_idx) if active_idx else np.zeros(n_user)
        num_under_a = np.sum(rates < a)  # 低于 a 的 STA 数量
        # AP数量 + 权重 * 低速 STA数量
        return float(np.sum(x)) + 0.1 * float(num_under_a)  # 权重 0.1 可调


    
    def min_rate_constraint_val(x): #revised on 2025-10-19
        active_idx = set(np.where(x > 0.5)[0].tolist())
        rates = compute_rates_given_active_set(active_idx) if active_idx else np.zeros(n_user)
        minr = float(np.min(rates))
        val = float(a - minr)
        # print(f"[constraint] #active={len(active_idx):02d}, minr={minr:.3e}, a={a:.3e}, a-minr={val:.3e}")
        return val 
    

    def nlopt_constraint(x, grad):
        return min_rate_constraint_val(x)

    # === 新增：迭代回调函数 (每个进程独立记录迭代历史) ===
    def nlopt_iteration_callback(x, grad, process_id, iter_history):
        obj_val = float(np.sum(x))
        c_val = min_rate_constraint_val(x)
        iter_history.append((obj_val, c_val))
        return False  # COBYLA 不使用返回值


    from multiprocessing import Pool, cpu_count
    import numpy as np
    import random
    import nlopt

    # -------------------- 单进程 NLopt --------------------
    def run_single_nlopt(seed):
        np.random.seed(seed)
        random.seed(seed)
        iter_history = []  # 每个进程的迭代历史
    
        opt = nlopt.opt(nlopt.LN_COBYLA, n_ap)        
        opt.set_lower_bounds([0.0] * n_ap)
        opt.set_upper_bounds([1.0] * n_ap)
        opt.set_min_objective(objective)
        opt.add_inequality_constraint(nlopt_constraint, 1e-6)
        opt.set_maxeval(maxeval)
    
        # ⚠️ 每次迭代记录 sum(x) 和约束值
        # COBYLA 无直接回调，使用 wrapper trick 在 constraint 中记录
        def nlopt_constraint_with_history(x, grad):
            val = nlopt_constraint(x, grad)
            obj_val = float(np.sum(x))
            active_idx = set(np.where(x > 0.5)[0].tolist())
            rates = compute_rates_given_active_set(active_idx) if active_idx else np.zeros(n_user)
            num_under_a = np.sum(rates < a)
            iter_history.append((obj_val, val, num_under_a))  # 多记录一项
            return val

        
        
        
        opt.add_inequality_constraint(nlopt_constraint_with_history, 1e-6)
    
        x0 = np.random.rand(n_ap)
        try:
            x_opt = opt.optimize(x0)
            obj_val = opt.last_optimum_value()
            c_val = min_rate_constraint_val(x_opt)
            feasible = (c_val <= 1e-6)
            log_entry = (seed, int(np.sum(x_opt > 0.5)), obj_val, c_val) if feasible else None
            print(f"[进程 {seed}] 完成: sum(x)={obj_val:.4f}, constraint={c_val:.3e}, 可行={feasible}")
            return x_opt, obj_val, c_val, log_entry, iter_history
        except Exception as e:
            print(f"[进程 {seed}] 发生错误: {e}")
            return None, 1e9, 1e9, None, iter_history
    
    
    # -------------------- 并行 NLopt --------------------
    def run_nlopt_random_restarts_parallel(n_restarts, maxeval):
        global feasible_solutions_log, iteration_history_all_processes
    
        # n_proc = min(cpu_count(), n_restarts)
        n_proc=24
        print(f"🚀 启动 {n_proc} 个进程并行执行 {n_restarts} 次随机重启 NLopt 优化")
    
        with Pool(processes=n_proc) as pool:
            results = pool.map(run_single_nlopt, range(n_restarts))
    
        feasible_solutions_log = []
        best_x, best_obj, best_c = None, 1e9, 1e9
        iteration_history_all_processes = []
    
        for x_opt, obj_val, c_val, log_entry, iter_history in results:
            iteration_history_all_processes.append(iter_history)  # 保存每个进程历史
            if log_entry is not None:
                feasible_solutions_log.append(log_entry)
            if x_opt is not None and c_val <= 1e-6 and obj_val < best_obj:
                best_x, best_obj, best_c = x_opt, obj_val, c_val
    
        print(f"\n✅ 并行优化完成: 最优 sum(x)={best_obj:.4f}, 约束={best_c:.3e}")
        return best_x, iteration_history_all_processes

    import matplotlib.pyplot as plt

    def print_nlopt_trace(iteration_history_all_processes, max_print=10, plot_ap_trend=True):
        """
        打印每个进程的迭代轨迹摘要，并可选绘制最优可行解进程的 AP 数量和未达标 STA 数量趋势图
        iteration_history_all_processes: list of list of (obj_val, constraint_val, num_under_a)
        max_print: 每个进程最多打印多少条记录
        plot_ap_trend: 是否绘制最优进程 AP 数量变化图
        """
        best_process_idx = None
        best_obj_val = float('inf')
    
        # 遍历所有进程，打印迭代摘要
        for i, history in enumerate(iteration_history_all_processes):
            if len(history) == 0:
                continue
            # print(f"\n--- Process {i}, total iterations: {len(history)} ---")
            # for j, (obj_val, cons_val, num_under_a) in enumerate(history[:max_print]):
            #     print(f"Iter {j:3d}: sum(x)={obj_val:.4f}, a-min_rate={cons_val:.4e}, #STA<a={num_under_a}")
            # if len(history) > max_print:
            #     print(f"... ({len(history)-max_print} more iterations skipped)")
    
            # 寻找迭代中可行解（约束 <= 0）的最优解
            for obj_val, cons_val, num_under_a in history:
                if cons_val <= 1e-6 and obj_val < best_obj_val:
                    best_obj_val = obj_val
                    best_process_idx = i
    
        # 如果需要绘图，绘制最优可行解进程的 AP 数量和未达标 STA 数量趋势
        if plot_ap_trend and best_process_idx is not None:
            best_history = iteration_history_all_processes[best_process_idx]
            iterations = list(range(len(best_history)))
            ap_numbers = [obj_val for obj_val, cons_val, num_under_a in best_history]
            sta_under_a = [num_under_a for obj_val, cons_val, num_under_a in best_history]
    
            fig, ax1 = plt.subplots(figsize=(8,5))
    
            color1 = 'tab:blue'
            ax1.set_xlabel("Iteration")
            ax1.set_ylabel("AP (sum(x))", color=color1)
            ax1.plot(iterations, ap_numbers, marker='o', linestyle='-', color=color1, label='AP')
            ax1.tick_params(axis='y', labelcolor=color1)
            ax1.grid(True)
    
            # 右侧 Y 轴
            ax2 = ax1.twinx()
            color2 = 'tab:red'
            ax2.set_ylabel("#STA<a", color=color2)
            ax2.plot(iterations, sta_under_a, marker='x', linestyle='--', color=color2, label='#STA<a')
            ax2.tick_params(axis='y', labelcolor=color2)
    
            # 合并图例
            lines, labels = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines + lines2, labels + labels2, loc='upper right')
    
            plt.title(f"Process {best_process_idx}")
            fig.tight_layout()
            plt.show()
        else:
            print("⚠️ 未找到可行解，无法绘制 AP 数量与未达标 STA 趋势图。")







    # ========================= NLopt + 贪心修复（严格 min_rate） =========================
    # 改动说明：
    # 1. 新增函数 round_and_repair_strict，用于替换原来的 round_and_repair。
    #    - 保证最终解满足 min_rate >= a
    #    - 无法满足时返回 None，并打印提示“无解”
    # 2. 主流程改动：
    #    - 对 NLopt 输出的连续解调用 round_and_repair_strict
    #    - 如果修复失败，则不输出不满足约束的解
    # =========================
    
    def round_and_repair_strict(x_cont, min_rate_target):
        """
        将连续解四舍五入为 0/1 (阈值 0.5)。
        如果四舍五入后的解不满足 min_rate >= min_rate_target，
        用贪心逐步加入最能提升 min_rate 的 AP。
        如果无法提升至 min_rate_target，则返回 None 表示无解。
        """
        x_bin = (x_cont > 0.5).astype(int)
        active_idx = set(np.where(x_bin == 1)[0].tolist())
        
        # 计算当前最小速率
        rates = compute_rates_given_active_set(active_idx)
        minr = float(np.min(rates)) if active_idx else 0.0
        print(f"After rounding: #AP = {len(active_idx)}, min_rate = {minr:.3f} Mbps")
        
        if minr >= min_rate_target:
            return x_bin  # 已满足约束
    
        # 贪心修复循环
        remaining = set(range(n_ap)) - active_idx
        iter_count = 0
        while minr < min_rate_target and remaining:
            best_ap = None
            best_minr = minr
            for ap in list(remaining):
                test_set = active_idx | {ap}
                test_rates = compute_rates_given_active_set(test_set)
                test_minr = float(np.min(test_rates))
                if test_minr > best_minr:
                    best_minr = test_minr
                    best_ap = ap
            if best_ap is None:
                # 无法进一步提升 min_rate
                print("无法通过单点贪心修复满足最小速率约束，无解。")
                
                return None
            active_idx.add(best_ap)
            remaining.remove(best_ap)
            minr = best_minr
            iter_count += 1
            if iter_count % 5 == 0:
                print(f" 修复迭代 {iter_count}, #AP={len(active_idx)}, min_rate={minr:.3f}")
    
        # 最终检查
        if minr < min_rate_target:
            print("贪心修复结束仍无法满足 min_rate，无解。")
            print("All APs have been tried:",len(active_idx))
            return None
    
        # 构造最终二进制解
        x_bin = np.zeros(n_ap, dtype=int)
        x_bin[list(active_idx)] = 1
        print(f"Repair finished: #AP = {x_bin.sum()}, min_rate = {minr:.3f} Mbps")
        return x_bin
    
    # ========================= 主流程改动 =========================
    if __name__ == "__main__":
        t0 = time.time()
        
        x_cont_best, iteration_history_all_processes = run_nlopt_random_restarts_parallel(n_restarts=n_restarts, maxeval=maxeval)



        x_bin = None
        if x_cont_best is not None:
            # 使用严格修复函数替代原来的 round_and_repair
            x_bin = round_and_repair_strict(x_cont_best, a)
    
        if x_bin is None:
            # 连续解或贪心修复都无法满足 min_rate
            print("最终无法找到满足 min_rate >= a 的 AP 部署方案。")
        else:
            active_aps = np.where(x_bin == 1)[0].tolist()
            final_rates = compute_rates_given_active_set(set(active_aps))
            print("\n===== 最终结果 =====")
            print("Active AP indices:", active_aps)
            NumOfAPs_fromNLOPT=int(x_bin.sum())
            print("Number of APs deployed:", int(x_bin.sum()))
            MinRate_fromNLOPT=float(np.min(final_rates))
            print("min rate (Mbps):", float(np.min(final_rates)))
            print("mean rate (Mbps):", float(np.mean(final_rates)))
            print("elapsed time: {:.1f}s".format(time.time() - t0))
            
            # k = 1
            elapsed = time.time() - t0
            msg = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] elapsed time: {elapsed:.1f}s"
            print(msg)
            
            filename = f"Nlopt_log_time_rhoH{rhoH}_dBm{P_InterferenceTop_dBm}.txt"
            with open(filename, "a", encoding="utf-8") as f:
                f.write(msg + "\n")

            
            # 保存 active AP 的坐标到文件
            active_ap_coords = aps[active_aps]  # shape (k,3)
            np.savetxt(f"NLOPT_active_ap_coords_{rhoH}M_{P_InterferenceTop_dBm}dBm_run{run}.csv", active_ap_coords, delimiter=",", 
                       header="x,y,z", comments="", fmt="%.6f")
            print("Active AP coordinates saved to active_ap_coords.csv")
            
            # 读回时跳过第一行表头
            active_ap_coords = np.loadtxt("active_ap_coords_NLopt.csv", delimiter=",", skiprows=1)
            print(active_ap_coords.shape)  # (k, 3)
            print(active_ap_coords)    
            # 绘制迭代轨迹
            print_nlopt_trace(iteration_history_all_processes, max_print=10, plot_ap_trend=True)










        
        
        
        
        
        
        # # -------------------- 可视化结果 --------------------
        # import matplotlib.pyplot as plt
        # from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        
        # fig = plt.figure(figsize=(10, 7))
        # ax = fig.add_subplot(111, projection="3d")
        
        # # 立方体边界
        # r = [0, int(Lx)]
        # s = [0, int(Ly)]
        # t = [0, int(Lz)]
        # verts = [
        #     [(r[0], s[0], t[0]), (r[1], s[0], t[0]), (r[1], s[1], t[0]), (r[0], s[1], t[0])],
        #     [(r[0], s[0], t[1]), (r[1], s[0], t[1]), (r[1], s[1], t[1]), (r[0], s[1], t[1])],
        #     [(r[0], s[0], t[0]), (r[0], s[0], t[1]), (r[0], s[1], t[1]), (r[0], s[1], t[0])],
        #     [(r[1], s[0], t[0]), (r[1], s[0], t[1]), (r[1], s[1], t[1]), (r[1], s[1], t[0])],
        #     [(r[0], s[0], t[0]), (r[1], s[0], t[0]), (r[1], s[0], t[1]), (r[0], s[0], t[1])],
        #     [(r[0], s[1], t[0]), (r[1], s[1], t[0]), (r[1], s[1], t[1]), (r[0], s[1], t[1])]
        # ]
        # ax.add_collection3d(Poly3DCollection(verts, alpha=0.1, facecolor="cyan"))
        
        # # 用户 (蓝色散点, 地板)
        # ax.scatter(users[:, 0], users[:, 1], users[:, 2],
        #            c="blue", s=5, label="Stations")
        
        # # 被选中的 AP (红色三角形, 天花板)
        # sel_ap_coords = aps[active_aps]
        # ax.scatter(sel_ap_coords[:, 0], sel_ap_coords[:, 1], sel_ap_coords[:, 2],
        #            c="red", marker="^", s=80, label="Selected APs")
        
        # # 坐标范围
        # ax.set_xlim(0, Lx)
        # ax.set_ylim(0, Ly)
        # ax.set_zlim(0, Lz)
        # ax.set_xlabel("X (m)")
        # ax.set_ylabel("Y (m)")
        # ax.set_zlabel("Z (m)")
        # ax.legend()
        # plt.title("Optimized AP Deployment and Station Distribution")
        # plt.show()
        
# if plotFigure==1:
    
        # -------------------- 可视化结果 --------------------
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        import matplotlib.patches as mpatches
        from scipy.spatial.distance import cdist
        import numpy as np
        
        # ===================== 3D 可视化 =====================
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")
        
        # -------------------- 立方体边界 --------------------
        r = [0, int(Lx)]
        s = [0, int(Ly)]
        t = [0, int(Lz)]
        verts = [
            [(r[0], s[0], t[0]), (r[1], s[0], t[0]), (r[1], s[1], t[0]), (r[0], s[1], t[0])],
            [(r[0], s[0], t[1]), (r[1], s[0], t[1]), (r[1], s[1], t[1]), (r[0], s[1], t[1])],
            [(r[0], s[0], t[0]), (r[0], s[0], t[1]), (r[0], s[1], t[1]), (r[0], s[1], t[0])],
            [(r[1], s[0], t[0]), (r[1], s[0], t[1]), (r[1], s[1], t[1]), (r[1], s[1], t[0])],
            [(r[0], s[0], t[0]), (r[1], s[0], t[0]), (r[1], s[0], t[1]), (r[0], s[0], t[1])],
            [(r[0], s[1], t[0]), (r[1], s[1], t[0]), (r[1], s[1], t[1]), (r[0], s[1], t[1])]
        ]
        ax.add_collection3d(Poly3DCollection(verts, alpha=0.1, facecolor="cyan"))
        
        # -------------------- 确保 sel_ap_coords 已定义 --------------------
        if x_bin is not None:
            active_aps = np.where(x_bin == 1)[0].tolist()
            
            # print(active_aps) #[8, 20, 27, 38],active_aps是一个列表，它放的是aps这个矩阵中的第几行，即第几个ap被选中
            
            sel_ap_coords = aps[active_aps]
            # print("=====")
            # print(aps)
            # print("=====")
            # print(sel_ap_coords)
            # # [[ 5. 15.  9.]
            # #  [45. 25.  9.]
            # #  [35. 35.  9.]
            # #  [65. 45.  9.]]
            
            # input()
        else:
            active_aps = []
            sel_ap_coords = np.empty((0,3))
        
        # -------------------- 用户颜色区分 --------------------
        user_coords = users[:, :3]
        user_xy = users[:, :2]
        
        if len(sel_ap_coords) > 0:
            ap_xy = sel_ap_coords[:, :2]
            dist_matrix = cdist(user_xy, ap_xy, metric="euclidean")
            min_dists = dist_matrix.min(axis=1)
            covered_mask = min_dists <= coverage
        else:
            covered_mask = np.zeros(len(users), dtype=bool)
        
        # 未覆盖用户 -> 蓝色
        ax.scatter(user_coords[~covered_mask, 0], user_coords[~covered_mask, 1], user_coords[~covered_mask, 2],
                   c="blue", s=5, label="Uncovered Stations")
        # 被覆盖用户 -> 黑色
        ax.scatter(user_coords[covered_mask, 0], user_coords[covered_mask, 1], user_coords[covered_mask, 2],
                   c="black", s=5, label="Covered Stations")
        
        # -------------------- 候选 AP --------------------
        ax.scatter(aps[:, 0], aps[:, 1], aps[:, 2],
                   c="gray", marker="^", s=40, alpha=0.3, label="Candidate APs")
        
        # -------------------- 被选中 AP --------------------
        ax.scatter(sel_ap_coords[:, 0], sel_ap_coords[:, 1], sel_ap_coords[:, 2],
                   c="red", marker="^", s=80, label="Selected APs")
        
        # -------------------- 覆盖圆 --------------------
        theta = np.linspace(0, 2*np.pi, 100)
        for (x, y, z) in aps:
            X = x + coverage * np.cos(theta)
            Y = y + coverage * np.sin(theta)
            Z = np.zeros_like(theta)
            ax.plot(X, Y, Z, c="gray", linestyle="--", alpha=0.2)
        
        for (x, y, z) in sel_ap_coords:
            X = x + coverage * np.cos(theta)
            Y = y + coverage * np.sin(theta)
            Z = np.zeros_like(theta)
            ax.plot(X, Y, Z, c="green", linestyle="-", alpha=0.8, linewidth=1.5)
        
        # -------------------- 坐标轴比例修正 --------------------
        def set_axes_equal(ax):
            """让3D图的 x,y,z 比例相等"""
            x_limits = ax.get_xlim3d()
            y_limits = ax.get_ylim3d()
            z_limits = ax.get_zlim3d()
            x_range = x_limits[1] - x_limits[0]
            y_range = y_limits[1] - y_limits[0]
            z_range = z_limits[1] - z_limits[0]
            max_range = max(x_range, y_range, z_range)
            x_middle = np.mean(x_limits)
            y_middle = np.mean(y_limits)
            z_middle = np.mean(z_limits)
            ax.set_xlim3d([x_middle - max_range/2, x_middle + max_range/2])
            ax.set_ylim3d([y_middle - max_range/2, y_middle + max_range/2])
            ax.set_zlim3d([z_middle - max_range/2, z_middle + max_range/2])
        
        set_axes_equal(ax)
        
        # -------------------- 俯视视角 --------------------
        ax.view_init(elev=90, azim=-90)
        
        # -------------------- 标签和图例 --------------------
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.legend()
        plt.title("Optimized AP Deployment with Coverage Circles")
        plt.show()
        
        # ===================== 第二个 3D 可视化图 =====================
        fig2 = plt.figure(figsize=(10, 7))
        ax2 = fig2.add_subplot(111, projection="3d")
        
        # -------------------- 立方体边界 --------------------
        verts = [
            [(r[0], s[0], t[0]), (r[1], s[0], t[0]), (r[1], s[1], t[0]), (r[0], s[1], t[0])],
            [(r[0], s[0], t[1]), (r[1], s[0], t[1]), (r[1], s[1], t[1]), (r[0], s[1], t[1])],
            [(r[0], s[0], t[0]), (r[0], s[0], t[1]), (r[0], s[1], t[1]), (r[0], s[1], t[0])],
            [(r[1], s[0], t[0]), (r[1], s[0], t[1]), (r[1], s[1], t[1]), (r[1], s[1], t[0])],
            [(r[0], s[0], t[0]), (r[1], s[0], t[0]), (r[1], s[0], t[1]), (r[0], s[0], t[1])],
            [(r[0], s[1], t[0]), (r[1], s[1], t[0]), (r[1], s[1], t[1]), (r[0], s[1], t[1])]
        ]
        ax2.add_collection3d(Poly3DCollection(verts, alpha=0.1, facecolor="cyan"))
        
        # -------------------- 用户位置 --------------------
        ax2.scatter(users[:, 0], users[:, 1], users[:, 2],
                    c="blue", s=5, label="Stations")
        
        # -------------------- 候选 AP --------------------
        ax2.scatter(aps[:, 0], aps[:, 1], aps[:, 2],
                    c="gray", marker="^", s=40, alpha=0.3, label="Candidate APs")
        
        # -------------------- 被选中 AP --------------------
        ax2.scatter(sel_ap_coords[:, 0], sel_ap_coords[:, 1], sel_ap_coords[:, 2],
                    c="red", marker="^", s=80, label="Selected APs")
        
        # -------------------- 坐标轴比例修正 --------------------
        set_axes_equal(ax2)
        
        # -------------------- 视角 --------------------
        ax2.view_init(elev=30, azim=45)  # 可调角度观察立体感
        
        # -------------------- 坐标和图例 --------------------
        ax2.set_xlabel("X (m)")
        ax2.set_ylabel("Y (m)")
        ax2.set_zlabel("Z (m)")
        ax2.legend()
        plt.title("3D Layout of Stations and APs")
        plt.show()


# ===================== 第三个 3D 可视化图（真实高度显示） =====================
        fig2 = plt.figure(figsize=(10, 7))
        ax2 = fig2.add_subplot(111, projection="3d")
        
        # -------------------- 立方体边界 --------------------
        verts = [
            [(r[0], s[0], t[0]), (r[1], s[0], t[0]), (r[1], s[1], t[0]), (r[0], s[1], t[0])],
            [(r[0], s[0], t[1]), (r[1], s[0], t[1]), (r[1], s[1], t[1]), (r[0], s[1], t[1])],
            [(r[0], s[0], t[0]), (r[0], s[0], t[1]), (r[0], s[1], t[1]), (r[0], s[1], t[0])],
            [(r[1], s[0], t[0]), (r[1], s[0], t[1]), (r[1], s[1], t[1]), (r[1], s[1], t[0])],
            [(r[0], s[0], t[0]), (r[1], s[0], t[0]), (r[1], s[0], t[1]), (r[0], s[0], t[1])],
            [(r[0], s[1], t[0]), (r[1], s[1], t[0]), (r[1], s[1], t[1]), (r[0], s[1], t[1])]
        ]
        ax2.add_collection3d(Poly3DCollection(verts, alpha=0.1, facecolor="cyan"))
        
        # -------------------- 用户位置 --------------------
        ax2.scatter(users[:, 0], users[:, 1], users[:, 2],
                    c="blue", s=5, label="Reference stations")
        
        # -------------------- 候选 AP --------------------
        ax2.scatter(aps[:, 0], aps[:, 1], aps[:, 2],
                    c="gray", marker="^", s=40, alpha=0.3, label="Candidate AP locations")
        
        # -------------------- 被选中 AP --------------------
        ax2.scatter(sel_ap_coords[:, 0], sel_ap_coords[:, 1], sel_ap_coords[:, 2],
                    c="red", marker="^", s=80, label="Selected AP locations")
        
        # -------------------- 坐标轴比例修正 --------------------
        def set_axes_xy_equal(ax):
            """让 X、Y 等比例，Z 高度保持真实"""
            x_limits = ax.get_xlim3d()
            y_limits = ax.get_ylim3d()
            z_limits = ax.get_zlim3d()
            
            x_range = x_limits[1] - x_limits[0]
            y_range = y_limits[1] - y_limits[0]
            max_range = max(x_range, y_range)
            
            x_middle = np.mean(x_limits)
            y_middle = np.mean(y_limits)
            
            ax.set_xlim3d([x_middle - max_range/2, x_middle + max_range/2])
            ax.set_ylim3d([y_middle - max_range/2, y_middle + max_range/2])
            # Z 高度保持原始
            ax.set_zlim3d(z_limits)
        
        set_axes_xy_equal(ax2)
        
        # -------------------- 视角 --------------------
        ax2.view_init(elev=30, azim=45)  # 可调角度观察立体感
        
        # -------------------- 坐标和图例 --------------------
        ax2.set_xlabel("X (m)")
        ax2.set_ylabel("Y (m)")
        ax2.set_zlabel("Z (m)")
        ax2.legend()
        # plt.title("3D Layout of Stations and APs")
        plt.savefig(f"Nlopt_3D_layout_{rhoH}_{P_InterferenceTop_dBm}.png", dpi=300, bbox_inches="tight")
        plt.show()


        
        
        from scipy.spatial.distance import cdist

        # ========== 计算用户到所选AP的关联 ==========
        if len(sel_ap_coords) > 0:
            user_coords = np.array(users)[:, :2]   # (N,2)，只取x,y
            ap_coords   = np.array(sel_ap_coords)[:, :2]  # (M,2)
        
            # 计算用户到所有选中AP的距离矩阵 (N,M)
            dist_matrix = cdist(user_coords, ap_coords, metric="euclidean")
        
            # 每个用户选择最近的AP
            min_dists = dist_matrix.min(axis=1)
        
            # 输出每个用户关联的AP距离
            # print("\n===== 用户与其最近AP的距离 =====")
            # for i, d in enumerate(min_dists):
            #     print(f"用户 {i}: 距离最近AP = {d:.2f} m")
        
            # 输出所有用户的最小距离
            print("\n所有用户到最近AP的距离中的最大值:", min_dists.max())
        else:
            print("⚠️ 没有选中的AP，无法计算用户-AP距离")




    
    
    
    
    
    
    
    




    print("Nlopt has been done!")
    if For4s==0:    
        os._exit(1)



# 4StageMethod
if FourStage==1:

    if start_greedy==1:
        AP = Func_Greedy()
        
        filename = 'SolsGreedy_{}Mbps_{}dBm_{}P_{}NumI.csv'.format(rhoH, P_InterferenceTop_dBm, P_interference,
                                                                   Num_of_interferences)
    if start_uniform==1:

        AP=Func_Uniformly()
        filename = 'SolsUniform_{}Mbps_{}dBm_{}P_{}NumI.csv'.format(rhoH, P_InterferenceTop_dBm, P_interference,
                                                                   Num_of_interferences)

    # if start_greedy==1: #why two greedys??
    #     AP = Func_Greedy()
    #     filename = 'SolsGreedy_{}Mbps_{}dBm_{}P_{}NumI.csv'.format(rhoH, P_InterferenceTop_dBm, P_interference,
    #                                                                Num_of_interferences)
    if start_random==1:

        AP=Random()
        filename = 'SolsRandom1_{}Mbps_{}dBm_{}P_{}NumI.csv'.format(rhoH, P_InterferenceTop_dBm, P_interference,
                                                                   Num_of_interferences)

    # Plot_Throughput()
    # input()

    # Plot_fig()
    NumOfAPs_1 = len(AP)
    Temp_time=time.time()
    Time_1=Temp_time-time_start
    if start_greedy==1:
        Greedy_Solutions=[str(NumOfAPs_1)]
        with open(filename, 'a+') as fp:
            fp.write(",".join(Greedy_Solutions) + "\n")
        Greedy_Times=[str(Time_1)]
        with open('GreedyTimes.csv', 'a+') as fp:
            fp.write(",".join(Greedy_Times) + "\n")

    if start_uniform==1:
        Uniform_Solutions=[str(NumOfAPs_1)]
        with open(filename, 'a+') as fp:
            fp.write(",".join(Uniform_Solutions) + "\n")
        Uniform_Times=[str(Time_1)]
        with open('UniformTimes.csv', 'a+') as fp:
            fp.write(",".join(Uniform_Times) + "\n")



    # with open(filename, 'a+') as fp:
    #     fp.write(",".join(Greedy_Solutions) + "\n")


    # print('test greedy AP r')
    # print(AP[0].r_24G)

    if start_greedy==1:

        file_AP_info = "AP_info_Greedy.csv"
        if (os.path.isfile(file_AP_info)):
            os.remove(file_AP_info)
        for i in range(len(AP)):
            file_AP_infos = []
            file_AP_infos.append(str(AP[i].x))
            file_AP_infos.append(str(AP[i].y))
            file_AP_infos.append(str(AP[i].r_24G))
            file_AP_infos.append(str(AP[i].r_5G1))
            file_AP_infos.append(str(AP[i].r_5G2))
            file_AP_infos.append(str(AP[i].r_6G))

            with open(file_AP_info, 'a+') as fp:
                fp.write(",".join(file_AP_infos) + "\n")
    if start_uniform == 1:

        file_AP_info = "AP_info_Uniform.csv"
        if (os.path.isfile(file_AP_info)):
            os.remove(file_AP_info)
        for i in range(len(AP)):
            file_AP_infos = []
            file_AP_infos.append(str(AP[i].x))
            file_AP_infos.append(str(AP[i].y))
            file_AP_infos.append(str(AP[i].r_24G))
            file_AP_infos.append(str(AP[i].r_5G1))
            file_AP_infos.append(str(AP[i].r_5G2))
            file_AP_infos.append(str(AP[i].r_6G))

            with open(file_AP_info, 'a+') as fp:
                fp.write(",".join(file_AP_infos) + "\n")

    if FirstStageOnly==1:
        print("test1")
        Plot_fig()
        print("test2")
        os._exit(1)


# ==================
# ⚠️ 注意：请确保在主文件末尾添加
# ==================



    AP = Func_RemoveAPs()
    NumOfAPs_2 = len(AP)
    Temp_time=time.time()
    Time_2=Temp_time-time_start



    # print('move AP r')
    # print(AP[0].r_24G)
    # print(AP[1].r_24G)
    # print(AP[2].r_24G)
    # print(AP[3].r_24G)
    #
    # print(AP[0].r_6G)
    # print(AP[1].r_6G)
    # print(AP[2].r_6G)
    # print(AP[3].r_6G)
# if __name__ == "__main__":
#     AP=Func_MergeTwoByOne_parallel()
#     print("test done")
#     input()
    # AP = Func_MergeTwoByOne()
    AP =Func_MergeTwoByOne_parallel(parallel_workers=None, chunk_size=16)
    # print('test 2 to 1 AP r')
    # print(AP[0].r_24G)
    # print(AP[1].r_24G)
    # print(AP[2].r_24G)
    # print(AP[3].r_24G)
    #
    # print(AP[0].r_6G)
    # print(AP[1].r_6G)
    # print(AP[2].r_6G)
    # print(AP[3].r_6G)
    #
    # input()




    NumOfAPs_3 = len(AP)
    Temp_time=time.time()
    Time_3=Temp_time-time_start
    Solutions = [str(NumOfAPs_1), str(NumOfAPs_2), str(NumOfAPs_3)]
    filename = 'Sols_{}Mbps_{}dBm_{}P_{}NumI_{}.csv'.format(rhoH, P_InterferenceTop_dBm, P_interference,
                                                            Num_of_interferences, 3)
    with open(filename, 'a+') as fp:
        fp.write(",".join(Solutions) + "\n")


    file_AP_info = "AP_info_3Stage.csv"
    if (os.path.isfile(file_AP_info)):
        os.remove(file_AP_info)
    for i in range(len(AP)):
        file_AP_infos = []
        file_AP_infos.append(str(AP[i].x))
        file_AP_infos.append(str(AP[i].y))
        file_AP_infos.append(str(AP[i].r_24G))
        file_AP_infos.append(str(AP[i].r_5G1))
        file_AP_infos.append(str(AP[i].r_5G2))
        file_AP_infos.append(str(AP[i].r_6G))

        with open(file_AP_info, 'a+') as fp:
            fp.write(",".join(file_AP_infos) + "\n")

    # print('test 2 to 1 AP r')
    # print(AP[0].r_24G)
    # print(AP[1].r_24G)
    # print(AP[2].r_24G)
    # print(AP[3].r_24G)
    #
    # print(AP[0].r_6G)
    # print(AP[1].r_6G)
    # print(AP[2].r_6G)
    # print(AP[3].r_6G)



    #The fourth stage
# if __name__ == '__main__':
    AP = Func_ReplaceThreeByTwo_parallel()
    # AP = Func_ReplaceThreeByTwo()
    NumOfAPs_4 = len(AP)
    # print(NumOfAPs_4)


    # print('test 3 to 2 AP r')
    # print(AP[0].r_24G)
    # print(AP[1].r_24G)
    # print(AP[2].r_24G)
    # print(AP[3].r_24G)
    #
    # print(AP[0].r_6G)
    # print(AP[1].r_6G)
    # print(AP[2].r_6G)
    # print(AP[3].r_6G)

    ap=np.array([[0,0]])
    Radio_result=np.array([[0,0,0,0]])
    Power_result=np.array([[0,0,0,0]])
    Channel_result=np.array([[0,0,0,0]])
    for i in range(len(AP)):
        # print('r_24G test')
        # print(AP[i].r_24G)
        # input()
        ap=np.append(ap,[[AP[i].x,AP[i].y]],axis=0)
        Radio_result = np.append(Radio_result, [[AP[i].r_24G, AP[i].r_5G1,AP[i].r_5G2,AP[i].r_6G]], axis=0)
        Power_result = np.append(Power_result, [[AP[i].P_24G, AP[i].P_5G1, AP[i].P_5G2, AP[i].P_6G]], axis=0)
        Channel_result=np.append(Channel_result,[[AP[i].C_24G, AP[i].C_5G1, AP[i].C_5G2, AP[i].C_6G]], axis=0)

    ap_temp=np.delete(ap,0,axis=0)
    Radio_temp = np.delete(Radio_result, 0, axis=0)
    Power_temp = np.delete(Power_result, 0, axis=0)
    Channel_temp=np.delete(Channel_result, 0, axis=0)

        # print(ap)

    np.savetxt(f'ap_info_4S_{rhoH}M_{P_InterferenceTop_dBm}dBm_run{run}.csv',ap_temp)
    np.savetxt('Radio_info.csv', Radio_temp)
    np.savetxt('Power_info.csv', Power_temp)
    np.savetxt('Channel_info.csv', Channel_temp)




    # file_AP_info = "AP_info_4Stage.csv"
    # if (os.path.isfile(file_AP_info)):
    #     os.remove(file_AP_info)
    # for i in range(len(AP)):
    #     file_AP_infos = []
    #     file_AP_infos.append(str(AP[i].x))
    #     file_AP_infos.append(str(AP[i].y))
    #     file_AP_infos.append(str(AP[i].r_24G))
    #     file_AP_infos.append(str(AP[i].r_5G1))
    #     file_AP_infos.append(str(AP[i].r_5G2))
    #     file_AP_infos.append(str(AP[i].r_6G))
    #
    #     with open(file_AP_info, 'a+') as fp:
    #         fp.write(",".join(file_AP_infos) + "\n")
    print('The solution with the 4S algorithm is: {},{},{},{}.'.format(NumOfAPs_1, NumOfAPs_2, NumOfAPs_3, NumOfAPs_4))

    time_end = time.time()
    time_run = time_end - time_start
    print('time_run=', time_run)

    Times=[str(Time_1),str(Time_2),str(Time_3),str(time_run)]

    finame = 'RunTime_{}Mbps_{}dBm_{}P_{}NumI_FourStage.csv'.format(rhoH, P_InterferenceTop_dBm, P_interference,
                                                          Num_of_interferences)
    with open(finame, 'a+') as fp:
        fp.write(",".join(Times) + "\n")

    # with open(finame, 'a+') as fp:
    #     fp.write(str(time_run) + "\n")

    Solutions = [str(NumOfAPs_1), str(NumOfAPs_2), str(NumOfAPs_3), str(NumOfAPs_4)]
    filename = 'Solutions_{}Mbps_{}dBm_{}P_{}NumI.csv'.format(rhoH, P_InterferenceTop_dBm, P_interference,
                                                              Num_of_interferences)
    with open(filename, 'a+') as fp:
        fp.write(",".join(Solutions) + "\n")
    print('The AP placement with Four-stage has done!')
    print("rhoH=",rhoH)
    print("P_InterferenceTop_dBm=",P_InterferenceTop_dBm)
    
    # print("The number of APs with NLopt is:",NumOfAPs_fromNLOPT)
    
    Plot_fig()
    PlotAPs3D()
    # print(len(STA))



#Get the throughput of each station
    # ReSet_Initial_StateOf_APs_STAs()  # step 1
    # Association()  # step 3
    # Init_P_and_C()  # step 4 adopt the channels and powers obtained from the planneing phase
    # # Get_NeighboringAPList()#step 5
    # # print('call3')
    # Func_GetNeiInt()  # Get interference AP list of each AP
    # Get_Interference_Of_APs_and_STAs()  # step 6
    # # RUassignment()#step 7
    # # print('testRU_line_6217')
    # # input()
    # RUassignment_APCoordination()
    # network_throughput=[]
    # for i in range(NumOfSTAs+Num_Of_Static_STAs):
    #     network_throughput.append(STA[i].Throughput)
    # np.savetxt('Throughput.csv',network_throughput)

    #Get the throughput_simulation of each station.

    if Throughput_model_verify==1:
        # ReSet_Initial_StateOf_APs_STAs()  # step 1
        # Association()  # step 3
        # Init_P_and_C()  # step 4 adopt the channels and powers obtained from the planneing phase
        # # Get_NeighboringAPList()#step 5
        # # print('call3')
        # Func_GetNeiInt()  # Get interference AP list of each AP
        # Get_Interference_Of_APs_and_STAs()  # step 6
        # # RUassignment()#step 7
        # # print('testRU_line_6217')
        # # input()
        # # RUassignment_APCoordination()
        
        Func_TestTheSolution_modelVerify()
        # RUassignment_APCoordination_modelVerify()
        network_throughput=[]
        for i in range(NumOfSTAs+Num_Of_Static_STAs):
            network_throughput.append(STA[i].Throughput_sim)
        np.savetxt('Throughput_SIM.csv',network_throughput)

        network_throughput=[]
        for i in range(NumOfSTAs+Num_Of_Static_STAs):
            network_throughput.append(STA[i].Throughput)
        np.savetxt('Throughput.csv',network_throughput)

    # os._exit(1)







    # the exhaustive search method is added for comparison
    if Exhaustive_search == 1:
        Func_ExhaustiveSearch(NumOfAPs_4)
        
        print("kdkdkd")
        os._exit(1)
    else:
        Plot_fig()
        os._exit(1)

if RandomMethod==1:
    Random()
    file_AP_info = "AP_info_Random.csv"
    if (os.path.isfile(file_AP_info)):
        os.remove(file_AP_info)
    for i in range(len(AP)):
        file_AP_infos = []
        file_AP_infos.append(str(AP[i].x))
        file_AP_infos.append(str(AP[i].y))
        file_AP_infos.append(str(AP[i].r_24G))
        file_AP_infos.append(str(AP[i].r_5G1))
        file_AP_infos.append(str(AP[i].r_5G2))
        file_AP_infos.append(str(AP[i].r_6G))

        with open(file_AP_info, 'a+') as fp:
            fp.write(",".join(file_AP_infos) + "\n")
    filename = 'Random_Sols_{}Mbps_{}dBm_{}P_{}NumI.csv'.format(rhoH, P_InterferenceTop_dBm, P_interference,
                                                              Num_of_interferences)
    # finame='Solution_Random.csv'
    with open(filename, 'a+') as fp:
        fp.write(str(len(AP)) + "\n")
    print('The AP placement with Random method has done!')
    Plot_fig()
    os._exit(1)
    # print('testttt')

# print('ddddddd')
# Plot_fig()




if LocalOptimizerMethod==1:

    LocalOptimizer()
    file_AP_info = "AP_info_LocalOptimizer.csv"
    if (os.path.isfile(file_AP_info)):
        os.remove(file_AP_info)
    for i in range(len(AP)):
        file_AP_infos = []
        file_AP_infos.append(str(AP[i].x))
        file_AP_infos.append(str(AP[i].y))
        file_AP_infos.append(str(AP[i].r_24G))
        file_AP_infos.append(str(AP[i].r_5G1))
        file_AP_infos.append(str(AP[i].r_5G2))
        file_AP_infos.append(str(AP[i].r_6G))

        with open(file_AP_info, 'a+') as fp:
            fp.write(",".join(file_AP_infos) + "\n")
    filename = 'LocalOptimizer_Sols_{}Mbps_{}dBm_{}P_{}NumI.csv'.format(rhoH, P_InterferenceTop_dBm, P_interference,
                                                              Num_of_interferences)
    # finame='Solution_Random.csv'
    with open(filename, 'a+') as fp:
        fp.write(str(len(AP)) + "\n")
    print('The AP placement with LocalOptimizer method has done!')
    Plot_fig()
    os._exit(1)




if plotFigure==1:

    PlotAPs3D()
    os._exit(1)




    
def get_state(AP, STA, Band):
    state = []
    M11 = len(STA)
    n11 = len(AP)
    L_avg = M11 / n11

    for i in range(len(AP)):
        for B in Band:

            # ---------- channel (categorical → normalized) ----------
            ch = getattr(AP[i], f'C_{B}')
            if B == '24G':
                ch_norm = (ch - 1) / 3.0
            elif B == '5G1':
                ch_norm = (ch - 5) / 14.0
            elif B == '5G2':
                ch_norm = (ch - 20) / 7.0
            elif B == '6G':
                ch_norm = (ch - 28) / 30.0

            # ---------- power ----------
            p = getattr(AP[i], f'P_{B}')
            if B == '24G':
                p_norm = (p - 24) / 3.0
            elif B in ['5G1', '5G2']:
                p_norm = (p - 25) / 3.0
            elif B == '6G':
                p_norm = (p - 27) / 3.0

            # ---------- load ----------
            # ---------- load ----------

            
            num_sta = len(getattr(AP[i], f'ID_Of_STAs_{B}'))
            num_sta_norm = num_sta / L_avg
            num_sta_norm = np.clip(num_sta_norm, 0.0, 3.0) / 3.0



            # ---------- interference ----------
            interf = getattr(AP[i], f'Total_Interference_{B}')
            interf_norm = np.log10(interf + 1e-12) / 10.0

            # ---------- SINR aggregation ----------
            sinr_dl = []
            sinr_ul = []
            for sta_id in getattr(AP[i], f'ID_Of_STAs_{B}'):
                sinr_dl.append(getattr(STA[sta_id], f'SINR_{B}'))
                sinr_ul.append(getattr(STA[sta_id], f'SINR_UP_{B}'))

            if len(sinr_dl) > 0:
                avg_sinr_dl = np.mean(sinr_dl) / 40.0
                min_sinr_dl = np.min(sinr_dl) / 40.0
                avg_sinr_ul = np.mean(sinr_ul) / 40.0
            else:
                avg_sinr_dl = 0.0
                min_sinr_dl = 0.0
                avg_sinr_ul = 0.0

            state.extend([
                ch_norm,
                p_norm,
                num_sta_norm,
                interf_norm,
                avg_sinr_dl,
                min_sinr_dl,
                avg_sinr_ul
            ])

    return np.array(state, dtype=np.float32)




def print_actual_actions(stage=""):
    print(f"\n=== Actual Actions (Vector Form) {stage} ===")
    print("[c_24G c_5G1 c_5G2 c_6G  p_24G p_5G1 p_5G2 p_6G]")

    for i in range(NumOfAPs):
        ap = AP[i]
        vec = [
            ap.C_24G, ap.C_5G1, ap.C_5G2, ap.C_6G,
            ap.P_24G, ap.P_5G1, ap.P_5G2, ap.P_6G
        ]
        print(f"AP {i:02d}: {vec}")



def apply_action_to_APs(action_reshape):
# action_reshape values are in [-1, 1]

    for i in range(NumOfAPs):
        # Helper to convert [-1, 1] to [0, 1]
        act = (action_reshape[i] + 1.0) / 2.0
        
        # ---------- CHANNELS ----------
        # 2.4G: Range [1, 4] (Size 3 steps) -> 1 + act * 3
        AP[i].C_24G = int(np.clip(np.round(1 + 3 * act[0]), 1, 4))
        
        # 5G1: Range [5, 19] (Size 14 steps) -> 5 + act * 14
        AP[i].C_5G1 = int(np.clip(np.round(5 + 14 * act[1]), 5, 19))
        
        # 5G2: Range [20, 27] (Size 7 steps) -> 20 + act * 7
        AP[i].C_5G2 = int(np.clip(np.round(20 + 7 * act[2]), 20, 27))
        
        # 6G: Range [28, 58] (Size 30 steps) -> 28 + act * 30
        AP[i].C_6G  = int(np.clip(np.round(28 + 30 * act[3]), 28, 58))

        # ---------- POWER LEVELS (Fixed to your specific ranges) ----------
        # P_24G: [24, 25, 26, 27] -> Base 24, Max Add 3
        AP[i].P_24G = int(np.clip(np.round(24 + 3 * act[4]), 24, 27))
        
        # P_5G1: [25, 26, 27, 28] -> Base 25, Max Add 3
        AP[i].P_5G1 = int(np.clip(np.round(25 + 3 * act[5]), 25, 28))
        
        # P_5G2: [25, 26, 27, 28] -> Base 25, Max Add 3
        AP[i].P_5G2 = int(np.clip(np.round(25 + 3 * act[6]), 25, 28))
        
        # P_6G:  [27, 28, 29, 30] -> Base 27, Max Add 3
        AP[i].P_6G  = int(np.clip(np.round(27 + 3 * act[7]), 27, 30))

        






# input()
def Gen_Loc_of_STAs_from_RWM(i_step):
    # STA_realLoc = np.loadtxt('STA_realLoc.csv', delimiter=',').reshape((NumOfSTAs, TotalSamples, 2))
    # STA_realLocs
    # print('i_step=',i_step)

    for j in range(NumOfSTAs):
        STA[j].x=STA_realLocs[j,i_step,0]
        STA[j].y = STA_realLocs[j, i_step, 1]
        # print(STA[j].x,STA[j].y)
        # input()
    testmove=0 #donot use this case. it is just for test
    if testmove==1:
        for j in range(NumOfSTAs):
            STA[j].x=random.uniform(0,RegionLength)
            STA[j].y=random.uniform(0,RegionWidth)



def reset_initial_state():
    global move_start_location
    ReSet_Initial_StateOf_APs_STAs() #step 1
    # print(len(STA))
    # input()
    # Gen_Loc_of_STAs()#step 2
    # print(total_timesteps % episode_timesteps)
    # print('initial')
    # input()


    # Gen_Loc_of_STAs_from_RWM((total_timesteps % episode_timesteps))

    # In each episode, the initial locations of stations are randomly selected

    # move_start_location=random.randint(0, 2880-episode_timesteps)
    # print('move_start_location=',move_start_location)   
    # move_start_location=0 #guarantee different methods have the same paths
    move_start_location = random.randint(0, TotalSamples - episode_timesteps - 1)
    Gen_Loc_of_STAs_from_RWM(move_start_location)
    # print(len(STA))
    # input()
    Association()#step 3
    Init_P_and_C()#step 4 adopt the channels and powers obtained from the planneing phase

    # Get_NeighboringAPList()#step 5
    # print('call3')
    Func_GetNeiInt()  # Get interference AP list of each AP

    Get_Interference_Of_APs_and_STAs()#step 6
    # RUassignment()#step 7
    # print('testRU_line_6217')
    # input()
    RUassignment_APCoordination()


    #Return 1 x 3*NumOfSTAs+NumOfStaticSTAs
    network_state=get_state(AP, STA, Band)

    return network_state




# print(a)
# print(len(a))
# input()
def Next_timeslot(action):
    # print('test random action (in Next_timeslot)=',action)
    # input()
    # print('reset')
    global move_start_location
    ReSet_Initial_StateOf_APs_STAs()
    Association()
    
    # print("actiontest")

    # print(action)
    
    action = np.clip(action, -1.0, 1.0)
    # print(action)
    
    # print("actiontest22")
    # input()
    action_reshape=action.reshape((NumOfAPs,8))
    print("\n=== Proxy Action (Actor Output) ===")
    print(action_reshape)
    # print('action',action_reshape)
    # input()
    apply_action_to_APs(action_reshape)
    print_actual_actions(stage="After Mapping (Reward Eval)")
   


        # AP[i].P_24G=27
        # AP[i].P_5G1 = 28
        # AP[i].P_5G2 = 28
        # AP[i].P_6G = 30

    #     print(AP[i].C_24G,AP[i].C_5G1,AP[i].C_5G2,AP[i].C_6G)
    # # #     print(AP[i].C)
    #     input()
    # # print('test',NumOfAPs,len(action),AP[9].P)
    # input()

    # Get_NeighboringAPList()
    # print('call4')
    Func_GetNeiInt()  # Get interference AP list of each AP
    Get_Interference_Of_APs_and_STAs()
    # RUassignment()

    # print('testRU_line_6279')
    # input()
    RUassignment_APCoordination()

    # network_state=[]
    global reward_rawMin
    reward_rawMin = float("inf")
    for i in range(NumOfSTAs+Num_Of_Static_STAs):
        # network_state.append(STA[i].Throughput)
        if STA[i].Throughput<reward_rawMin:
            reward_rawMin=STA[i].Throughput

    # print("reward=",reward)
    # input()
    # reward = np.log1p(reward)
    # Replace hard min with soft-min
    beta = 5.0
    thr = np.array([STA[i].Throughput for i in range(len(STA))])
    reward = - (1/beta) * np.log(np.sum(np.exp(-beta * thr)))
    reward = np.log1p(reward)



    ReSet_Initial_StateOf_APs_STAs()

    apply_action_to_APs(action_reshape)

        
    




    #The locations of stations in next step change.
    move_start_location=move_start_location+1
    # Gen_Loc_of_STAs_from_RWM((total_timesteps % episode_timesteps))
    Gen_Loc_of_STAs_from_RWM(move_start_location)
    Association()
    # Init_P_and_C()

    # for i in range(NumOfAPs): #在新的位置坐标上，采用现有的信道，即上一个动作，产生下一个状态
    #     AP[i].C=int(action[i])

    # Get_NeighboringAPList()
    # print('call5')
    Func_GetNeiInt()  # Get interference AP list of each AP
    Get_Interference_Of_APs_and_STAs()
    # RUassignment()
    # print('testRU_line_6329')
    # input()
    RUassignment_APCoordination()

    network_state = get_state(AP, STA, Band)


    return network_state, reward, False


def Func_ES():
    ii=0
    MaxTh=0
    for t in range(4):
        for j in range(4):
            for k in range(4):
                for q in range(4):
                    # for u in range(4):
                    ii=ii+1
                    if ii%100==0:
                        print("progressES:",ii)
                    # print(t,j,k,q)

                    ReSet_Initial_StateOf_APs_STAs()
                    Association()
                    # action_temp=[C_2dot4G[t],5,20,28,27,28,28,30,\
                    #              C_2dot4G[j],5,20,28,27,28,28,30,\
                    #                  C_2dot4G[k],5,20,28,27,28,28,30,\
                    #                  C_2dot4G[q],5,20,28,27,28,28,30]
                    action_temp = [C_2dot4G[t], C_2dot4G[j], C_2dot4G[k], C_2dot4G[q]]
                    # action_temp=np.array(action_temp)
                    # action_temp=action_temp.reshape(NumOfAPs,8)
                    
                    
                    # print(action_temp)
                    # input()

                    for i in range(NumOfAPs):
                        AP[i].C_24G = action_temp[i]
                        
                        # AP[i].C_24G = int(action_reshape[i][0])
                        AP[i].C_5G1 = 5
                        AP[i].C_5G2 = 20
                        AP[i].C_6G = 28
                        # AP[i].C_5G1 = int(action_temp[i][1])
                        # AP[i].C_5G2 = int(action_temp[i][2])
                        # AP[i].C_6G = int(action_temp[i][3])
                        
                        AP[i].P_24G = 27
                        AP[i].P_5G1 = 28
                        AP[i].P_5G2 = 28
                        AP[i].P_6G = 30

                    #     print(AP[i].C_24G,AP[i].C_5G1,AP[i].C_5G2,AP[i].C_6G)
                    # # #     print(AP[i].C)
                    #     input()
                    # # print('test',NumOfAPs,len(action),AP[9].P)
                    # input()

                    # Get_NeighboringAPList()
                    Func_GetNeiInt()  # Get interference AP list of each AP
                    Get_Interference_Of_APs_and_STAs()
                    RUassignment_APCoordination()

                    # network_state=[]
                    reward = float("inf")
                    for i in range(NumOfSTAs + Num_Of_Static_STAs):
                        # network_state.append(STA[i].Throughput)
                        if STA[i].Throughput < reward:
                            reward = STA[i].Throughput

                    if reward>MaxTh:
                        MaxTh=reward
                        action_2dot4G=action_temp
                        # action=action_temp
                        # print(MaxTh)
    # print(action)
    # input()
    
    
    
    
    
    action=[action_2dot4G[0],5,20,28,27,28,28,30,\
                                      action_2dot4G[1],5,20,28,27,28,28,30,\
                                          action_2dot4G[2],5,20,28,27,28,28,30,\
                                          action_2dot4G[3],5,20,28,27,28,28,30]
    
    # action=np.array(action)
    return action


class ReplayBuffer(object):
    def __init__(self, max_size=1e6):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def add(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size) #这里的batch_size是100
        # print(len(ind))
        # input()
        x, y, u, r, d = [], [], [], [], []
        for i in ind:
            X, Y, U, R, D = self.storage[i]
            x.append(np.asarray(X))
            y.append(np.asarray(Y))
            u.append(np.asarray(U))
            r.append(np.asarray(R))
            d.append(np.asarray(D))
        return np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device=torch.device("cpu")

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 400)#神经元数量会不会太小？
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)
        
        self.max_action = max_action
        nn.init.uniform_(self.l3.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.l3.bias, -3e-3, 3e-3)
        

    def forward(self, x):
        x = F.relu(self.l1(x),inplace=False)
        x = F.relu(self.l2(x),inplace=False)              
        # a = torch.tanh(self.l3(x))
        return self.max_action * torch.tanh(self.l3(x))
    

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400 + action_dim, 300)
        self.l3 = nn.Linear(300, 1)

    def forward(self, x, u):
        # x=x.cuda()
        # u=u.cuda()
        x = F.relu(self.l1(x),inplace=False)
        x = F.relu(self.l2(torch.cat([x, u], 1)),inplace=False)#cat， 指的是x和u按列拼接，如2行3列和2行4列按1（列）拼接，成2行7列
        # print(torch.cat([x, u], 1))
        # input()
        x = self.l3(x) #reward是直接输出，没有任可输出范围的约束？（上面的x和u应该是：状态和动作合成一个张量）
        return x

temp_loss1=[]
temp_loss2=[]
class DDPG(object):
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4, weight_decay=1e-3)
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), weight_decay=1e-2)

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, iterations, batch_size, discount, tau):#discount=0.99
        for _ in range(iterations):
            # print(batch_size)
            # input()
            x, y, u, r, d = replay_buffer.sample(batch_size)
            state = torch.FloatTensor(x).to(device)
            action = torch.FloatTensor(u).to(device)
            next_state = torch.FloatTensor(y).to(device)
            done = torch.FloatTensor(1 - d).to(device)
            reward = torch.FloatTensor(r).to(device)
            # print(reward)
            # input()
            target_Q = self.critic_target(next_state, self.actor_target(next_state))
            # print(done)
            # input()
            # print("discount=",discount)
            target_Q = reward + (done * discount * target_Q).detach()

            current_Q = self.critic(state, action)

            critic_loss = F.mse_loss(current_Q, target_Q)#这是不正常的，它有波动
            # if episode_num == 6:
            #     temp_loss1.append(critic_loss.detach().numpy())
            # print(temp_loss1)
            # input()

            temp_loss1.append(critic_loss.detach().cpu().numpy())
            np.savetxt('critic_loss'+str(Gamma) + str(Num_of_interferences) + '_' + str(P_interference) + 'dBm' +Ver+ '.csv',
                       temp_loss1)

            self.critic_optimizer.zero_grad()
            # print('test')
            critic_loss.backward()
            # print('test')
            self.critic_optimizer.step()

            actor_loss = -self.critic(state, self.actor(state)).mean()#这是正常的，一直下降
            # if episode_num == 6:
            #     temp_loss2.append(actor_loss.detach().numpy())
            temp_loss2.append(actor_loss.detach().cpu().numpy())
            np.savetxt('actor_loss'+str(Gamma) + str(Num_of_interferences) + '_' + str(P_interference) + 'dBm' + Ver+'.csv',
                       temp_loss2)
            self.actor_optimizer.zero_grad()
            # print('dddddddd')
            actor_loss.backward()
            # print('dddddd')
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            self.actor_optimizer.step()

            # temp_x=0
            # if episode_num==6:

            #     # temp_x=temp_x+1
            #     # print(temp_x)

            #     # input()
            #     plt.plot(temp_loss1,color="blue")
            #     plt.plot(temp_loss2, color="red")
            #     plt.xlabel('Step')
            #     plt.ylabel('Loss')
            #     plt.legend(['Critic NN loss','Actor NN loss'])
            #
            #     plt.pause(0.1)

            #     #     plt.close()



            # Update model
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    #DDPGSetting
    #DeepRL
DDPG_alg=0
if DDPG_alg==1:
    if __name__ == "__main__":
        # global episode_timesteps
        start = time.time()
        parser = argparse.ArgumentParser()
        parser.add_argument("--env_name", default="WiFi-v1")
        parser.add_argument("--seed", default=0, type=int, help='Sets PyTorch and Numpy seeds')
        # parser.add_argument("--start_timesteps", default=1e4, type=int, help='how many step random policy run')
        # parser.add_argument("--start_timesteps", default=2e3, type=int, help='how many step random policy run')
    
        parser.add_argument("--start_timesteps", default=1000, type=int, help='how many step random policy run')
        parser.add_argument("--max_timesteps", default=5000, type=float, help='max_timesteps')
        episode_timesteps=50
    
        parser.add_argument("--expl_noise", default=0.05, type=float, help='Gaussian exploration')
        parser.add_argument("--batch_size", default=64, type=int, help='Batch size')
        # parser.add_argument("--GAMMA", default=0.99, type=float, help='Discount')
        parser.add_argument("--GAMMA", default=0.0, type=float, help='Discount')
        parser.add_argument("--tau", default=0.005, type=float, help='DDPG update rate')
        parser.add_argument("--policy_noise", default=0.2, type=float, help='Noise to target policy during critic update')
        parser.add_argument("--noise_clip", default=0.5, type=float, help='Range to clip target policy noise')
        parser.add_argument("--policy_freq", default=2, type=int, help=' Frequency of delayed policy updates')
        parser.add_argument("--grad_steps", default=1, type=int, help="gradient updates per environment step")
        args = parser.parse_args()
    
        

        
        # Get APs' locations
        ap=np.loadtxt('ap_info_4S_5M_-68dBm_run1.csv')
        
    
       
    
        # Get STAs' locations
        STA_realLocs = np.loadtxt('STA_realLoc.csv', delimiter=',').reshape((NumOfSTAs, TotalSamples, 2))
    
        
    
        ###Generate APs according to the results of the planning phase for the operation phase
        AP=[]
        for i in range(len(ap)):
            AP.append(AccessPoint())
            AP[i].x=ap[i][0]
            AP[i].y = ap[i][1]
    
        
        NumOfAPs=len(AP)
        state_dim = NumOfAPs* len(Band) * 7
        
        
        
        
        
        # NumOfChannels=(len(C_2dot4G) + len(C_5GI) + len(C_5GII) + len(C_6G))
        # input()
        aps=NumOfAPs*8
        action_dim = aps # Each AP has 4 band; each band has 1 channel. Thus, each AP has 4 channels
        # print(action_dim)
        # input()
    
        #
        # max_action = float(env.action_space.high[0])
        # max_action = float(aps)#有什么用？？
        max_action=1.0
        # print(max_action)
        # input()
        # policy = DDPG(state_dim, action_dim, max_action)
        policy = DDPG(state_dim, action_dim, max_action)#这句话什么意思？是实例化一个DDPG对象plicy
        replay_buffer = ReplayBuffer()
        total_timesteps = 0
        timesteps_since_eval = 0
        episode_num = 0
        episode_reward = 0
        episode_reward_rawMin = 0
    
        # episode_timesteps=100
        done = True
        # done=False
    
        # obs = reset_initial_state()#一开始给一个初始化状态 ok
    
        RewardTest=[]
    
        Result_Episode_Nums=[]
        Result_Rewards=[]
        Execution_time=[]
        Result_Rewards_rawMin=[]
        # np.savetxt('Num_episodes.csv', Result_Episode_Nums)
        #
        # np.savetxt('Result_reward.csv', Result_Rewards)
        file_name='_Sim_softmax'+str(Num_of_interferences)+'_'+str(P_interference)+' dBm'+Ver
        Time_training_start=time.time()
        while total_timesteps < args.max_timesteps+1:
            # print('begin test')
            # input()
            # print(STA[1].x)
            # print(STA[1].y)
    
            if (total_timesteps) % episode_timesteps == 0 and total_timesteps!=0:#一个回合结束
                done=True
                print('Episode=',episode_num)
                
    
    
    
                # input()
    
            if done: #如果一个回合已结束
                
                if total_timesteps != 0:
                    print(("Total T: %d Episode Num: %d Episode T: %d Reward: %f") % (
                    total_timesteps, episode_num, episode_timesteps, episode_reward/episode_timesteps))
                    
                    # print("AP:",len(AP))
                    # print("sta:",len(STA))
                    # input()
                    
                    if FixedResource_f==0 and ES_method_for_oper==0 and total_timesteps >= args.start_timesteps-1:
                        
                        # if total_timesteps>args.start_timesteps:
                        print('test training progress_start!')
                            # if FixedResource == 0:
                            #     print('test training')
                            #     input()
                        Trainin_step_time1=time.time()
                        # policy.train(replay_buffer, episode_timesteps, args.batch_size, args.GAMMA, args.tau)
                        Training_step_time2=time.time()
                        Training_step_time=Training_step_time2-Trainin_step_time1
                        print('Training_step_time:',Training_step_time)
                        print('test training progress_end!')
                    episode_reward=episode_reward/episode_timesteps
                    Result_Episode_Nums.append(episode_num)
                    Result_Rewards.append(episode_reward)
                    # os.remove('Num_episodes.csv')
                    # np.savetxt('Num_episodes.csv', Result_Episode_Nums)
                    
                    episode_reward_rawMin=episode_reward_rawMin/episode_timesteps
                    Result_Rewards_rawMin.append(episode_reward_rawMin)
                    
    
                    # os.remove('Result_reward.csv')
                    # Num_of_interferences
                    # P_interference
                    
                    if ES==1 and ES_method_for_oper==1:
                        np.savetxt('ES_Reward' + str(Gamma) + str(Num_of_interferences) + '_' + str(P_interference) + 'dBm' + '.csv',Result_Rewards)
                    
                    
                    np.savetxt('Reward'+str(Gamma)+str(Num_of_interferences)+'_'+str(P_interference)+'dBm'+Ver+'.csv', Result_Rewards)
                    np.savetxt('Reward_raw'+str(Gamma)+str(Num_of_interferences)+'_'+str(P_interference)+'dBm'+Ver+'.csv', Result_Rewards_rawMin)
    
                    if plotReward==1:
                        
                        print("total_timesteps--plot test",total_timesteps,flush=True)
    
                        plt.plot(Result_Episode_Nums,Result_Rewards,'gp-')
                        # plt.plot(episode_num,episode_reward,'gp-')
                        plt.xlabel('Number of episode')
                        plt.ylabel('Average reward (Mbps)')
                        plt.title('Interference source power = ' + str(
                            P_interference) + ' dBm' + '\n' + 'Number of interference source = ' + str(Num_of_interferences))
                        
                        if total_timesteps==(args.max_timesteps):
                            plt.savefig('Result'+str(Gamma)+file_name+'.pdf')
                        plt.pause(0.1)
    
    
               
                obs = reset_initial_state()
                # print(obs)
                # input()
                done = False
                episode_reward = 0
    
                # episode_num += 1
                episode_num = episode_num+1
    
            if total_timesteps < args.start_timesteps:
                
    
    
                if FixedResource_f == 1:
                    print('Fixed allocation stage steps:', total_timesteps)
    
                    action = FixedResource()
                    # print(action)
                    # input()
                elif ES==1 and ES_method_for_oper==1:
                    print('Exhaustive search steps:',total_timesteps)
                    
                    #What is the idea at that time?
                    
                    EStime_start=time.time()
    
                    action=Func_ES()
                    EStime_end=time.time()
                    ES_oper_time=EStime_end-EStime_start
                    print("ES_time=",ES_oper_time)
                    
                    # filename = 'Random_Sols_{}Mbps_{}dBm_{}P_{}NumI.csv'.format(rhoH, P_InterferenceTop_dBm, P_interference, Num_of_interferences)
                    # finame='Solution_Random.csv'
                    with open("ES_OperationPhase_runtime.csv", 'a+') as fp:
                        fp.write(str(ES_oper_time) + "\n")
                    print(action)
                    # input()
                    
                    
                else:
                    print('Start_timesteps=',args.start_timesteps)
                    print('Random stage steps:', total_timesteps)
    
                    # action = FixedResource()
    
                    action = Random_C() #Random_C()结果是离散的，是不是要改为连续的？？不用吧，就算是浮点数，不也是离散的？
                    
    
            else:
                print('total_timesteps=',total_timesteps)
                
                print("episode_num:",episode_num)
                # print('Ori_action',action)
                
                actt=np.array(action)
                t_action=actt.reshape(NumOfAPs,8)
                
    
                Time_training_end = time.time()
                Training_time=Time_training_end-Time_training_start
                print('Training_time:',Training_time)
                t_1=time.time()
    
                # Select a method
                # RandomResource_f=1
                if FixedResource_f == 1:
                    # action = policy.select_action(np.array(obs))
                    action = FixedResource()
                else:
                    action = policy.select_action(np.array(obs))
    
    
                t_2=time.time()
                Action_Time=t_2-t_1
    
                Action_Time_t= [str(Action_Time)]
                with open('Action_Time_avg.csv', 'a+') as fp:
                    fp.write(",".join(Action_Time_t) + "\n")
                    
                    
                    
                # if episode_num% 2==0 and total_timesteps<500 and total_timesteps>50:
                #     args.expl_noise=1
                # else:
                #     args.expl_noise=0
    
    
    # input()
                # ===== exploration noise (proxy action space) =====
                if args.expl_noise > 0:
                    noise = np.random.normal(
                        loc=0.0,
                        scale=args.expl_noise,
                        size=action.shape
                    )
                    action = action + noise
                    action = np.clip(action,-1.0, 1.0)

                
    
            # to move next step, total_timesteps += 1
            # total_timesteps += 1
            total_timesteps=total_timesteps+1
    
            # print('test Random action=',action)
            # input()
    
            new_obs, reward, done = Next_timeslot(action) #Perform action and move to next state
            RewardTest.append(reward)
    

            episode_reward=episode_reward+reward
            episode_reward_rawMin=episode_reward_rawMin+reward_rawMin
    
            # replay_buffer.add((obs, new_obs, action, reward, done_bool))
            replay_buffer.add((obs, new_obs, action, reward, float(done)))
            # print(replay_buffer.size())
            
            
            # =============== TRAIN EVERY ENV STEP ===============
            if FixedResource_f == 0 and ES_method_for_oper == 0 \
               and total_timesteps >= args.start_timesteps \
               and len(replay_buffer.storage) >= args.batch_size:
                print("traning.....")
            
                policy.train(
                    replay_buffer,
                    iterations=args.grad_steps,   # <<< THIS IS THE ONLY USE
                    batch_size=args.batch_size,
                    discount=args.GAMMA,
                    tau=args.tau
                )
            # input()
            obs = new_obs
            # print(action)
            # input()
    
            # total_timesteps += 1
            # timesteps_since_eval += 1
            timesteps_since_eval=timesteps_since_eval+1
    
    
        end=time.time()
        Execution_time.append(end-start)
        print(Execution_time)
        np.savetxt('Execution_time.csv', Execution_time)
    
        print("Execution time:",Execution_time)
        os._exit(1)
        
        

        
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import random
import time
import argparse
import os
import matplotlib.pyplot as plt
import sys
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
# ===============================
# A2C Network (Proper Version)
# ===============================
class A2C_Network(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.max_action = max_action

        self.shared = nn.Sequential(
            nn.Linear(state_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU()
        )

        # Actor
        self.mu = nn.Linear(300, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        # Critic: V(s)
        self.value = nn.Linear(300, 1)

        nn.init.uniform_(self.mu.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.value.weight, -3e-3, 3e-3)

    def forward(self, state):
        feat = self.shared(state)
        return self.mu(feat), self.log_std.expand_as(self.mu(feat)), self.value(feat)

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        mu, log_std, _ = self.forward(state)
        std = log_std.exp()
        dist = Normal(mu, std)
        z = dist.sample()
        action = torch.tanh(z) * self.max_action
        return action.cpu().data.numpy().flatten()


class A2C:
    def __init__(self, state_dim, action_dim, max_action):
        self.net = A2C_Network(state_dim, action_dim, max_action).to(device)
        self.network = self.net   # <<< ADD THIS LINE
        self.optim = optim.Adam(self.net.parameters(), lr=3e-4)
        self.gamma = 0.99
        self.value_coef = 0.5
        self.entropy_coef = 0.01


    def update(self, trajectory):
        states, actions, rewards, dones = trajectory

        states = torch.FloatTensor(states).to(device)
        actions = torch.FloatTensor(actions).to(device)

        with torch.no_grad():
            _, _, values = self.net(states)
            values = values.squeeze()

        returns = []
        R = 0
        for r, d in zip(reversed(rewards), reversed(dones)):
            R = r + self.gamma * R * (1 - d)
            returns.insert(0, R)

        returns = torch.FloatTensor(returns).to(device)
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages = advantages.clamp(-5.0, 5.0)   # ADD HERE

        mu, log_std, values_pred = self.net(states)
        std = log_std.exp()
        dist = Normal(mu, std)

        log_probs = dist.log_prob(actions).sum(dim=1)
        entropy = dist.entropy().sum(dim=1).mean()

        policy_loss = -(log_probs * advantages.detach()).mean()
        value_loss = F.mse_loss(values_pred.squeeze(), returns)

        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
        self.optim.step()

        return {"policy_loss": policy_loss.item(),"value_loss": value_loss.item(),"total_loss": loss.item()
}

# --------------------------
# Main Training Pipeline (Modified for A2C)

# --------------------------

A2Cmethod=1
if A2Cmethod==1:
    if __name__ == "__main__":
        start = time.time()
        parser = argparse.ArgumentParser()
        parser.add_argument("--env_name", default="WiFi-v1")
        parser.add_argument("--seed", default=0, type=int)
        parser.add_argument("--start_timesteps", default=0, type=int)  # Same as DDPG a2c do not need random stage, so is 0
        parser.add_argument("--max_timesteps", default=4000, type=float)  # Same as DDPG
        episode_timesteps=100
        parser.add_argument("--expl_noise", default=0.05, type=float)  # Same exploration noise as DDPG
        parser.add_argument("--batch_size", default=64, type=int)  # Not used (A2C trains on trajectories)
        parser.add_argument("--tau", default=0.005, type=float)  # Not used (A2C has no target networks)
        args = parser.parse_args()
    
        # Set seeds (for reproducibility)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
    
        # Load AP/STA locations (same as DDPG)
        ap = np.loadtxt('ap_info_4S_5M_-64dBm.csv')
        STA_realLocs = np.loadtxt('STA_realLoc.csv', delimiter=',').reshape((NumOfSTAs, TotalSamples, 2))
    
        # Initialize APs and STAs
        AP = []
        for i in range(len(ap)):
            AP.append(AccessPoint())
            AP[i].x = ap[i][0]
            AP[i].y = ap[i][1]
        NumOfAPs = len(AP)
        # STA = [STA() for _ in range(NumOfSTAs + Num_Of_Static_STAs)]
    
        # State/Action dimensions (same as DDPG)
        state_dim = NumOfAPs * len(Band) * 7
        action_dim = NumOfAPs * 8  # 4 bands × (channel + power) = 8 per AP
        max_action = 1.0  # Same as DDPG
    
        # Initialize A2C agent
        agent = A2C(state_dim, action_dim, max_action)
    
        # Training variables
        total_timesteps = 0
        episode_num = 0
        episode_reward = 0.0
        episode_reward_rawMin = 0.0
        done = True
        obs = None
    
        # Logging variables
        Result_Episode_Nums = []
        Result_Rewards = []
        Result_Rewards_rawMin = []
        Execution_time = []
        loss_log = []
    
        # A2C trajectory buffer (collects data for 1 episode)
        trajectory = {
            "states": [],
            "actions": [],
            "rewards": [],
            
            
            "done": False
        }
    
        file_name = '_A2C_Sim_softmax' + str(Num_of_interferences) + '_' + str(P_interference) + ' dBm' + Ver
        Time_training_start = time.time()
    
        while total_timesteps < args.max_timesteps + 1:
            # Reset episode if done
            if done:
                if total_timesteps != 0:
                    # Log episode results
                    avg_reward = episode_reward / episode_timesteps
                    avg_reward_rawMin = episode_reward_rawMin / episode_timesteps
                    print(("Total T: %d Episode Num: %d Episode T: %d Avg Reward: %f Avg RawMin Reward: %f") % (
                        total_timesteps, episode_num, episode_timesteps, avg_reward, avg_reward_rawMin),flush=True)
                    Result_Episode_Nums.append(episode_num)
                    Result_Rewards.append(avg_reward)
                    Result_Rewards_rawMin.append(avg_reward_rawMin)
    
                    # Save results
                    np.savetxt('A2C_Reward' + str(Gamma) + str(Num_of_interferences) + '_' + str(P_interference) + 'dBm' + Ver + '.csv', Result_Rewards)
                    np.savetxt('A2C_Reward_raw' + str(Gamma) + str(Num_of_interferences) + '_' + str(P_interference) + 'dBm' + Ver + '.csv', Result_Rewards_rawMin)
    
                    # Plot reward
                    if plotReward == 1:
                        plt.plot(Result_Episode_Nums, Result_Rewards, 'rp-', label='A2C')
                        plt.xlabel('Number of episode')
                        plt.ylabel('Average reward (Mbps)')
                        plt.title('Interference source power = ' + str(P_interference) + ' dBm' + '\n' + 'Number of interference source = ' + str(Num_of_interferences))
                        plt.legend()
                        if total_timesteps == args.max_timesteps:
                            plt.savefig('A2C_Result' + str(Gamma) + file_name + '.pdf')
                        plt.pause(0.1)
    
                    # Train A2C after collecting 1 episode (if past start_timesteps)
                    if total_timesteps >= args.start_timesteps and FixedResource_f == 0 and ES_method_for_oper == 0:
                        print("A2C Training...")
                        dones = [0] * (len(trajectory["rewards"]) - 1) + [1]

                        train_losses = agent.update((
                            trajectory["states"],
                            trajectory["actions"],
                            trajectory["rewards"],
                            
                            
                            dones
                        ))
                        loss_log.append(train_losses)
                        print(f"Training Losses: Policy={train_losses['policy_loss']:.4f}, Value={train_losses['value_loss']:.4f}, Total={train_losses['total_loss']:.4f}")
    
                # Reset trajectory buffer
                trajectory = {"states": [], "actions": [], "rewards": [],  "done": False}
                # Reset environment
                obs = reset_initial_state()
                done = False
                episode_reward = 0.0
                episode_reward_rawMin = 0.0
                episode_num += 1
    
            # Select action
            if total_timesteps < args.start_timesteps:
                # Random exploration (same as DDPG)
                print('Random stage steps:', total_timesteps)
                action = np.random.uniform(-1.0, 1.0, size=action_dim)
            else:
                # A2C policy + exploration noise (same as DDPG's exploration)
                print('A2C policy stage steps:', total_timesteps)
                action = agent.network.select_action(obs)

    
            # Record state and action in trajectory
            trajectory["states"].append(obs.copy())
            trajectory["actions"].append(action.copy())
    
            # Get value estimate for current state (for GAE)

    
            # Execute action
            new_obs, reward, done = Next_timeslot(action)
            total_timesteps += 1
    
            # Record reward
            trajectory["rewards"].append(reward)
            episode_reward += reward
            episode_reward_rawMin += reward_rawMin
            # ADD HERE
            if len(trajectory["rewards"]) >= episode_timesteps:
                done = True

    
            # Update next_value and done for trajectory (at end of episode)
            if done:
                trajectory["done"] = done
                
            else:
                # Get value of next state (for GAE)
                next_state_tensor = torch.FloatTensor(new_obs.reshape(1, -1)).to(device)
                _, _, next_value = agent.network(next_state_tensor)
                
    
            # Update observation
            obs = new_obs
    
        # Final logging
        end = time.time()
        Execution_time.append(end - start)
        print("A2C Execution time:", Execution_time)
        np.savetxt('A2C_Execution_time.csv', Execution_time)
        np.savetxt('A2C_Loss_Log.csv', np.array([[l['policy_loss'], l['value_loss'], l['total_loss']] for l in loss_log]))
        os._exit(1)
            
        

