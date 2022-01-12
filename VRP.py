# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 10:28:16 2021

@author: stijn
"""

import numpy as np
from gurobipy import *
import matplotlib.pyplot as plt
import pandas as pd
import math

#%% ----- Problem -----

model = Model('Vehicle Routing Problem')

#%% ----- Data -----

#cost matrix input
data_input1 = pd.read_excel (r'C:/Users/stijn/Dropbox/Stijn/Research Project/Python/03 Final models/VRP/VRP_input.xlsx', sheet_name='Cost matrix')
data_input1 = data_input1.iloc[: , 1:]        #delete first column of pandas dataframe
cm = data_input1.values.tolist()              #cost matrix

#location input
data_input2 = pd.read_excel (r'C:/Users/stijn/Dropbox/Stijn/Research Project/Python/03 Final models/VRP/VRP_input.xlsx', sheet_name='Location')
LOC_ID = data_input2['LOC_ID'].tolist()
xc     = data_input2['XCOORD'].tolist()
yc     = data_input2['YCOORD'].tolist()
S_TIME = data_input2['SERVICE_TIME'].tolist()
DEMAND = data_input2['DEMAND'].tolist()

d = 1                                                       #number of depots
n = len(LOC_ID) - d                                         #number of customers (total minus depot)
Q = 3                                                       #vehicle capacity
M = 1000000                                    

#%% ----- Sets and indices -----

C = [i for i in range (1, n+1)]                             #set of customers
N = [0] + C                                                 #set of nodes
A = [(i, j) for i in N for j in N if i != j]                #set of arcs

#%% ----- Parameters -----

c = {(i, j): np.hypot(xc[i]-xc[j],yc[i]-yc[j]) for i, j in A}         #euclidean distance
# c = {(i, j): tm[i][j] for i, j in A}                                  #cost matrix distance
s = S_TIME
q = {i: DEMAND[i] for i in N}                                         #demand of the customer

#%% ----- Decision variables -----

x = {}
for i in N:
    for j in N:
        x[i,j] = model.addVar (lb = 0, vtype = GRB.BINARY)
        
T = {}
for i in N:
    T[i] = model.addVar (lb = 0, vtype = GRB.CONTINUOUS)
    
L = {}
for i in N:
    L[i] = model.addVar (lb = 0, vtype = GRB.CONTINUOUS)

#%% ----- Objective function (minimize total distance) -----

Total_distance_travelled = quicksum (c[i,j]*x[i,j] for i in N for j in N if i != j)

model.setObjective (Total_distance_travelled)
model.modelSense = GRB.MINIMIZE
model.update ()

#%% ----- Constraints -----

con1 = {}
for i in C:
    con1[i,j] = model.addConstr(quicksum(x[i,j] for j in N) == 1)
    
con2 = {}
for j in C:
    con2[i,j] = model.addConstr(quicksum(x[i,j] for i in N) == 1)
        
con3 = {}
for i in N:
    con3[i,i] = model.addConstr(x[i,i] == 0)    
    
# con4 = {}
# for j in N:
#     con4[i,j] = model.addConstr(quicksum(x[i,j] for i in N) == quicksum(x[j,i] for i in N))

con5 = {}
for i in N:
    for j in N:
        if j >= d:
            if i != j:
                con5[i,j] = model.addConstr(T[i] + c[i,j]*x[i,j] + s[i] - M*(1-x[i,j]) <= T[j])

con6 = {}
for i in N:
    for j in N:
        if j >= d:
            con6[i,j] = model.addConstr(L[i] - q[j] + M*(1-x[i,j]) >= L[j])
            
con7 = {}
for i in N:
    con7[i] = model.addConstr(L[i] + q[i] <= Q)

#%% ----- Solve -----

model.update ()

model.setParam( 'OutputFlag', True) # silencing gurobi output or not
model.setParam ('MIPGap', 0);       # find the optimal solution
model.write("output.lp")            # print the model in .lp format file

model.optimize ()

#%% ----- Results -----

print ('\n--------------------------------------------------------------------\n')
if model.status == GRB.Status.OPTIMAL:                          # If optimal solution is found
    print ('Minimal distance : %10.2f ' % model.objVal)
    print('\nFinished\n')
else:
    print ('\nNo feasible solution found\n')

active_arcs = [a for a in A if x[a].x > 0.99]
print (('Route : '), sorted(active_arcs))

fig, ax = plt.subplots(figsize=(6,6))
plt.xlim([0, 6])
plt.ylim([0, 6])

for i, j in active_arcs:
    plt.plot([xc[i], xc[j]], [yc[i], yc[j]], c='grey', zorder=0)
plt.plot(xc[0], yc[0], c='black', marker='s', markersize=10)
plt.scatter(xc[1:], yc[1:], c='purple')

#plot LOC_ID next to points
for i, txt in enumerate(LOC_ID):
    plt.annotate(txt, (xc[i], yc[i]), xytext=(xc[i]+0.12, yc[i]+0.25), bbox=dict(boxstyle="round", alpha=0.1))

print("\nNumber of vehicles necessary is: " + str(int(math.ceil(n/Q))))

#print decision variable T = Time counter (TC)
print ('\nArrival time:')
print ('%16.0f' % LOC_ID[0] + '%8.0f' % 0)

for i in C: 
    AT_C1 = LOC_ID[i]               #arrival time column 1
    AT_C2 = T[i].x                  #arrival time column 2
    AT = '%16.0f' % AT_C1 + '%8.0f' % AT_C2
    print(AT)

print ('\nLoad counter:')
print ('%16.0f' % LOC_ID[0] + '%8.0f' % Q)
    
for i in C:
    LC_C1 = LOC_ID[i]               #load counter column 1
    LC_C2 = L[i].x                  #load counter column 2
    LC = '%16.0f' % LC_C1 + '%8.0f' % LC_C2
    print(LC)





