# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 23:47:36 2022

@author: sinha
"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform
from mpl_toolkits.mplot3d.axes3d import Axes3D
import math

plt.rcParams.update({'font.size':24})
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams.update({'font.weight':'bold'})
plt.rcParams["font.family"] = "Times New Roman"
rc = {"font.family" : "serif", 
      "mathtext.fontset" : "stix"}
plt.rcParams.update(rc)
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]


##################################################################
# Anlytical solution
##################################################################
mu_0=4*np.pi*1e-7    # in N/A**2
gamma=1.76*1e11      # in 1/(s.T)
Hz=0                  # in A/m
alpha=0.1#0.0135
h_cross=1.054*1e-34
theta_SHE=0.5
AMTJ=np.pi*0.25*15*15*1e-18
ASHM=15*15*1e-18
t_SHM=3.15*1e-9
lambda_sf=2.1*1e-9
Ms=300*1000#450*1000
Ic=0#-1e-5
P_SHE=(AMTJ/ASHM)*theta_SHE*(1-(1/np.cosh(t_SHM/lambda_sf)))
sigma=np.array([0,0,1])#0,1,0
Is=P_SHE*Ic
V=AMTJ*0.5*1e-9
k_b=1.38*1e-23
k_u=0.0044*k_b*300/V #0.0044*k_b*300/V
u=np.array([0,0,1])
D=np.array([0.066,0.911,0.022])
e=1.6*1e-19

w=mu_0*gamma*Hz  # in 1/s

t_coarse_ns=np.linspace(0,2,501) # in ns
t_coarse=t_coarse_ns*1e-9                 # in s

# mx=0.5*np.cos(w*t_coarse)
# my=0.5*np.sin(w*t_coarse)
# mz=(np.sqrt(3)/2.0)*np.ones(len(t_coarse))

m_vec_mag=np.zeros(len(t_coarse))

# for i in range(len(t_coarse)):
# 	m_vec_mag[i]=np.sqrt((mx[i])**2+(my[i])**2+(mz[i])**2)
# 	
##################################################################
# Numerical solution
##################################################################
def LLGS(m,H,ti):
    global mu_0,gamma,alpha
    Hth=np.random.normal(0,np.sqrt(2*alpha*k_b*300*10**14/(abs(gamma)*Ms*V*5)))*200000#*1
    #print(Hth)
    H=-4*np.pi*Ms*m[2]*np.array([0,0,1])+np.array([Hth,Hth,Hth])+(2*k_u/(mu_0*Ms))*np.dot(m,u)*u#m[0]*np.array[1,0,0]
    #H=(2*k_u/(mu_0*Ms))*np.dot(m,u)*u*math.sin(math.pi*ti/500)-Ms*np.multiply(D,m)*math.sin(math.pi*ti/500)
    precision=-(gamma*mu_0)*np.cross(m,H)
    dmdt=(precision/(1+alpha**2))+alpha*np.cross(m,precision)/(1+alpha**2)-(gamma*h_cross*Is/(2*e*Ms*V))*np.cross(m,np.cross(m,sigma))/(1+alpha**2)+alpha*gamma*h_cross*Is*np.cross(m,sigma)/((2*e*Ms*V)*(1+alpha**2))

    return dmdt

def mag(M):
    magnitude = np.sqrt((M[0])**2+(M[1])**2+(M[2])**2)
    return magnitude

t_fine_ns=np.linspace(0,t_coarse_ns[-1],50*len(t_coarse_ns))
t_fine=t_fine_ns*1e-9
n=len(t_fine)

m=np.zeros((n,3))
mx0=0.000
mz0=0
my0=np.sqrt(1-mx0**2)
m[0,:]=[mx0,my0,mz0]

ti=0

m_mag=np.zeros(n)
m_mag[ti]=mag(m[ti,:])

H=np.array([0,0,Hz])

h_step=t_fine[2]-t_fine[1]
t_fine_ns_array=[]
t_fine_ns_array=np.append(t_fine_ns_array,t_fine_ns[0])
mx_array=[]
my_array=[]
mz_array=[]
mx_array=np.append(mx_array, m[0,0])
my_array=np.append(my_array, m[0,1])
mz_array=np.append(mz_array, m[0,2])

#fig=plt.figure(figsize=(16,12))



# Heun
while ti<(n-1):
    k1=LLGS(m[ti,:],H,ti)
    m_k1=(m[ti,:]+h_step*k1)
    k2=LLGS(m_k1,H,ti)
    m_k2= (m[ti,:]+h_step*k2)
    #k3=LLGS(m_k2,H)
    #m_k3=(m[ti,:]+h_step*k3)
    #k4=LLGS(m_k3,H)

    m[ti+1,:]=m[ti,:]+(h_step/2.0)*(k1+k2)
    #m_mag[ti+1]=mag(m[ti+1,:])
    ti=ti+1
    t_fine_ns_array=np.append(t_fine_ns_array,t_fine_ns[ti])
    mx_array=np.append(mx_array, m[ti,0])
    my_array=np.append(my_array, m[ti,1])
    mz_array=np.append(mz_array, m[ti,2])
ti=0
while ti<n-1:
    ti+=1
    if mz_array[ti]>0:
        mz_array[ti]=1
    elif mz_array[ti]<0:
        mz_array[ti]=-1
