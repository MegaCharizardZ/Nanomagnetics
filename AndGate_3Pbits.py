# -*- coding: utf-8 -*-
"""
Created on Wed May 25 22:16:00 2022

@author: sinha
"""

# -*- coding: utf-8 -*-
"""
Created on Fri May 20 23:59:15 2022

@author: sinha
"""

import numpy as np
import matplotlib.pyplot as plt
import math

mu_0=4*np.pi*1e-7
gamma=1.76*1e11      # in 1/(s.T)
Hz=0                  # in A/m
alpha=0.1
#alpha=0.007
h_cross=1.054*1e-34
theta_SHE=0.5
AMTJ=np.pi*0.25*15*15*1e-18
ASHM=15*15*1e-18
t_SHM=3.15*1e-9
lambda_sf=2.1*1e-9
#Ms=58000
Ms=300*1000
#Ic=0.002*1e-9
Ic=0.001
P_SHE=(AMTJ/ASHM)*theta_SHE*(1-(1/np.cosh(t_SHM/lambda_sf)))
sigma=np.array([0,1,0])
Is=P_SHE*Ic
V=AMTJ*0.5*1e-9
k_b=1.38*1e-23
k_u=0.42*k_b*300/V
u=np.array([0,0,1])
D=np.array([0.066,0.911,0.022])
e=1.6*1e-19
delta_t=10e-13
w=mu_0*gamma*Hz  # in 1/s

t_coarse_ns=np.linspace(0,2,501) # in ns
t_coarse=t_coarse_ns*1e-9                 # in s
t_fine_ns=np.linspace(0,t_coarse_ns[-1],50*len(t_coarse_ns))
t_fine_ns_array=[]
t_fine_ns_array=np.append(t_fine_ns_array,t_fine_ns[0])
ti=0
n=len(t_fine_ns)
while ti<n-1:
    ti+=1
    t_fine_ns_array=np.append(t_fine_ns_array,t_fine_ns[0])

t_me=np.linspace(0,2,502)

class pbit:
	def __init__(self,mx_array,my_array,mz_array):
		self.mx_array=mx_array
		self.my_array=my_array
		self.mz_array=mz_array
	def LLGS(self,m,H):
		Hth=np.sqrt(2*alpha*k_b*300/((1+alpha**2)*abs(gamma)*mu_0*Ms*V*delta_t))*np.array([np.random.normal(0,1),np.random.normal(0,1),np.random.normal(0,1)])*1000
		H=-4*np.pi*Ms*m[0]+Hth+(2*k_u/(mu_0*Ms))*np.dot(m,u)*u
		precision=-(gamma*mu_0)*np.cross(m,H)
		dmdt=(precision/(1+alpha**2))+alpha*np.cross(m,precision)/(1+alpha**2)-(gamma*h_cross*Is/(2*e*Ms*V))*np.cross(m,np.cross(m,sigma))/(1+alpha**2)+alpha*gamma*h_cross*Is*np.cross(m,sigma)/((2*e*Ms*V)*(1+alpha**2))

		return dmdt
	def mag(self,t):
		return math.sqrt(self.mx_array[t]**2+self.my_array[t]**2+self.mz_array[t]**2)
	def solve(self):
		t_fine_ns=np.linspace(0,t_coarse_ns[-1],50*len(t_coarse_ns))
		t_fine=t_fine_ns*1e-9
		n=len(t_fine)

		m=np.zeros((n,3))
		my0=1
		mz0=0
		mx0=np.sqrt(1-my0**2)
		m[0,:]=[mx0,my0,mz0]

		ti=0

		#m_mag=np.zeros(n)
		#m_mag[ti]=mag(m[ti,:])

		H=np.array([0,0,Hz])

		h_step=t_fine[2]-t_fine[1]
		t_fine_ns_array=[]
		t_fine_ns_array=np.append(t_fine_ns_array,t_fine_ns[0])
		self.mx_array=[]
		self.my_array=[]
		self.mz_array=[]
		self.mx_array=np.append(self.mx_array, m[0,0])
		self.my_array=np.append(self.my_array, m[0,1])
		self.mz_array=np.append(self.mz_array, m[0,2])

		#fig=plt.figure(figsize=(16,12))



# Heun
		while ti<(n-1):
				k1=self.LLGS(m[ti,:],H)
				m_k1=(m[ti,:]+h_step*k1)
				k2=self.LLGS(m_k1,H)
				m_k2= (m[ti,:]+h_step*k2)

				m[ti+1,:]=m[ti,:]+(h_step/2.0)*(k1+k2)
	#m_mag[ti+1]=mag(m[ti+1,:])
				ti=ti+1
				t_fine_ns_array=np.append(t_fine_ns_array,t_fine_ns[ti])
				self.mx_array=np.append(self.mx_array, m[ti,0])
				self.my_array=np.append(self.my_array, m[ti,1])
				self.mz_array=np.append(self.mz_array, m[ti,2])
				if self.mz_array[ti]>0:
					self.mz_array[ti]=min(10000000*self.mz_array[ti],1)
				elif self.mz_array[ti]<0:
					self.mz_array[ti]=max(10000000*self.mz_array[ti],-1)
	#print(mx_array)
		#plt.plot(t_fine_ns_array,mz_array)
		#plt.grid()
		#plt.show()
		return m

class andgate:
	def __init__(self,J,hT):
		self.J=J
		self.hT=hT
	def E(self,t):
		a=pbit([],[],[])
		a.solve()
		b=pbit([],[],[])
		b.solve()
		c=pbit([],[],[])
		c.solve()
		#k=self.J*a.np.array()
		c1=self.J[0,0]*a.mz_array[t]*a.mz_array[t]+self.J[0,1]*a.mz_array[t]*b.mz_array[t]+self.J[1,0]*a.mz_array[t]*b.mz_array[t]+self.J[1,1]*b.mz_array[t]*b.mz_array[t]+self.J[1,2]*c.mz_array[t]*b.mz_array[t]+self.J[1,2]*c.mz_array[t]*b.mz_array[t]+self.J[2,2]*c.mz_array[t]*c.mz_array[t]
		c2=self.hT[0]*a.mz_array[t]+self.hT[1]*b.mz_array[t]+self.hT[2]*c.mz_array[t]
		return -(P_SHE*Ic/(4*e*alpha*k_b*300*k_u*V))*(c2+c1/2)

Aa=andgate(np.array([[0,-1,2],[-1,0,2],[2,2,0]]),np.array([1,1,-2]))
Aaa=[]
#Aaa=np.append(Aaa,Aa.E(1))
a=pbit([],[],[])
a.solve()
b=pbit([],[],[])
b.solve()
c=pbit([],[],[])
c.solve()
for t in range(501):
	c1=Aa.J[0,0]*a.mz_array[t]*a.mz_array[t]+Aa.J[0,1]*a.mz_array[t]*b.mz_array[t]+Aa.J[1,0]*a.mz_array[t]*b.mz_array[t]+Aa.J[1,1]*b.mz_array[t]*b.mz_array[t]+Aa.J[1,2]*c.mz_array[t]*b.mz_array[t]+Aa.J[2,1]*c.mz_array[t]*b.mz_array[t]+Aa.J[2,2]*c.mz_array[t]*c.mz_array[t]+Aa.J[0,2]*a.mz_array[t]*c.mz_array[t]+Aa.J[2,0]*a.mz_array[t]*c.mz_array[t]
	c2=Aa.hT[0]*a.mz_array[t]+Aa.hT[1]*b.mz_array[t]+Aa.hT[2]*c.mz_array[t]
	Aaa=np.append(Aaa,-(P_SHE*Ic*h_cross*k_b*300/(4*e*alpha*k_b*300*k_u*V))*(c2+c1/2))
mini=10000
pos=0
for t in range(501):
    if Aaa[t]<mini:
        mini=Aaa[t]
        pos=t
        
print(a.mz_array[pos])
print(b.mz_array[pos])
print(c.mz_array[pos])
# for t in range(501):
# 	#print("im gay")
# 	 Aaa=np.append(Aaa,Aa.E(t))
# 	 break
#plt.plot(t_coarse_ns,Aaa)
#plt.show()
#plt.close()
plt.plot(t_coarse_ns,Aaa)