import numpy as np
from random import gauss
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d 


plt.rcParams.update({'font.size':24})
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams.update({'font.weight':'bold'})
plt.rcParams["font.family"] = "Times New Roman"
rc = {"font.family" : "serif", 
      "mathtext.fontset" : "stix"}
plt.rcParams.update(rc)
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
#################### Real unit  LLGS ###########################
def deterministic_LLGS(m,h):

	precision=-np.cross(m,h)
	damping=-(alpha)*np.cross(m,np.cross(m,h))
	
	dmdt=precision+damping

	return dmdt
#################################################################
def stochastic_LLGS(m,h_T):
	s_precision=-np.cross(m,h_T)
	s_damping=-(alpha)*np.cross(m,np.cross(m,h_T))
	
	s_dmdt=s_precision+s_damping

	return s_dmdt
'''
##################### Vector magnitude #############################
def mag(M):
	magnitude = np.sqrt((M[0])**2+(M[1])**2+(M[2])**2)
	return magnitude
#################################################################

################### LLGS calculation result #########################
def calculate_LLGS(m,Hk,Hd,H_ME_coeff):
	global n, alpha, K_B, gamma, Ms, V, delta_t, mu0, T
	h_step_tau = delta_t*dim_less_time_fac
	tau_i=0
	m_tilda=np.zeros((n,3))
	H_uni=np.zeros(3)
	H_demag=np.zeros(3)
	
	while tau_i<(n-1):
		#print(ti)
		mx=m[tau_i,0]
		my=m[tau_i,1]
		mz=m[tau_i,2]
		############## Uniaxial Field ######################
		H_uni=np.array([0, 0, Hk*mz])
		############## Demag Field #######################
		H_demag=np.array([Hd[0]*mx,Hd[1]*my,Hd[2]*mz])
		############## ME Field ##########################
		H_ME=np.array([0,0,H_ME_coeff[2]*mz])
		############### Thermal section ###################
		G01=np.zeros(3)
		G01[0]=gauss(0,1)
		G01[1]=gauss(0,1)
		G01[2]=gauss(0,1)
		alpha_const=alpha/(1+alpha**2)
	
		T_K=1.0*T     # in Kelvin
		const=(2*alpha*K_B*T_K)/(gamma*Ms*V*delta_t)  
		Therm_const=np.sqrt(const)
		H_therm=0.5*Therm_const*G01

		H_eff=H_uni+H_demag+H_ME+H_therm
		h_eff=H_eff/Hk

		m_tilda[tau_i+1,:]=m[tau_i,:]+h_step_tau*LLGS(m[tau_i,:],h_eff)
		m[tau_i+1,:]=m[tau_i,:]+(h_step_tau/2.0)*(LLGS(m[tau_i,:],h_eff)+LLGS(m_tilda[tau_i+1,:],h_eff))

		m[tau_i+1,:]=m[tau_i+1,:]/mag(m[tau_i+1,:])

		tau_i=tau_i+1

	return m
	
'''

#################### constant parameters ############################
gamma=1.76e11;           # Gyromagnetic ratio [(rad)/(s.T)]
mu0=4*np.pi*1e-7 ;      # in T.m/A

q=1.6e-19;               # in Coulomb
hbar=1.054e-34;          # Reduced Planck's constant (J-s)
K_B=1.38064852e-23    #in J/K
T=300                             # in Kelvin
#################### parameters related to nanomagnet ################
alpha=0.15              # Gilbert damping parameter
Ms=250*1e3            # in A/m


Length=16e-9
Width=8e-9
t_FL=0.9e-9

A_MTJ=(np.pi/4.0)*Length*Width 
Area=A_MTJ
V=A_MTJ*t_FL;             # Volume [m^3]

Ki=0.068*1e-3

Ku2=Ki/t_FL              # in J/m^3
################# Anisotropy field ###################################
Hk=2*Ku2/(mu0*Ms)             # Uniaxial field [in A/m]
hk=Hk/Ms                                # Normalized uniaxial field

print('Uniaxial = ' + str(hk))
################## Demagnetization field related #######################

Nxx=0
Nyy=0
Nzz=1.0

hdx=-Nxx*Ms	# in A/m
hdy=-Nyy*Ms	# in A/m
hdz=-Nzz*Ms		# in A/m
Hd=np.array([hdx, hdy, hdz])
hd=Hd/Ms              #normalized demag field

print('Demag = ' + str(hd))
################## Magneto-electric field related #######################
alpha_ME=0.03*1e-7
tox_ME=5e-9
V_IN=0.0
H_ME_coeff=alpha_ME*(V_IN/tox_ME)  # in Tesla
H_ME=H_ME_coeff/mu0         # in A/m
h_ME=H_ME/Ms            # Dimensionless Magneto-electric field

print('ME = ' + str(h_ME))
###################### time related portion ###########################
dim_less_time_fac = gamma*mu0*Ms

print('dimless time factor = ' + str(dim_less_time_fac))

stop_time=5.0e-9             # in s
stop_tau=stop_time*dim_less_time_fac

print('Stop Tau = ' + str(stop_tau))

delta_t=1*1e-12
delta_tau = delta_t*dim_less_time_fac
print('Delta tau = ' + str(delta_tau))

n=int(stop_tau/delta_tau)
print('n = ' + str(n))
t=np.linspace(0,stop_time,n) 
tau=np.linspace(0,stop_tau,n) # in ns
delta_tau=tau[1]-tau[0]

##################### Initial Magnetization ##########################
m=np.zeros((n,3))            
print('n = ' + str(n))
mz0=0.5
mx0=np.sqrt(1-mz0**2)
my0=0

m[0,:]=[mx0,my0,mz0]
print('n = ' + str(n))

loop=n-1
for i in range(loop):
	print('---------------------------------------------------------')
	print('i = ' + str(i))
	h_uni_i=hk*np.array([0,0,m[i,2]])
	h_demag_i=np.array([hd[0]*m[i,0], hd[1]*m[i,1], hd[2]*m[i,2]])
	h_ME_i=np.array([0,0,h_ME*m[i,2]])
	
	h_eff_det_i = h_uni_i + h_demag_i + h_ME_i
	
	Deterministic_Euler_update = deterministic_LLGS(m[i,:],h_eff_det_i)
	
	T_K=300     # in Kelvin
	const=(2*alpha*K_B*T_K)/(gamma*Ms*V*delta_t)  
	H_T=(1/mu0)*np.sqrt(const)
	h_T=(H_T/Ms)
	print('Thermal = ' + str(h_T))
	G01=np.zeros(3)
	G01[0]=gauss(0,1)
	G01[1]=gauss(0,1)
	G01[2]=gauss(0,1)
	h_T=h_T*G01
	eta_n=np.sqrt(delta_tau)
	
	
	m_i_var = m[i,:]+(stochastic_LLGS(m[i,:],h_T)*eta_n)
	
	Stochastic_Huen_update = 0.5*(stochastic_LLGS(m_i_var,h_T) + stochastic_LLGS(m[i,:],h_T))*eta_n
	print('Stochastic_Huen_update = ' + str(Stochastic_Huen_update))
	
	m[i+1,:]=m[i,:]+Deterministic_Euler_update*delta_tau+Stochastic_Huen_update

	

t=t*1e9 # time converted to ns scale

fig = plt.figure(figsize=(14,7))
lw=2.2
#plt.subplot(1,2,1)
plt.plot(t[0:loop],m[0:loop,0], 'b', linewidth=lw, label='mx')
plt.plot(t[0:loop],m[0:loop,1], 'k', linewidth=lw, label='my')
plt.plot(t[0:loop],m[0:loop,2], 'r', linewidth=lw, label='mz')
#plt.plot(t,m[:,1], 'k', linewidth=lw, label='my')
#plt.plot(t,m[:,2], 'r', linewidth=lw, label='mz')
plt.grid()
plt.legend()
plt.xlabel('Time(ns)')
plt.ylabel(r"$m$")
#plt.ylim([-1.2,1.2])
plt.savefig('Stochastic_Euler_Huen_pbit_result.pdf', bbox_inches='tight', pad_inches=0.2)
'''
plt.subplot(1,2,2)
plt.plot(t,M_2[:,2], linewidth=lw)
plt.grid()
plt.xlabel('Time(ns)')
#plt.ylabel(r"$m_z$")
plt.title(r"$MTJ_H$")
plt.ylim([-1.2,1.2])
'''
plt.show()
