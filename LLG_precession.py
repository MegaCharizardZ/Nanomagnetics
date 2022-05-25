import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform
from mpl_toolkits.mplot3d.axes3d import Axes3D

plt.rcParams.update({'font.size':24})
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams.update({'font.weight':'bold'})
plt.rcParams["font.family"] = "Times New Roman"
rc = {"font.family" : "serif", 
      "mathtext.fontset" : "stix"}
plt.rcParams.update(rc)
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]

class Arrow3D(FancyArrowPatch):

    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)

    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)
  
def _arrow3D(ax, x, y, z, dx, dy, dz, *args, **kwargs):
    '''Add an 3d arrow to an `Axes3D` instance.'''

    arrow = Arrow3D(x, y, z, dx, dy, dz, *args, **kwargs)
    ax.add_artist(arrow)


setattr(Axes3D, 'arrow3D', _arrow3D)

##################################################################
# Anlytical solution
##################################################################
mu_0=4*np.pi*1e-7    # in N/A**2
gamma=1.76*1e11      # in 1/(s.T)
Hz=10*1e3                  # in A/m

w=mu_0*gamma*Hz  # in 1/s

t_coarse_ns=np.linspace(0,20,101) # in ns
t_coarse=t_coarse_ns*1e-9                 # in s

mx=0.5*np.cos(w*t_coarse)
my=0.5*np.sin(w*t_coarse)
mz=(np.sqrt(3)/2.0)*np.ones(len(t_coarse))

m_vec_mag=np.zeros(len(t_coarse))

for i in range(len(t_coarse)):
	m_vec_mag[i]=np.sqrt((mx[i])**2+(my[i])**2+(mz[i])**2)
	
##################################################################
# Numerical solution
##################################################################
def LLGS(m,H):
	global mu_0,gamma

	precision=-(gamma*mu_0)*np.cross(m,H)    
	dmdt=precision

	return dmdt
	
def mag(M):
	magnitude = np.sqrt((M[0])**2+(M[1])**2+(M[2])**2)
	return magnitude
	
t_fine_ns=np.linspace(0,t_coarse_ns[-1],50*len(t_coarse_ns))
t_fine=t_fine_ns*1e-9
n=len(t_fine)

m=np.zeros((n,3))
mz0=(np.sqrt(3)/2.0)
mx0=np.sqrt(1-mz0**2)
my0=0
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

fig=plt.figure(figsize=(16,12))

ax=fig.add_subplot(2,2,1)
ax.plot(t_coarse_ns,mx,'k*',markersize=10.0,label='Analytical')
ax.plot(t_fine_ns_array,mx_array,'r',linewidth=3.0, label='Numerical')
#plt.xlabel('t (ns)')
ax.set_ylabel(r"$m_x$")
ax.set_xticks([0, 0.5*t_coarse_ns[-1], t_coarse_ns[-1]])
ax.set_ylim([-1.1,1.1])
ax.legend(fontsize=20,loc='upper center', ncol=2, frameon=False)
ax.grid()

ax=fig.add_subplot(2,2,2)
ax.plot(t_coarse_ns,my,'k*',markersize=10.0,label='Analytical')
ax.plot(t_fine_ns_array,my_array,'g', linewidth=3.0, label='Numerical')
#plt.xlabel('t (ns)')
ax.set_ylabel(r"$m_y$")
ax.set_xticks([0, 0.5*t_coarse_ns[-1], t_coarse_ns[-1]])
ax.set_ylim([-1.1,1.1])
ax.legend(fontsize=20,loc='upper center', ncol=2, frameon=False)
ax.grid()

ax=fig.add_subplot(2,2,3)
ax.plot(t_coarse_ns,mz,'k*',markersize=10.0,label='Analytical')
ax.plot(t_fine_ns_array,mz_array,'c', linewidth=3.0, label='Numerical')
#plt.xlabel('t (ns)')
ax.set_ylabel(r"$m_z$")
ax.set_xticks([0, 0.5*t_coarse_ns[-1], t_coarse_ns[-1]])
ax.set_ylim([-1.1,1.1])
ax.legend(fontsize=20,loc='lower center', ncol=2, frameon=False)
ax.grid()

ax=fig.add_subplot(2,2,4, projection='3d')
l_len=2.5
ax.plot([-l_len,l_len],[0,0],[0,0],color='black')
ax.plot([0,0],[-l_len,l_len],[0,0],color='black')
ax.plot([0,0],[0,0],[-l_len,l_len],color='black')
ax.arrow3D(0,0,0,
           m[0,0],m[0,1],m[0,2],
           mutation_scale=15,
           ec ='blue',
           fc='blue')
ax.scatter(mx_array[-1],0,0, color='r', s=15)
ax.scatter(0,my_array[-1],0, color='g', s=15)
ax.scatter(0,0,mz_array[-1], color='c', s=15)
ax.view_init(elev=30., azim=60)
plt.axis('off')
ti_str=str(ti)
zfi=ti_str.zfill(5)
plt.savefig('mag_dynamics_'+str(zfi)+'.png')

#exit()


# RK4
while ti<(n-1):
	k1=LLGS(m[ti,:],H)
	m_k1=(m[ti,:]+h_step*k1/2.0)
	k2=LLGS(m_k1,H)
	m_k2= (m[ti,:]+h_step*k2/2.0)
	k3=LLGS(m_k2,H)
	m_k3=(m[ti,:]+h_step*k3)
	k4=LLGS(m_k3,H)

	m[ti+1,:]=m[ti,:]+(h_step/6.0)*(k1+2*k2+2*k3+k4)
	#m_mag[ti+1]=mag(m[ti+1,:])
	ti=ti+1
	t_fine_ns_array=np.append(t_fine_ns_array,t_fine_ns[ti])
	mx_array=np.append(mx_array, m[ti,0])
	my_array=np.append(my_array, m[ti,1])
	mz_array=np.append(mz_array, m[ti,2])
	#print(mx_array)
	
	fig=plt.figure(figsize=(16,12))

	ax=fig.add_subplot(2,2,1)
	ax.plot(t_coarse_ns,mx,'k*',markersize=10.0,label='Analytical')
	ax.plot(t_fine_ns_array,mx_array,'r',linewidth=3.0, label='Numerical')
	#plt.xlabel('t (ns)')
	ax.set_ylabel(r"$m_x$")
	ax.set_xticks([0, 0.5*t_coarse_ns[-1], t_coarse_ns[-1]])
	ax.set_ylim([-1.1,1.1])
	ax.legend(fontsize=20,loc='upper center', ncol=2, frameon=False)
	ax.grid()

	ax=fig.add_subplot(2,2,2)
	ax.plot(t_coarse_ns,my,'k*',markersize=10.0,label='Analytical')
	ax.plot(t_fine_ns_array,my_array,'g',linewidth=3.0, label='Numerical')
	#plt.xlabel('t (ns)')
	ax.set_ylabel(r"$m_y$")
	ax.set_xticks([0, 0.5*t_coarse_ns[-1], t_coarse_ns[-1]])
	ax.set_ylim([-1.1,1.1])
	ax.legend(fontsize=20,loc='upper center', ncol=2, frameon=False)
	ax.grid()

	ax=fig.add_subplot(2,2,3)
	ax.plot(t_coarse_ns,mz,'k*',markersize=10.0,label='Analytical')
	ax.plot(t_fine_ns_array,mz_array,'c',linewidth=3.0, label='Numerical')
	#plt.xlabel('t (ns)')
	ax.set_ylabel(r"$m_z$")
	ax.set_xticks([0, 0.5*t_coarse_ns[-1], t_coarse_ns[-1]])
	ax.set_ylim([-1.1,1.1])
	ax.legend(fontsize=20,loc='lower center', ncol=2, frameon=False)
	ax.grid()

	ax=fig.add_subplot(2,2,4, projection='3d')
	l_len=2.5
	ax.plot([-l_len,l_len],[0,0],[0,0],color='black')
	ax.plot([0,0],[-l_len,l_len],[0,0],color='black')
	ax.plot([0,0],[0,0],[-l_len,l_len],color='black')
	ax.arrow3D(0,0,0,
		   m[ti,0],m[ti,1],m[ti,2],
		   mutation_scale=15,
		   ec ='blue',
		   fc='blue')
	plt.axis('off')
	ax.scatter(mx_array[-1],0,0, color='r', s=15)
	ax.scatter(0,my_array[-1],0, color='g', s=15)
	ax.scatter(0,0,mz_array[-1], color='c', s=15)
	ax.view_init(elev=30., azim=60)
	ti_str=str(ti)
	zfi=ti_str.zfill(5)
	plt.savefig('mag_dynamics_'+str(zfi)+'.png')
	plt.close('all')
	

