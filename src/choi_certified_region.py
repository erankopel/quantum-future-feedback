
import numpy as np, matplotlib.pyplot as plt
from scipy.linalg import expm
I=np.eye(2,dtype=complex); X=np.array([[0,1],[1,0]],complex); Y=np.array([[0,-1j],[1j,0]],complex); Z=np.diag([1,-1]); P0=np.diag([1,0]); P1=np.diag([0,1])
def ry(a): return expm(-1j*a*Y/2)
anc=np.zeros((4,4),complex);anc[0,0]=1
SW=np.zeros((8,8),complex)
for m in range(2):
 for f in range(2):
  for l in range(2):SW[(l*2+f)*2+m,(m*2+f)*2+l]=1
def ptr(r):
 a=r.reshape(2,2,2,2,2,2);return np.trace(np.trace(a,axis1=2,axis2=5),axis1=1,axis2=3)
k0=.4323098*np.pi;b0=.23903823*np.pi;th0=.16345853*np.pi;ph0=.20061939*np.pi
def U(th,ph):
 UW=np.kron(P0,np.kron(ry(np.pi-2*th),I))+np.kron(P1,np.kron(ry(2*th),I))
 Uw=expm(-1j*k0/2*np.kron(I,np.kron(Z,Y)))
 Uf=np.cos(ph)*np.eye(8)-1j*np.sin(ph)*SW
 return np.kron(ry(b0),np.kron(I,I))@Uf@Uw@UW
def lmin(th,ph):
 u=U(th,ph);J=np.zeros((4,4),complex)
 for i in range(2):
  for j in range(2):
   E=np.zeros((2,2),complex);E[i,j]=1;J+=np.kron(E,ptr(u@np.kron(E,anc)@u.conj().T))/2
 H=J.reshape(2,2,2,2).transpose(2,1,0,3).reshape(4,4)
 return np.linalg.eigvalsh(H)[0]
mu=-lmin(th0,ph0)
xs=np.linspace(th0-.07*np.pi,th0+.07*np.pi,121);ys=np.linspace(ph0-.09*np.pi,ph0+.09*np.pi,121)
D=np.array([[lmin(x,y) for x in xs] for y in ys])
fig,ax=plt.subplots(figsize=(8.5,6.2));im=ax.imshow(D,origin='lower',extent=[xs[0]/np.pi,xs[-1]/np.pi,ys[0]/np.pi,ys[-1]/np.pi],aspect='auto',cmap='coolwarm',vmin=-.35,vmax=.12);fig.colorbar(im,ax=ax,label='minimum eigenvalue of Choi partial transpose')
ax.contour(xs/np.pi,ys/np.pi,D,levels=[0],colors='black',linewidths=1.5)
# certified exact-sine boundary sample: 4(sin(|dt|/2)+sin(|dp|/2))=mu
pts=[]
for dt in np.linspace(-2*np.arcsin(mu/4),2*np.arcsin(mu/4),400):
 rem=mu/4-np.sin(abs(dt)/2)
 if rem>=0:
  dp=2*np.arcsin(min(1,rem));pts.append((th0+dt,ph0+dp))
for dt in np.linspace(2*np.arcsin(mu/4),-2*np.arcsin(mu/4),400):
 rem=mu/4-np.sin(abs(dt)/2)
 if rem>=0:
  dp=2*np.arcsin(min(1,rem));pts.append((th0+dt,ph0-dp))
pts=np.array(pts);ax.plot(pts[:,0]/np.pi,pts[:,1]/np.pi,color='#FFD43B',lw=3,label='analytic certified region')
ax.plot([th0/np.pi],[ph0/np.pi],'wo',mec='black',ms=7,label='reference point')
ax.set_xlabel('theta / pi');ax.set_ylabel('phi / pi');ax.set_title('Certified non-entanglement-breaking slice',fontweight='bold',color='#173F5F');ax.legend();fig.tight_layout();fig.savefig('/mnt/data/choi_certified_region.png',dpi=220,bbox_inches='tight',facecolor='white')
print(mu,2*np.arcsin(mu/8),2*np.arcsin(mu/8)/np.pi)

# ------------------------------------------------------------------------------