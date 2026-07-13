
import numpy as np
from scipy.optimize import brentq
D=np.load('/mnt/data/nonEB_result.npz');A=D['A'];rho=D['rho'];met=D['metrics'];v=np.array([2*rho[0,1].real,-2*rho[0,1].imag,(rho[0,0]-rho[1,1]).real])
nA=np.linalg.norm(A,2);R=np.linalg.norm(np.linalg.inv(np.eye(3)-A),2);C=met[1];Info=met[2];mu=met[3]
def h(x):
 if x<=0 or x>=1:return 0
 return -x*np.log2(x)-(1-x)*np.log2(1-x)
# s=sin(dt/2)+sin(dp/2), square s=2sin(delta/2)
s_npt=mu/4
s_stab=(1-nA)/12
# coherence change bound
f=lambda s: R*(4*np.sqrt(3)*s+12*s*np.linalg.norm(v))/(1-R*12*s)-C
s_coh=brentq(f,1e-12,min(0.99/(12*R),.05))
# info epsilon=2s; continuity bound
fi=lambda s: 3*h(2*s)+2*s*np.log2(3)-Info
s_info=brentq(fi,1e-12,.249)
print('nA,R,vnorm,C,I,mu',nA,R,np.linalg.norm(v),C,Info,mu)
print('s limits',s_npt,s_stab,s_coh,s_info)
for name,s in [('npt',s_npt),('stab',s_stab),('coh',s_coh),('info',s_info),('combined',min(s_npt,s_stab,s_coh,s_info))]:
 delta=2*np.arcsin(s/2); print(name,'delta',delta,'delta/pi',delta/np.pi)

# ------------------------------------------------------------------------------