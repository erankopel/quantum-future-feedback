
import numpy as np
D=np.load('/mnt/data/nonEB_result.npz');A=D['A'];rho=D['rho'];met=D['metrics'];v=np.array([2*rho[0,1].real,-2*rho[0,1].imag,(rho[0,0]-rho[1,1]).real])
nA=np.linalg.norm(A,2);R=np.linalg.norm(np.linalg.inv(np.eye(3)-A),2);C=met[1];I0=met[2];mu=met[3]
def h(x): return -x*np.log2(x)-(1-x)*np.log2(1-x) if 0<x<1 else 0
for dpi in [.003,.0035,.0036]:
 d=dpi*np.pi;s=2*np.sin(d/2); da=12*s;dc=4*np.sqrt(3)*s
 stab=1-(nA+da)
 dv=R*(dc+da*np.linalg.norm(v))/(1-R*da)
 coh=C-dv
 eps=2*s
 info=I0-(3*h(eps)+eps*np.log2(3))
 npt=mu-4*s
 print(dpi,s,'stab_margin',stab,'dv',dv,'cohLB',coh,'infoLB',info,'npt_margin',npt)

# ------------------------------------------------------------------------------