
import numpy as np
from scipy.optimize import brentq
Aup=.71333; Rup=2.3874; vup=.5852; Clo=.5710; Ilo=1.5685; mulo=.2051
def h(x): return -x*np.log2(x)-(1-x)*np.log2(1-x) if 0<x<1 else 0
s_npt=mulo/4;s_stab=(1-Aup)/12
f=lambda s:Rup*(4*np.sqrt(3)*s+12*s*vup)/(1-Rup*12*s)-Clo
s_coh=brentq(f,1e-12,.03)
fi=lambda s:3*h(2*s)+2*s*np.log2(3)-Ilo
s_info=brentq(fi,1e-12,.2)
print(s_npt,s_stab,s_coh,s_info)
for s in [s_npt,s_stab,s_coh,s_info]:print(2*np.arcsin(s/2)/np.pi)
dpi=.0034;d=dpi*np.pi;s=2*np.sin(d/2);da=12*s;dc=4*np.sqrt(3)*s
print('chosen',dpi,'s',s,'stab',1-(Aup+da),'coh',Clo-Rup*(dc+da*vup)/(1-Rup*da),'info',Ilo-(3*h(2*s)+2*s*np.log2(3)),'npt',mulo-4*s)

# ------------------------------------------------------------------------------