
import sys, numpy as np
sys.path.insert(0,'/mnt/data')
from coherent_model_compute import channel_affine, fixed, bloch
for kb in [(0.3,0.4),(0.7,0.2),(1.0,0.8)]:
 k,b=kb; par=(0.0,k,np.pi/2,0.0,0.0,0.0,b)
 A,c=channel_affine(par); rho,_,_,ev=fixed(par)
 Aa=np.array([[0,0,-np.cos(b)*np.sin(k)],[0,0,0],[0,0,np.sin(b)*np.sin(k)]])
 ca=np.array([np.sin(b)*np.cos(k),0,np.cos(b)*np.cos(k)])
 den=1-np.sin(b)*np.sin(k)
 va=np.array([np.cos(k)*(np.sin(b)-np.sin(k))/den,0,np.cos(b)*np.cos(k)/den])
 print(k,b,'Aerr',np.max(abs(A-Aa)),'cerr',np.max(abs(c-ca)),'verr',np.max(abs(bloch(rho)-va)), 'ev',ev,'formula',np.sin(b)*np.sin(k))

# ------------------------------------------------------------------------------