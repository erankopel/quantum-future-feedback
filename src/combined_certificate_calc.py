
import numpy as np
D=np.load('/mnt/data/nonEB_result.npz') if __import__('os').path.exists('/mnt/data/nonEB_result.npz') else None
if D:
 A=D['A']; c=D['c']; rho=D['rho']; ev=D['ev']; met=D['metrics']
 print('A',A,'sv',np.linalg.svd(A,compute_uv=False),'norm',np.linalg.norm(A,2),'gapnorm',1-np.linalg.norm(A,2))
 print('v',np.array([2*rho[0,1].real,-2*rho[0,1].imag,(rho[0,0]-rho[1,1]).real]))
 print('coh',met[1], 'info', met[2], 'neg',met[3])
print('invnorm',np.linalg.norm(np.linalg.inv(np.eye(3)-A),2))

# ------------------------------------------------------------------------------