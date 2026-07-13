
import numpy as np
from scipy.linalg import expm
I=np.eye(2,dtype=complex); X=np.array([[0,1],[1,0]],complex); Y=np.array([[0,-1j],[1j,0]],complex); Z=np.diag([1,-1]); P0=np.diag([1,0]); P1=np.diag([0,1])
def ry(a): return expm(-1j*a*Y/2)
anc=np.zeros((4,4),complex); anc[0,0]=1
SW=np.zeros((8,8),complex)
for m in range(2):
 for f in range(2):
  for l in range(2): SW[(l*2+f)*2+m,(m*2+f)*2+l]=1
def ptrM(r):
 a=r.reshape(2,2,2,2,2,2)
 return np.trace(np.trace(a,axis1=2,axis2=5),axis1=1,axis2=3)
def U(th,k,ph,b):
 UW=np.kron(P0,np.kron(ry(np.pi-2*th),I))+np.kron(P1,np.kron(ry(2*th),I))
 Uw=expm(-1j*k/2*np.kron(I,np.kron(Z,Y)))
 Uf=np.cos(ph)*np.eye(8)-1j*np.sin(ph)*SW
 R=np.kron(ry(b),np.kron(I,I)); return R@Uf@Uw@UW
def app(E,par):
 u=U(*par); return ptrM(u@np.kron(E,anc)@u.conj().T)
def choi(par):
 J=np.zeros((4,4),complex)
 for i in range(2):
  for j in range(2):
   E=np.zeros((2,2),complex);E[i,j]=1;J+=np.kron(E,app(E,par))/2
 return J
def pt(J): return J.reshape(2,2,2,2).transpose(2,1,0,3).reshape(4,4)
par_pi=np.array([0.16345853,0.4323098,0.20061939,0.23903823]); par=par_pi*np.pi
E=np.linalg.eigvalsh((pt(choi(par))+pt(choi(par)).conj().T)/2)
print('par rad',par);print('PT eigenvalues',E);print('lambda_min',E[0]);print('cert uniform rad',-E[0]/6,'rad /pi',-E[0]/(6*np.pi))
# evaluate actual rectangle for context, not proof
for d_pi in [.002,.005,.01,.02,.04]:
 vals=[]
 for dt in np.linspace(-d_pi*np.pi,d_pi*np.pi,31):
  for dp in np.linspace(-d_pi*np.pi,d_pi*np.pi,31):
   pp=par.copy();pp[0]+=dt;pp[2]+=dp; vals.append(np.linalg.eigvalsh(pt(choi(pp)))[0])
 print(d_pi,max(vals),min(vals))

# ------------------------------------------------------------------------------