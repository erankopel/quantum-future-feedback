
import numpy as np
from scipy.linalg import expm
I=np.eye(2, dtype=complex); X=np.array([[0,1],[1,0]],complex); Y=np.array([[0,-1j],[1j,0]],complex); Z=np.diag([1,-1]); P0=np.diag([1,0]); P1=np.diag([0,1]); ps=[X,Y,Z]
def ry(a): return expm(-1j*a*Y/2)
def ptr(M):
 a=M.reshape(2,2,2,2,2,2); return np.trace(np.trace(a,axis1=2,axis2=5),axis1=1,axis2=3)
SW=np.zeros((8,8),complex)
for m in range(2):
 for f in range(2):
  for l in range(2): SW[(l*2+f)*2+m,(m*2+f)*2+l]=1
anc=np.zeros((4,4),complex); anc[0,0]=1
for k,b in [(.3,.4),(.7,.2),(1.0,.8)]:
 UW=np.kron(P0,np.kron(X,I))+np.kron(P1,np.kron(I,I))
 Uw=np.cos(k/2)*np.eye(8)-1j*np.sin(k/2)*np.kron(I,np.kron(Z,Y))
 Uf=-1j*SW; R=np.kron(ry(b),np.kron(I,I)); U=R@Uf@Uw@UW
 def app(r): return ptr(U@np.kron(r,anc)@U.conj().T)
 def bl(r): return np.array([np.trace(r@q).real for q in ps])
 c=bl(app(I/2)); A=np.zeros((3,3))
 for j,q in enumerate(ps): A[:,j]=(bl(app((I+q)/2))-bl(app((I-q)/2)))/2
 Aa=np.array([[0,0,-np.cos(b)*np.sin(k)],[0,0,0],[0,0,np.sin(b)*np.sin(k)]])
 ca=np.array([np.sin(b)*np.cos(k),0,np.cos(b)*np.cos(k)])
 den=1-np.sin(b)*np.sin(k); va=np.array([np.cos(k)*(np.sin(b)-np.sin(k))/den,0,np.cos(b)*np.cos(k)/den])
 vn=np.linalg.solve(np.eye(3)-A,c)
 print(k,b,np.max(abs(A-Aa)),np.max(abs(c-ca)),np.max(abs(vn-va)),np.linalg.eigvals(A))

# ------------------------------------------------------------------------------