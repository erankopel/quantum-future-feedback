
import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt

I=np.eye(2,dtype=complex); X=np.array([[0,1],[1,0]],complex); Y=np.array([[0,-1j],[1j,0]],complex); Z=np.diag([1,-1]); P0=np.diag([1,0]); P1=np.diag([0,1]); PA=[X,Y,Z]
def ry(a): return expm(-1j*a*Y/2)
anc=np.zeros((4,4),complex); anc[0,0]=1
SW=np.zeros((8,8),complex)
for m in range(2):
 for f in range(2):
  for l in range(2): SW[(l*2+f)*2+m,(m*2+f)*2+l]=1

def ptr(r,keep):
 dims=[2,2,2]; n=3; arr=r.reshape(dims+dims)
 for j in sorted([i for i in range(n) if i not in keep],reverse=True): arr=np.trace(arr,axis1=j,axis2=j+arr.ndim//2)
 d=2**len(keep); return arr.reshape(d,d)
def entropy(r):
 e=np.linalg.eigvalsh((r+r.conj().T)/2); e=e[e>1e-12]; return float(-(e*np.log2(e)).sum())
def unitary(th,k,ph,b):
 UW=np.kron(P0,np.kron(ry(np.pi-2*th),I))+np.kron(P1,np.kron(ry(2*th),I))
 Uw=expm(-1j*k/2*np.kron(I,np.kron(Z,Y)))
 Uf=np.cos(ph)*np.eye(8)-1j*np.sin(ph)*SW
 R=np.kron(ry(b),np.kron(I,I))
 return R@Uf@Uw@UW

def app(r,par,joint=False):
 U=unitary(*par); q=U@np.kron(r,anc)@U.conj().T
 return q if joint else ptr(q,[0])
def bl(r): return np.array([np.trace(r@p).real for p in PA])
def affine(par):
 c=bl(app(I/2,par)); A=np.zeros((3,3))
 for j,p in enumerate(PA): A[:,j]=(bl(app((I+p)/2,par))-bl(app((I-p)/2,par)))/2
 return A,c
def fixed(par):
 A,c=affine(par); v=np.linalg.solve(np.eye(3)-A,c); r=(I+sum(v[j]*PA[j] for j in range(3)))/2; return r,A,c,np.linalg.eigvals(A)
def choi(par):
 # normalized Choi: sum |i><j| tensor Phi(|i><j|)/2
 J=np.zeros((4,4),complex)
 for i in range(2):
  for j in range(2):
   E=np.zeros((2,2),complex); E[i,j]=1
   J += np.kron(E,app(E,par))/2
 return J
def pt_first(J): return J.reshape(2,2,2,2).transpose(2,1,0,3).reshape(4,4)
def negativity_choi(par):
 e=np.linalg.eigvalsh((pt_first(choi(par))+pt_first(choi(par)).conj().T)/2)
 return float(-e[e<0].sum()), e
def info(par,rho):
 th,k,ph,b=par
 UW=np.kron(P0,np.kron(ry(np.pi-2*th),I))+np.kron(P1,np.kron(ry(2*th),I))
 Uw=expm(-1j*k/2*np.kron(I,np.kron(Z,Y)))
 q=Uw@UW@np.kron(rho,anc)@UW.conj().T@Uw.conj().T
 fl=ptr(q,[1,2]); f=ptr(q,[1]); l=ptr(q,[2]); return entropy(f)+entropy(l)-entropy(fl)
def metrics(par):
 r,A,c,e=fixed(par); neg,_=negativity_choi(par); return {'rho':r,'v':bl(r),'A':A,'c':c,'ev':e,'gap':1-max(abs(e)),'coh':2*abs(r[0,1]),'pur':np.trace(r@r).real,'neg':neg,'info':info(par,r)}

# no-go check theta=0 across partial swaps
rng=np.random.default_rng(12); maxneg=0
for _ in range(200):
 par=(0,rng.uniform(0,np.pi/2),rng.uniform(0,np.pi/2),rng.uniform(0,np.pi/2))
 maxneg=max(maxneg,negativity_choi(par)[0])
print('theta0 max choi negativity',maxneg)
# search 4-param family
best=[]
for _ in range(1200):
 par=(rng.uniform(.03,.24)*np.pi,rng.uniform(.03,.45)*np.pi,rng.uniform(.03,.47)*np.pi,rng.uniform(.02,.45)*np.pi)
 try: m=metrics(par)
 except np.linalg.LinAlgError: continue
 if m['gap']>.03 and m['coh']>.08 and m['info']>.01 and m['neg']>1e-4:
  score=m['gap']*m['coh']*m['info']*np.sqrt(m['neg']); best.append((score,par,m))
best.sort(key=lambda z:z[0],reverse=True)
for s0,par,m in best[:5]: print('BEST',s0,'par/pi',np.array(par)/np.pi,'gap',m['gap'],'coh',m['coh'],'info',m['info'],'neg',m['neg'],'v',m['v'],'ev',m['ev'])
par0=best[0][1]; m0=best[0][2]
# 2D local maps theta,phi keeping k,b
ths=np.linspace(.01,.25,31)*np.pi; phs=np.linspace(.01,.49,31)*np.pi
Ds=[np.full((31,31),np.nan) for _ in range(4)]
for iy,ph in enumerate(phs):
 for ix,th in enumerate(ths):
  try:
   m=metrics((th,par0[1],ph,par0[3])); vals=[m['gap'],m['coh'],m['info'],m['neg']]
   for D,v in zip(Ds,vals): D[iy,ix]=v
  except: pass
fig,axs=plt.subplots(2,2,figsize=(10,8),constrained_layout=True)
titles=['Stability gap','Fixed-point coherence','Quantum information I(F:L)','Choi negativity (non-EB witness)']; cm=['viridis','plasma','magma','cividis']
for ax,D,t,cmap in zip(axs.flat,Ds,titles,cm):
 im=ax.imshow(D,origin='lower',extent=[.01,.25,.01,.49],aspect='auto',cmap=cmap,vmin=0)
 ax.plot([par0[0]/np.pi],[par0[2]/np.pi],'wo',mec='#173F5F'); ax.set_xlabel('theta / pi'); ax.set_ylabel('phi / pi'); ax.set_title(t,fontweight='bold',color='#173F5F'); fig.colorbar(im,ax=ax,fraction=.046,pad=.04)
fig.suptitle('Four-parameter partial-swap extension (kappa and beta fixed)',fontweight='bold',fontsize=14,color='#173F5F')
fig.savefig('/mnt/data/nonEB_partial_swap_maps.png',dpi=220,bbox_inches='tight',facecolor='white')
np.savez('/mnt/data/nonEB_result.npz',par=par0,A=m0['A'],c=m0['c'],rho=m0['rho'],ev=m0['ev'],metrics=np.array([m0['gap'],m0['coh'],m0['info'],m0['neg'],m0['pur']]))

# ------------------------------------------------------------------------------