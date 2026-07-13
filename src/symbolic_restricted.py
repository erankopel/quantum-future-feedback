
import sympy as s
I=s.I
k,p,b=s.symbols('kappa phi beta', real=True)
Id=s.eye(2); X=s.Matrix([[0,1],[1,0]]); Y=s.Matrix([[0,-I],[I,0]]); Z=s.diag(1,-1)
P0=s.diag(1,0); P1=s.diag(0,1)
def kron(*args): return s.kronecker_product(*args)
def Ry(a): return s.cos(a/2)*Id-I*s.sin(a/2)*Y
# theta=0: m=0 flips F, m=1 leaves F; anti-copy F=1-m
UW=kron(P0,X,Id)+kron(P1,Id,Id)
# weak Z_F - Y_L coupling
G=kron(Id,Z,Y)
Uw=s.cos(k/2)*s.eye(8)-I*s.sin(k/2)*G
# swap M,L
SW=s.zeros(8)
for m in range(2):
 for f in range(2):
  for l in range(2): SW[(l*2+f)*2+m,(m*2+f)*2+l]=1
Ufb=s.cos(p)*s.eye(8)-I*s.sin(p)*SW
RM=kron(Ry(b),Id,Id)
U=s.simplify(RM*Ufb*Uw*UW)
anc=s.zeros(4); anc[0,0]=1
paulis=[X,Y,Z]

def ptr_FL(M):
 out=s.zeros(2)
 for m in range(2):
  for n in range(2):
   out[m,n]=sum(M[(m*2+f)*2+l,(n*2+f)*2+l] for f in range(2) for l in range(2))
 return s.simplify(out)
def apply(rho): return ptr_FL(s.simplify(U*kron(rho,anc)*U.conjugate().T))
def bloch(rho): return s.Matrix([s.simplify(s.trace(rho*P)) for P in paulis])
r0=apply(Id/2); c=s.simplify(bloch(r0)); A=s.zeros(3)
for j,P in enumerate(paulis): A[:,j]=s.simplify((bloch(apply((Id+P)/2))-bloch(apply((Id-P)/2)))/2)
print('A='); s.print_latex(s.trigsimp(A)); print(s.simplify(A))
print('c='); print(s.simplify(c))
# solve fixed point symbolically
v=s.simplify((s.eye(3)-A).inv()*c)
print('v='); print([s.trigsimp(x) for x in v]); s.print_latex(s.Matrix([s.trigsimp(x) for x in v]))
print('charpoly factor'); lam=s.symbols('lambda'); print(s.factor(A.charpoly(lam).as_expr()))
open('/mnt/data/symbolic_output.txt','w').write('A='+str(A)+'\nc='+str(c)+'\nv='+str(v)+'\nchar='+str(s.factor(A.charpoly(lam).as_expr())))

# ------------------------------------------------------------------------------