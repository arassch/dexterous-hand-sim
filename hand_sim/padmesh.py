"""Volumetric tet mesh for an ellipsoidal fingertip pad (cube grid mapped to a sphere, 6 Kuhn tets per hex)."""
import numpy as np, itertools
def ellipsoid_tets(R, n=4):
    g=np.linspace(-1,1,n); idx={}; P=[]
    for i,j,k in itertools.product(range(n),repeat=3):
        x,y,z=g[i],g[j],g[k]
        # cube -> sphere mapping (keeps grid connectivity, avoids degenerate tets)
        s=np.array([x*np.sqrt(1-y*y/2-z*z/2+y*y*z*z/3),
                    y*np.sqrt(1-z*z/2-x*x/2+z*z*x*x/3),
                    z*np.sqrt(1-x*x/2-y*y/2+x*x*y*y/3)])
        idx[i,j,k]=len(P); P.append(s*R)
    P=np.array(P); T=[]
    kuhn=[(0,1,3,7),(0,1,5,7),(0,2,3,7),(0,2,6,7),(0,4,5,7),(0,4,6,7)]
    for i,j,k in itertools.product(range(n-1),repeat=3):
        c=[idx[i+a,j+b,k+cc] for a,b,cc in [(0,0,0),(1,0,0),(0,1,0),(1,1,0),(0,0,1),(1,0,1),(0,1,1),(1,1,1)]]
        for t in kuhn:
            v=[c[q] for q in t]
            if np.linalg.det(np.array([P[v[1]]-P[v[0]],P[v[2]]-P[v[0]],P[v[3]]-P[v[0]]]))<0: v[2],v[3]=v[3],v[2]
            T.append(v)
    return P,np.array(T)
