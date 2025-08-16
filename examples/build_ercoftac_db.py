from post.OperatorsSpectral import Cheb2D
import numpy as np
from numpy import pi
import tecio_wrapper

xmin , xmax = [-1,-2] , [+2,+3]

n , n_dst = [32,32] , [24,18]

ops = Cheb2D(xmin=xmin, xmax=xmax, n=n)
X , Y = ops.grid()                     
F = np.sin(np.pi*X) * np.cos(np.pi*Y)

# Interpolation sur une grille cible
x_dst = np.linspace(xmin[0],xmax[0],n_dst[0])
y_dst = np.linspace(xmin[1],xmax[1],n_dst[1])
X_dst , Y_dst = np.meshgrid( x_dst , y_dst , indexing='ij' )

F_dst = ops.interpolate(F, x_dst, y_dst)
   
tecio_wrapper.write_szplt("spectral_o.plt", ["x", "y","fi","der"], [X,Y,ops.dx(F), pi * np.cos(pi*X) * np.cos(pi*Y)])
tecio_wrapper.write_szplt("spectral_f.plt", ["x", "y","fi"], [X,Y,np.cos(np.pi*X) * np.cos(np.pi*Y) * pi])

tecio_wrapper.write_szplt("spectral_i.plt", ["x", "y","fi"], [X_dst,Y_dst,F_dst])
 
 
res = np.abs( ops.dx(F) - pi * np.cos(pi*X) * np.cos(pi*Y) )
der , bb = ops.dx(F) ,  pi * np.cos(pi*X) * np.cos(pi*Y)
for i in np.arange(n[0]):
    print(der[i,3],bb[i,3],res[i,3])
 
 
exit()


x , y  = ops._x1_phys_ , ops._x2_phys_
X,Y=np.meshgrid(x,y) 
fi = np.cos(pi*X)*np.cos(pi*Y)
tecio_wrapper.write_szplt("spectral.plt", ["x", "y","fi"], [X,Y,fi])
# Cibles physiques quelconques
x_t = np.linspace(-1, 1, 32)
y_t = np.linspace(-1, 1, 64)
X_t,Y_t=np.meshgrid(x_t,y_t) 

FI_interp = ops.interpolate_cheb_x1x2(fi, x_t, y_t)  
print(np.shape(FI_interp))
#tecio_wrapper.write_szplt("interpolate.plt", ["x", "y","fi"], [X_t,Y_t,FI_interp])

exit()
from pathlib import Path

def build_db_tree(db_name):
    # build the database tree of directories similar to Sebilleau
    base = Path(db_name)
    dirs = [ base / "Basic_stat",base / "Budgets",base / "Data_midwidth",base / "Nu",base / "WSS",]
    # Budget subfolders: X_0p2 .. X_0p8
    x_list = ["0p2", "0p3", "0p4", "0p5", "0p6", "0p7", "0p8"]
    dirs += [base / "Budgets" / f"X_{x}" for x in x_list] 
    # Création
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        print(f"créé: {d}")

db_name='db_1e12'
build_db_tree(db_name)

# post-processing of the basic stats
## the variables to provide for comparisons

varnames_sebilleau=["y","y+","U","V","uu","vv","ww","uv","T","TT","uT","vT"]
mapping = {"x": "y", "y": "z","z": "x","u": "v","v": "w","w": "u"}
def permute_name(name):
    out = ""
    for ch in name:
        out += mapping.get(ch.lower(), ch)
    return out
varnames = [permute_name(v) for v in varnames_sebilleau]
print(varnames)

import statistics_loader as sl 

db=sl.H5DB("stats_hii.h5")
y, z = db.get_many("y", "z")

v, w, T = db.get_many("v", "w", "T")
vv, ww, uu = db.get_many("v.v", "w.w", "u.u")
TT, vT, wT = db.get_many("T.T", "v.T", "w.T")

#ux,uy,uz= db.get_many("ux", "uy", "uz")
#vx,vy,vz= db.get_many("vx", "vy", "vz")
#wx,wy,wz= db.get_many("wx", "wy", "wz")
#ux_ux , T_T= db.get_many("ux.ux","T.T")

db.summary()
import tecio_wrapper
tecio_wrapper.write_szplt("toto.plt", ["y", "z","T.T","v.v"], [y,z,TT,vv])

# création d'un interpolateur 


