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

import stats as sl 

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
import tecio 
tecio.write_szplt("toto.plt", ["y", "z","T.T","v.v"], [y,z,TT,vv])

import numpy as np
import discr as dis
xmin, xmax = [0.0, 0.0], [1.0, 1.0]
n=np.shape(y)
#n = [48, 64]
ops = dis.discr_2d(xmin=xmin, xmax=xmax, n=n)

tmp = ops.interpolate_line_x(TT, 0.2)
print(tmp)

tecio.write_ndarray_1d("demo_1d.plt", ["y", "TT"], [y[:,0],tmp])

# création d'un interpolateur 


