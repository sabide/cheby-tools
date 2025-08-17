from pathlib import Path



# create the 
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

# build the database
db_name='db_1e12'
build_db_tree(db_name)

# post-processing of the basic stats
## the variables to provide for comparisons
varnames_sebilleau=["x","y","y+","U","V","uu","vv","ww","uv","T","TT","uT","vT"]
mapping = {"x": "y", "y": "z","z": "x","u": "v","v": "w","w": "u"}
def permute_name(name):
    out = ""
    for ch in name:
        out += mapping.get(ch.lower(), ch)
    return out

varnames = [permute_name(v) for v in varnames_sebilleau]

print("variables to load from the h5 stats")
print(varnames)
####


import stats as sl 

db=sl.H5DB("stats_hii.h5")
x, y = db.get_many("y","z")

U, V, T = db.get_many("v", "w", "T")
uu, vv, ww = db.get_many("v.v", "w.w", "u.u")
TT, vT, wT = db.get_many("T.T", "v.T", "w.T")

#ux,uy,uz= db.get_many("ux", "uy", "uz")
#vx,vy,vz= db.get_many("vx", "vy", "vz")
#wx,wy,wz= db.get_many("wx", "wy", "wz")
#ux_ux , T_T= db.get_many("ux.ux","T.T")

db.summary()

import tecio 
tecio.write_szplt("toto.plt", ["y", "x","V","U","T"], [y,x,V,U,T])


import numpy as np
import discr as dis
xmin, xmax = [0.0, 0.0], [1.0, 1.0]
n=np.shape(y)
ops = dis.discr_2d(xmin=xmin, xmax=xmax, n=n)

# export des profils pour la demande  
filename="/Users/abides/workdir/git/Cavity_DNS_database_ERCOFTAC/db_1e11_lin/Basic_stat/Basic_stat_X_0p5.dat"
df = sl.load_dns_database_sebilleau(filename, ["y", "U", "V", "uu", "vv", "ww", "k" , "T" ,"TT"])
y, U , V , uu,vv,ww,k , T, TT = df['y'] , df['U'] , df['V'] , df['uu'], df['vv'], df['ww'], df['k'] , df['T'] ,df['TT'] 
y, U, V, uu, vv, ww, k, T, TT = sl.wall_profiles( y, U, V, uu, vv, ww, k, T, TT, average=True) 
tecio.write_ndarray_1d("Basic_stat_X_0p5_sebilleau.plt", 
                       ["y", "U", "V" , "uu","vv","ww", "k" , "T" ,"TT"] , 
                       [ y , U , V , uu , vv , ww , k , T , TT ] )

# export of the data from the sebilleau database
filename="/Users/abides/workdir/git/Cavity_DNS_database_ERCOFTAC/db_1e11_lin/Data_midwidth/Data_midwidth.dat"
df = sl.load_dns_database_sebilleau(filename, ["x", "U", "V" , "uu" , "vv" , "k" , "T" ,"TT"])
x , U , V , uu , vv , k , T , TT = df['x'], df['U'], df['V'], df['uu'], df['vv'], df['k'] ,df['T'] ,df['TT'] 
ww = 2*k - uu - vv 
x, U, V, uu, vv, ww, k, T, TT = sl.wall_profiles( x, U, V, uu, vv, ww, k, T, TT, average=True) 
tecio.write_ndarray_1d("Data_midwidth_sebilleau.plt", 
                       ["x", "U", "V" , "uu" , "vv" , "ww" , "k" , "T" ,"TT"] , 
                       [x,U,V,uu,vv,ww,k,T,TT] )

#
varnames = [permute_name(v) for v in ["x", "U", "V" ,"uu" ,"vv" , "ww", "T" ,"TT"]]
varnames_to_load=sl.dotify_vars(varnames,["u", "v" ,"T"])
# to take a look on the variable to load with the statistic loader
x,y,U,V,uu,vv,ww,T,TT=db.get_many('y','z','v', 'w', 'v.v', 'w.w', 'u.u', 'T', 'T.T')
k=0.5*(uu+vv+ww)
xv,yv,y0=x[:,0],y[0,:],0.5
U, V, uu, vv, ww, k, T, TT = [
    ops.interpolate_line_y(a, y0, xv) for a in (U, V, uu, vv, ww, k, T, TT)
]
xv,U,V,uu,vv,ww,k,T,TT=sl.wall_profiles(xv, U, V, uu, vv, ww, k, T, TT, average=True) 

tecio.write_ndarray_1d("Data_midwidth_tcheby.plt", 
                       ["x", "U", "V" , "uu" , "vv" , "ww" , "k" , "T" ,"TT"] , 
                       [xv,U,V,uu,vv,ww,k,T,TT] )
# 

x,y,U,V,uu,vv,ww,T,TT=db.get_many('y','z','v', 'w', 'v.v', 'w.w', 'u.u', 'T', 'T.T')
k=0.5*(uu+vv+ww)
xv,yv,x0=x[:,0],y[0,:],0.5
U, V, uu, vv, ww, k, T, TT = [
    ops.interpolate_line_x(a, x0, yv) for a in (U, V, uu, vv, ww, k, T, TT)
]
yv,U,V,uu,vv,ww,k,T,TT=sl.wall_profiles(yv, U, V, uu, vv, ww, k, T, TT, average=True) 
tecio.write_ndarray_1d("Basic_stat_X_0p5_tcheby.plt", 
                       ["y", "U", "V" , "uu" , "vv" , "ww" , "k" , "T" ,"TT"] , 
                       [yv,U,V,uu,vv,ww,k,T,TT] )

exit()











x,U,V,uu,vv,ww,T,TT=db.get_many('y', 'v', 'w', 'u.u', 'v.v', 'w.w', 'T', 'T.T')
k=0.5*(uu+vv+ww)
##
x0=0.5
x=x[:,0]
U = ops.interpolate_line_y(U,x0 ,x)
V = ops.interpolate_line_y(V,x0 ,x)
uu = ops.interpolate_line_y(uu,x0 ,x)
vv = ops.interpolate_line_y(vv,x0,x)
ww = ops.interpolate_line_y(ww,x0,x)
k = ops.interpolate_line_y(k,x0,x)
T = ops.interpolate_line_y(T,x0,x)
TT = ops.interpolate_line_y(TT,x0,x)

tecio.write_ndarray_1d("tcheby.plt", 
                       ["x", "U", "V" , "k" , "T" ,"TT"] , 
                       [x,U,-V,k,-T,TT] )


y,U,V,uu,vv,ww,T,TT=db.get_many('z', 'v', 'w', 'u.u', 'v.v', 'w.w', 'T', 'T.T')
k=0.5*(uu+vv+ww)



ya,ka=sl.wall_profile(y,k)
ya,TTa=sl.wall_profile(y,TT)
tecio.write_ndarray_1d("sebilleau_X_0p5.plt", 
                       ["y","k","TT"] , 
                       [ya,ka,TTa] )

z,U,V,uu,vv,ww,T,TT=db.get_many('z', 'v', 'w', 'u.u', 'v.v', 'w.w', 'T', 'T.T')
k=0.5*(uu+vv+ww)

y0=0.5
y=z[0,:]

U = ops.interpolate_line_x(U,y0 ,y)
V = ops.interpolate_line_x(V,y0 ,y)
uu = ops.interpolate_line_x(uu,y0 ,y)
vv = ops.interpolate_line_x(vv,y0,y)
ww = ops.interpolate_line_x(ww,y0,y)
k = ops.interpolate_line_x(k,y0,y)
T = ops.interpolate_line_x(T,y0,y)
TT = ops.interpolate_line_x(TT,y0,y)
k=0.5*(uu+vv+ww)

tecio.write_ndarray_1d("tcheby-horizontal_0p5.plt", 
                       ["y", "U", "V" , "k" , "T" ,"TT"] , 
                       [y,-U,V,k,-T,TT] )


ya,ka=sl.wall_profile(y,k)
ya,TTa=sl.wall_profile(y,TT)
tecio.write_ndarray_1d("tcheby-horizontal_0p5.plt", 
                       ["y","k","TT"] , 
                       [ya,ka,TTa] )



print(U)





exit()
#filename="/Users/abides/workdir/git/Cavity_DNS_database_ERCOFTAC/db_1e11_lin/Basic_stat/Basic_stat_X_0p5.dat"

#df = sl.load_dns_database_sebilleau(filename, ["y", "T" , "uu" ])
#x_db , TT_db = df['y'] , df['uu'] 

x_tmp,tmp= sl.wall_profile(x_db, T_db, mode=-1)
tecio.write_ndarray_1d("sebilleau.plt", ["y", "TT"], [x_tmp,tmp])

#tmp = ops.interpolate_line_x(T, 0.5,x_db)
#x_tmp,tmp= sl.wall_profile(x_db, tmp, mode=-1)
#tecio.write_ndarray_1d("david.plt", ["y", "TT"], [x_tmp,tmp])





