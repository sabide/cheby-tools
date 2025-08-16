import discr as dis  

xmin, xmax = [-1.0, -2.0], [2.0, 3.0]
n = [48, 64]
ops = dis.discr_2d(xmin=xmin, xmax=xmax, n=n)
dis._test_module()
