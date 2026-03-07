import numpy as np
from .tecio_wrapper import write_szplt_2d, write_ndarray_1d,write_szplt_3d ,Szplt3DWriter

def demo_1d():
    """Petit exemple d’écriture 1D"""
    import numpy as np
    x = np.linspace(0, 1, 100)
    y = np.sin(2 * np.pi * x)
    write_ndarray_1d("demo_1d.plt", ["x", "y"], [x, y])

def demo_2d():
    """Petit exemple d’écriture 2D"""
    import numpy as np
    x = np.linspace(-1, 1, 100)
    y = np.linspace( 0, 2, 20 )
    X, Y = np.meshgrid(x, y)
    Z = np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
    write_szplt_2d("demo_2d.plt", ["X", "Y", "Z"], [X, Y, Z])

def demo_3d():
    """Petit exemple d’écriture 2D"""
    import numpy as np
    x = np.linspace(-1, 1, 100)
    y = np.linspace( 0, 2, 20 )
    z = np.linspace( 0, 2, 20 )
    X, Y, Z = np.meshgrid(x, y, z,indexing="ij")
    FI = np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
    write_szplt_3d("demo_3d.plt", ["X", "Y", "Z", "FI"], [X, Y, Z,FI])




__all__ = ["write_szplt_2d", "write_ndarray_1d","write_szplt_3d","demo_1d","Szplt3DWriter"]
