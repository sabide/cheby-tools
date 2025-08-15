from .tecio_wrapper import write_szplt, write_ndarray_1d
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
    write_szplt("demo_2d.plt", ["X", "Y", "Z"], [X, Y, Z])


