import numpy as np

def load_dns_database_sebilleau(filename, variables=None):
    """
    Charge la base DNS (format Sébilleau) et retourne un dictionnaire de tableaux numpy.
    
    Parameters
    ----------
    filename : str
        Chemin du fichier à lire.
    variables : list of str, optional
        Liste des variables à extraire (ex: ["x", "T", "uu"]).
        Si None, toutes les colonnes sont retournées.
    
    Returns
    -------
    data : dict[str, np.ndarray]
        Dictionnaire {nom_variable: tableau numpy}
    """
    # Lire le fichier ligne par ligne pour trouver l'entête colonnes
    with open(filename, "r") as f:
        lines = f.readlines()
    
    # Trouver la ligne qui commence par un '|'
    header_line = None
    for i, line in enumerate(lines):
        if line.strip().startswith("|"):
            header_line = i
            break
    
    if header_line is None:
        raise RuntimeError("Impossible de trouver la ligne d'entête avec les noms de variables.")
    
    # Extraire les noms de colonnes
    header = [h.strip() for h in lines[header_line].strip("| \n").split("|")]
    
    # Charger les données numériques à partir de la ligne suivante
    data_array = np.loadtxt(filename, skiprows=header_line+1)
    
    # Construire un dictionnaire {colonne: tableau}
    data = {name: data_array[:, idx] for idx, name in enumerate(header)}
    
    # Si l'utilisateur demande un sous-ensemble
    if variables is not None:
        missing = [v for v in variables if v not in data]
        if missing:
            raise ValueError(f"Colonnes non trouvées dans le fichier: {missing}")
        data = {v: data[v] for v in variables}
    
    return data


import numpy as np



import numpy as np

def wall_profile(y, f, mode=+1):
    """
    Extrait un profil à la paroi:
      mode = -1 -> paroi inférieure (y in [0, 0.5]),   d = y
      mode = +1 -> paroi supérieure (y in [0.5, 1]),   d = 1 - y
    Retourne (d, f_half) avec d croissant depuis la paroi (0 -> 0.5).
    """
    y = np.asarray(y)
    f = np.asarray(f)

    if mode not in (-1, +1):
        raise ValueError("mode doit être -1 (bas) ou +1 (haut)")

    if mode == -1:
        # moitié basse
        mask = y <= 0.5
        d = y[mask]
        f_half = f[mask]
        # trier par distance croissante
        idx = np.argsort(d)
        d, f_half = d[idx], f_half[idx]
    else:
        # moitié haute
        mask = y >= 0.5
        d = 1.0 - y[mask]  # distance à la paroi du haut
        f_half = f[mask]
        # trier par distance croissante (depuis la paroi du haut)
        idx = np.argsort(d)
        d, f_half = d[idx], f_half[idx]

    return d, f_half

