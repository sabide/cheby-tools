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



def wall_profile(y, f, mode=+1, average=True, rtol=1e-6, atol=1e-6):
    """
    Extrait un profil à la paroi.

    Paramètres
    ----------
    y : array_like
        Coordonnée verticale normalisée dans [0, 1].
    f : array_like
        Champ à extraire.
    mode : {-1, +1}
        -1 -> paroi inférieure (y in [0, 0.5]), d = y
        +1 -> paroi supérieure (y in [0.5, 1]), d = 1 - y
        Ignoré si average=True.
    average : bool
        Si True, on suppose un maillage symétrique et on renvoie la moyenne
        (bas/haut) pour des distances d identiques. Si les d ne coïncident pas
        (hors tolérance), on lève une erreur.
    rtol, atol : float
        Tolérances pour la comparaison des distances (np.allclose).

    Retour
    ------
    d, f_out : np.ndarray
        d croissant depuis la paroi (0 -> 0.5),
        f_out est soit la moitié choisie (mode), soit la moyenne (si average=True).
    """
    y = np.asarray(y)
    f = np.asarray(f)
    if y.shape != f.shape:
        raise ValueError("y et f doivent avoir la même forme")

    # bas : d=y ; haut : d=1-y
    mask_b = y <= 0.5
    mask_h = y >= 0.5

    d_b = y[mask_b].copy()
    f_b = f[mask_b].copy()
    d_h = (1.0 - y[mask_h]).copy()
    f_h = f[mask_h].copy()

    # trier par distance croissante (depuis chaque paroi)
    idx_b = np.argsort(d_b)
    d_b, f_b = d_b[idx_b], f_b[idx_b]
    idx_h = np.argsort(d_h)
    d_h, f_h = d_h[idx_h], f_h[idx_h]

    if average:
        # Impair : on prend la plus petite moitié (on retire le point médian en double)
        m = min(len(d_b), len(d_h))
        d_b, f_b = d_b[:m], f_b[:m]
        d_h, f_h = d_h[:m], f_h[:m]

        # Vérifier que les distances coïncident (maillage réellement symétrique)
        if not np.allclose(d_b, d_h, rtol=rtol, atol=atol):
            for i,d in enumerate(d_b):
                print(i,abs(d-d_h[i])) 
            raise ValueError(
                "Distances d non identiques entre bas et haut (hors tolérance). "
                "Le maillage n'est pas parfaitement symétrique."
            )

        d = d_b  # identique à d_h
        f_out = 0.5 * (f_b + f_h)
        return d, f_out

    # Comportement original si average=False
    if mode not in (-1, +1):
        raise ValueError("mode doit être -1 (bas) ou +1 (haut)")

    if mode == -1:
        return d_b, f_b
    else:
        return d_h, f_h


import numpy as np

def wall_profiles(y, *fields, func=None, **kwargs):
    """
    Applique `wall_profile` (ou `func`) à y et à chaque champ de `fields`.
    Retourne (d, f1, f2, ...), prêt à être unpacké.
    """
    if func is None:
        func = wall_profile  # suppose que wall_profile est défini plus haut dans le fichier

    if len(fields) == 0:
        raise ValueError("Aucun champ à transformer.")

    d, out0 = func(y, fields[0], **kwargs)
    outs = [out0]
    for f in fields[1:]:
        _, fi = func(y, f, **kwargs)
        outs.append(fi)
    return (d, *outs)

