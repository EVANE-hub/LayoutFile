
def join_path(*args):
    return '/'.join(str(s).rstrip('/') for s in args)


def get_top_percent_list(num_list, percent):
    """
    Obtient les premiers éléments d'une liste selon un pourcentage donné
    :param num_list: la liste d'entrée
    :param percent: le pourcentage à extraire
    :return: la liste des premiers éléments
    """
    if len(num_list) == 0:
        top_percent_list = []
    else:
        # Trie la liste imgs_len_list
        sorted_imgs_len_list = sorted(num_list, reverse=True)
        # Calcule l'index pour le pourcentage
        top_percent_index = int(len(sorted_imgs_len_list) * percent)
        # Prend les premiers éléments selon le pourcentage
        top_percent_list = sorted_imgs_len_list[:top_percent_index]
    return top_percent_list


def mymax(alist: list):
    if len(alist) == 0:
        return 0  # Retourne 0 pour une liste vide
    else:
        return max(alist)


