import utilities.Filenames as F
import sqlite3
import utilities.MyUtils_dbs as MyUtils_dbs
import logging
import utilities.MyUtils as MyUtils
import utilities.MyUtils_strings as MyUtils_strings
import numpy as np
import scipy.spatial.distance as distance


##################
###### Distance between vectors, and computation of the distance measure between a P and a Q for ranking purposes
#### Version 1 : cosine distance, euclidean distance, manhattan distance
def get_distance_between_vectors_allmeasures(u,v):
    cosine_dist = distance.cosine(u,v)
    euclidean_dist = np.linalg.norm(u-v)
    manhattan_dist = distance.cityblock(u,v)

    distance_uv = 0.5 * cosine_dist + 0.25 * euclidean_dist + 0.25 * manhattan_dist
    return distance_uv

#### Version 2 (current) : only cosine distance
def get_distance_between_vectors(u,v):
    cosine_dist = distance.cosine(u,v)
    return cosine_dist


def get_title_text_distance(prod_tuple, q_tuple):
    try:
        p_titlevec = np.array(MyUtils_strings.fromstring_toarray(prod_tuple.titlevec))
        q_questionVec = np.array(MyUtils_strings.fromstring_toarray(q_tuple.questionVec))
        ptitle_qtext_distance = get_distance_between_vectors(p_titlevec, q_questionVec)
        return ptitle_qtext_distance
    except NameError:
        return None

def get_desc_text_distance(prod_tuple, q_tuple):
    try:
        p_descvec = np.array( MyUtils_strings.fromstring_toarray(prod_tuple.descvec) )
        q_questionVec = np.array( MyUtils_strings.fromstring_toarray(q_tuple.questionVec) )
        pdesc_qtext_distance = get_distance_between_vectors(p_descvec, q_questionVec)
        return pdesc_qtext_distance
    except NameError:
        return None


def get_keywords_text_distance(prod_tuple, q_tuple):
    try:
        p_keywords = np.array(MyUtils_strings.fromlls_toarrays(prod_tuple.kwsVectors))
        q_questionVec = np.array(MyUtils_strings.fromstring_toarray(q_tuple.questionVec))
        cosine_distances = list(map(lambda q_keywordV: distance.cosine(q_keywordV, q_questionVec), p_keywords))
        avg_cosine_dist = sum(cosine_distances) / len(cosine_distances)
        return avg_cosine_dist
    except NameError:
        return None

def get_keywords_keywords_distance(prod_tuple, q_tuple):
    try:
        p_keywords = np.array(MyUtils_strings.fromlls_toarrays(prod_tuple.kwsVectors))
        q_keywords = np.array(MyUtils_strings.fromlls_toarrays(q_tuple.kwsVectors))
        m = len(p_keywords)
        n = len(q_keywords)
        sim_matrix = np.ones(shape=(m, n)) * -1
        for i in range(m):
            kw_vec_1 = p_keywords[i]
            for j in range(n):
                kw_vec_2 = q_keywords[j]
                sim_matrix[i][j] = 1 - distance.cosine(u=kw_vec_1, v=kw_vec_2)
        # logging.debug("\nThe sim.matrix : %s", sim_matrix)

        max_similarities = MyUtils.pick_maxmatches_matrix(sim_matrix)

        min_distances = list( map(lambda sim : 1-sim , max_similarities))

        avg_min_distance = np.average(min_distances)

        return avg_min_distance
    except NameError:
        return None


def compute_dist_pq(prod_tuple, q_tuple):
    logging.debug("***************")
    ptitle_qtext_distance = get_title_text_distance(prod_tuple, q_tuple)
    pdesc_qtext_distance = get_desc_text_distance(prod_tuple, q_tuple)
    pkeyws_qtext_distance = get_keywords_text_distance(prod_tuple, q_tuple)
    pkeyws_qkeyws_distance = get_keywords_keywords_distance(prod_tuple, q_tuple)

    #n: the first distance, at index [0], is the distance between the core features of P.desc and Q.text
    distances_all = [pdesc_qtext_distance, ptitle_qtext_distance, pkeyws_qtext_distance, pkeyws_qkeyws_distance]
    distances_not_none = list(filter( lambda distance: distance is not None, distances_all))
    #logging.info("distances_not_none: %s", distances_not_none)

    if pdesc_qtext_distance is not None:
        weights_unit = 1 / (len(distances_not_none) + 1)
        weighted_distances_not_none = list ( map (lambda dist : weights_unit * dist ,distances_not_none))
        weighted_distances_not_none[0] = 2* weighted_distances_not_none[0] #main features
        #logging.info("weighted_distances_not_none: %s", weighted_distances_not_none)
    else:
        #no core features; the others have equal weights
        weights_unit = 1 / len(distances_not_none)
        weighted_distances_not_none = list(map(lambda dist: weights_unit * dist, distances_not_none))

    pq_distance = sum(weighted_distances_not_none)
    #logging.info("pq_distance: %s", pq_distance)

    return pq_distance

##################
