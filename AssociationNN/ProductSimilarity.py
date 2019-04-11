import sys
import os

import utilities.MyUtils as MyUtils
import utilities.MyUtils_strings as MyUtils_strings
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath('../Product Features'))
#print(sys.path)
import logging
import scipy.spatial.distance
import numpy
import ast
from time import time

def compute_cosinesim_2arraystrings(a1_string, a2_string):
    start_0 = time()
    a1 = MyUtils_strings.fromstring_toarray(a1_string)
    end_0 = time()
    logging.debug("\t From string to array: time : %s seconds", round(end_0-start_0,6))
    a2 = MyUtils_strings.fromstring_toarray(a2_string)
    start_1 = time()
    cosine_sim = 1 - scipy.spatial.distance.cosine(u=a1, v=a2)
    end_1 = time()
    logging.debug("\t Compute cosine similarity: time : %s seconds", round(end_1 - start_1, 6))
    return cosine_sim


###### Keywords' vectors : matrix processing

def get_kws_similarity(kwsVectors_stringls_1, kwsVectors_stringls_2):
    #logging.debug(kwsVectors_stringls_1)
    #logging.debug(type(kwsVectors_stringls_1))
    kwsVectors_ls_1 = MyUtils_strings.fromlls_toarrays(kwsVectors_stringls_1)
    kwsVectors_ls_2 = MyUtils_strings.fromlls_toarrays(kwsVectors_stringls_2)
    m = len(kwsVectors_ls_1)
    n = len(kwsVectors_ls_2)
    sim_matrix = numpy.ones(shape=(m,n)) * -1
    for i in range(m):
        kw_vec_1 = kwsVectors_ls_1[i]
        #logging.debug("Vector 1 : %s, kw_vec_1)
        for j in range(n):
            kw_vec_2 = kwsVectors_ls_2[j]
            #logging.debug(kw_vec_2)
            sim_matrix[i][j] =  1 - scipy.spatial.distance.cosine(u=kw_vec_1, v=kw_vec_2)
    logging.debug("\nThe keywords sim.matrix : %s", sim_matrix)
    kws_aggregated_sim = numpy.average(MyUtils.pick_maxmatches_matrix(sim_matrix))
    logging.debug("Aggregated keywords' similarity: %s", kws_aggregated_sim)
    return kws_aggregated_sim

############

def get_categories_contribution(ls_1, ls_2, d2v_model):
    weight_jaccardsim= 0.7
    weight_avgbestmatches = 0.3

    jaccardsim = compute_Jaccardsim(ls_1, ls_2)
    jaccardsim_adjusted = (jaccardsim * 2) - 1 #the Jaccard similarity gets transposed from [0,1] to [-1,1]

    # 2nd part: average of the best matches from the matrix of comparisons (similarly to the keywords)
    lls1_vectors = [d2v_model.infer_vector([category]) for category in ls_1] #a bit slower. Save to database the inferred vectors?
    lls2_vectors = [d2v_model.infer_vector([category]) for category in ls_2]
    avgbestmatches = get_kws_similarity(str(lls1_vectors), str(lls2_vectors))
    logging.debug("(Transposed) Jaccard similarity: %s", jaccardsim_adjusted )
    logging.debug("Average of best matches: %s", avgbestmatches)
    return weight_jaccardsim * jaccardsim_adjusted + weight_avgbestmatches * avgbestmatches


def compute_Jaccardsim(ls_1, ls_2):
    #logging.info(ls_1)
    #logging.info(ls_2)
    set_1 = set(ls_1)
    set_2 = set(ls_2)
    intersection = set_1.intersection(set_2)
    union = set_1.union(set_2)
    return len(intersection) / len(union)


###### Computing the similarity between 2 products:

def compute_2products_similarity_singleprocess(prod1_tuple, prod2_tuple, d2v_model):
    weights = {"w_titlevec":-1, "w_descvec":-1,
               "w_mdcategories":-1, "w_kwsVectors":-1}
    titlevec_cosine_sim = -2
    descvec_cosine_sim = -2
    categories_sim = -2
    kws_sim = -2

    start_0 = time()
    if prod1_tuple.descvec == "NODESCVEC" or prod2_tuple.descvec == "NODESCVEC":
        weights["w_descvec"] = 0
        weights["w_kwsVectors"] = 0
    else:
        descvec_cosine_sim = compute_cosinesim_2arraystrings(prod1_tuple.descvec, prod2_tuple.descvec)
        kws_sim = get_kws_similarity(prod1_tuple.kwsVectors, prod2_tuple.kwsVectors)
        end_0 = time()
        logging.debug("Descvec cosine sim: %s", descvec_cosine_sim)
        logging.debug("Kws sim: %s", kws_sim)
        logging.debug("Product similarity: desc & keywords time : %s seconds", round(end_0-start_0,6))
    start_1 = time()

    if prod1_tuple.titlevec == "NOTITLEVEC" or prod2_tuple.titlevec == "NOTITLEVEC":
         weights["w_titlevec"] = 0
    else:
         titlevec_cosine_sim = compute_cosinesim_2arraystrings(prod1_tuple.titlevec, prod2_tuple.titlevec)
         logging.debug("Titlevec cosine sim: %s", titlevec_cosine_sim)
    end_1 = time()
    logging.debug("Product similarity: title time : %s seconds", round(end_1 - start_1, 6))

    if len(prod1_tuple.mdcategories) > 1 and len(prod2_tuple.mdcategories) > 1:
        if len(ast.literal_eval(prod1_tuple.mdcategories)) == 0 or len(ast.literal_eval(prod2_tuple.mdcategories)) == 0:
             weights["w_mdcategories"] = 0
        else:
             categories_sim = get_categories_contribution(ast.literal_eval(prod1_tuple.mdcategories),
                                ast.literal_eval(prod2_tuple.mdcategories), d2v_model)
             logging.debug("Categories sim: %s" ,categories_sim)
    else:
        weights["w_mdcategories"] = 0

    #logging.info(weights)
    weights = establish_factors_weights(weights)

    final_sim = weights["w_titlevec"] * titlevec_cosine_sim + \
                weights["w_descvec"] * descvec_cosine_sim + \
                weights["w_kwsVectors"] * kws_sim + \
                weights["w_mdcategories"] * categories_sim
    sim_parts = (titlevec_cosine_sim, descvec_cosine_sim, kws_sim, categories_sim)  # for exploration purposes

    transposed_final_sim = (final_sim + 1) / 2  # from [-1,1] to [0,1]

    if transposed_final_sim is None: #debug gate
        for name, value in globals().items():
            logging.error(name, value)
            raise Exception

    return (transposed_final_sim, sim_parts)



def establish_factors_weights(ws_dict):

    if ws_dict["w_descvec"] == -1 and ws_dict["w_kwsVectors"] == -1 \
    and ws_dict["w_titlevec"] == -1 and ws_dict["w_mdcategories"] == -1:

        ws_dict["w_descvec"] = 0.55
        ws_dict["w_kwsVectors"] = 0.15
        ws_dict["w_titlevec"] = 0.15
        ws_dict["w_mdcategories"] = 0.15
        return ws_dict

    if ws_dict["w_descvec"] == 0  and ws_dict["w_kwsVectors"] == 0\
    and ws_dict["w_titlevec"] == -1 and ws_dict["w_mdcategories"] == -1:

        ws_dict["w_titlevec"] = 0.7
        ws_dict["w_mdcategories"] = 0.3
        return ws_dict

    if ws_dict["w_descvec"] == 0 and ws_dict["w_kwsVectors"] == 0 \
    and ws_dict["w_titlevec"] == 0 and ws_dict["w_mdcategories"] == -1:

        ws_dict["w_mdcategories"] = 1
        return ws_dict

    if ws_dict["w_descvec"] == 0 and ws_dict["w_kwsVectors"] == 0 \
    and ws_dict["w_titlevec"] == -1 and ws_dict["w_mdcategories"] == 0:

        ws_dict["w_titlevec"] = 1
        return ws_dict

    if ws_dict["w_descvec"] == -1 and ws_dict["w_kwsVectors"] == -1 \
    and ws_dict["w_titlevec"] == 0 and ws_dict["w_mdcategories"] == -1:

        ws_dict["w_descvec"] = 0.6
        ws_dict["w_kwsVectors"] = 0.2
        ws_dict["w_mdcategories"] = 0.2
        return ws_dict

    if ws_dict["w_descvec"] == -1 and ws_dict["w_kwsVectors"] == -1 \
    and ws_dict["w_titlevec"] == -1 and ws_dict["w_mdcategories"] == 0:

        ws_dict["w_descvec"] = 0.6
        ws_dict["w_kwsVectors"] = 0.2
        ws_dict["w_titlevec"] = 0.2
        return ws_dict


    if ws_dict["w_descvec"] == -1 and ws_dict["w_kwsVectors"] == -1 \
    and ws_dict["w_titlevec"] == 0 and ws_dict["w_mdcategories"] == 0:

        ws_dict["w_descvec"] = 0.7
        ws_dict["w_kwsVectors"] = 0.3
        return ws_dict

    if ws_dict["w_descvec"] == -1 and ws_dict["w_kwsVectors"] == 0 \
    and ws_dict["w_titlevec"] == 0 and ws_dict["w_mdcategories"] == 0:

        ws_dict["w_descvec"] = 1
        return ws_dict


    if ws_dict["w_descvec"] == 0 and ws_dict["w_kwsVectors"] == 0 \
    and ws_dict["w_titlevec"] == 0 and ws_dict["w_mdcategories"] == 0:
        return ws_dict



############
