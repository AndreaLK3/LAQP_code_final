import numpy as np
import logging
import sys
import os

import utilities.MyUtils_strings

sys.path.append(os.path.abspath('../Product Features'))
import utilities.Filenames as F
import utilities.MyUtils as MyUtils
import ast
import pandas as pd
import pickle
import sqlite3
from time import time
import gensim.models.doc2vec as D2V
#import networkx as nx
#import json
from gc import collect

########## Function to obtain the numerical encoding of a Question, for the input to the NN
def get_num_encoding_quest(quest_t):

    if quest_t.questionType == 'open-ended':
        has_questionType = True
        qf_type = [0]
    elif quest_t.questionType =='yes/no':
        has_questionType = True
        qf_type = [2]
    else:
        has_questionType = False
        qf_type = []

    if quest_t.questionVec == "NOQUESTVEC":
        has_questionVec = False
        qf_descvec = []
    else:
        has_questionVec = True
        qf_descvec = list(utilities.MyUtils_strings.fromstring_toarray(quest_t.questionVec)) #the array of 200 features, was first in Doc2Vec, then in csv

    if quest_t.kwsVectors == "NOKWSVECTORS":
        has_kwsVectors = False
        qf_kwsVectors_avg = []
    else:
        has_kwsVectors = True
        qf_kwsVectors_all = utilities.MyUtils_strings.fromlls_toarrays(quest_t.kwsVectors)
        #logging.info(qf_kwsVectors_all)
        qf_kwsVectors_avg = (np.average(qf_kwsVectors_all, axis=0)).tolist()
        if str(qf_kwsVectors_avg) == "nan":
            has_kwsVectors = False
            qf_kwsVectors_avg = []
            logging.warning("Kws vectors: %s", qf_kwsVectors_all)

    flags = [has_questionType, has_questionVec, has_kwsVectors]
    encodings = [qf_type, qf_descvec , qf_kwsVectors_avg]

    return (flags, encodings)
##########


########## Main function for questions: computes the encoding for all of them, and saves it
########## to database. The features that each element may (or may not) have are indicated by flags.
def compute_encoding_questions(questions_filepath, out_db_filepath):
    MyUtils.init_logging("ComputeEncodingQuestions.log")
    questions_file = open(questions_filepath, "r")

    f = open(out_db_filepath, mode="w");
    f.close()  # clean between runs
    db_conn = sqlite3.connect(out_db_filepath)
    c = db_conn.cursor()
    c.execute('''CREATE TABLE qs_numenc(q_id varchar(63) NOT NULL,
                                        has_questionType tinyint,    
                                        has_questionVec tinyint,
                                        has_kwsVectors tinyint,
                                        encoding_questionType varchar(15),
                                        encoding_questionVec varchar(8191),
                                        encoding_kwsVectors varchar(8191),
                                        PRIMARY KEY (q_id) )''')
    db_conn.commit()

    segment_size = 5 * 10**3
    segment_id = 1
    for input_segment in pd.read_csv(questions_file, sep="_", chunksize=segment_size):
        segment_start = time()
        for quest_t in input_segment.itertuples():
            if len(quest_t.id) >=5 : #filter out undue headers
                #logging.info(quest_t)
                (q_flags, encodings) = get_num_encoding_quest(quest_t)
                c.execute('''INSERT INTO qs_numenc VALUES (?,?,?,?,?,?,?);''',
                          (quest_t.id, int(q_flags[0]), int(q_flags[1]), int(q_flags[2]),
                                 str(encodings[0]), str(encodings[1]), str(encodings[2])))
        segment_end = time()
        db_conn.commit()
        logging.info("Encoded questions' chunk n. %s ... Time elapsed = %s", segment_id, round(segment_end - segment_start,3))
        segment_id = segment_id+1



########## Part II: Products

########## Method for categories: Creating the vocabulary based on one-hot encoding
########## n: currently unused. (see: number of resulting input features, etc.)
def create_onehotvocab_forcategories():
    MyUtils.init_logging("TrainingSet_AssociationNN.log", logging.DEBUG)
    current_index = 0
    indices_dict = {}

    segment_size = 5* 10**4
    cats_file = open(F.CATEGORIES_DICTIONARY, "wb")
    for segment in pd.read_csv(F.PRODUCTS_FINAL_TRAIN, iterator=True, chunksize=segment_size, sep="_"):
        for prod_t in segment.itertuples():
            if len(prod_t.id) < 10:  #(again, filtering out any undue headers)
                continue
            p_cats = ast.literal_eval(prod_t.mdcategories)
            if len(p_cats) > 1:
                logging.debug(p_cats)
            for category in p_cats:
                if category not in indices_dict:
                    indices_dict[category] = current_index
                    current_index = current_index+1
        logging.info("Registering categories in the 'index vocabulary' for the encoding; chunk processed...")
        logging.info("Number of categories currently registered: %s", len(indices_dict))
    pickle.dump(file=cats_file, obj=indices_dict, protocol=pickle.HIGHEST_PROTOCOL)
    logging.info("Categories' 'index vocabulary' completed. Number of categories: %s", len(indices_dict))
    logging.info("%s", str(indices_dict))
    cats_file.close()


########## Obtaining the numerical encoding of a list of categories
def get_onehotcode_categories(mdcategories_ls, categories_indicesdict):
    num_allcats = len(categories_indicesdict)
    pf_cats = np.zeros(shape=num_allcats)
    for category in mdcategories_ls:
        index = categories_indicesdict[category]
        pf_cats[index] = 1
    logging.debug(pf_cats)
    return pf_cats.tolist()

##########


########## Method for categories: average of the D2V vectors. Uses infer_vector
def get_categories_doc2vec_average(categories_ls, doc2vec_model):
    w_unit = 1 / ( len(categories_ls) + 1 )
    array_result = np.zeros(doc2vec_model.infer_vector(["word"]).shape) #gives (200,) or other, depending on what we have been using
    for i in range(len(categories_ls)):
        if i == 0:
            weight = w_unit * 2
        else:
            weight = w_unit
        array_result = array_result + weight * doc2vec_model.infer_vector([categories_ls[i]])
    return array_result


########## Method for categories (exploring, experimental): build a graph
# def create_graph_forcategories():
#     G = nx.DiGraph()
#     #G.add_edge(1, 2)
#     #G.add_edges_from([(1, 2), (1, 3)])
#     #plt.subplot(121)
#     #nx.draw(G, with_labels=True, font_weight='bold')
#     #plt.show()
#
#     segment_size = 10**4
#     for segment in pd.read_csv(F.PRODUCTS_FINAL_TRAIN, iterator=True, chunksize=segment_size, sep="_"):
#         seg_start = time()
#         for prod_t in segment.itertuples():
#              if len(prod_t.id) < 10:  #(again, filtering out any undue headers)
#                  continue
#              p_cats = ast.literal_eval(prod_t.mdcategories)
#              for i in range(len(p_cats) - 1):
#                 u = p_cats[i]
#                 v = p_cats[i+1]
#                 G.add_edge(u, v)
#         seg_end = time()
#         logging.info("Creating the categories' graph: chunk of products processed... Time elapsed=%s",
#                      round(seg_end-seg_start,3))
#     nx.write_gpickle(G=G, path="graph_of_categories.pickle")
#     graphdata = nx.readwrite.json_graph.node_link_data(G)# {'link': 'edges', 'source': 'from', 'target': 'to'}
#     with open('categories_graphdata.json', 'w') as graph_file:
#         s2 = json.dumps(graphdata)# default={'link': 'edges', 'source': 'from', 'target': 'to'}
#         graph_file.write(s2)


########## Function to obtain the numerical encoding of a Product, for the input to the NN
def get_num_encoding_prod(prod_t, doc2vec_model):

    if prod_t.titlevec == "NOTITLEVEC":
        has_titlevec = False
        pf_titlevec = []
    else:
        has_titlevec = True
        pf_titlevec = list(utilities.MyUtils_strings.fromstring_toarray(prod_t.titlevec))


    if prod_t.descvec == "NODESCVEC":
        has_descvec = False
        pf_descvec = []
    else:
        has_descvec = True
        try:
            pf_descvec = list(utilities.MyUtils_strings.fromstring_toarray(prod_t.descvec))
        except SyntaxError as e:
            logging.warning(e)
            logging.warning("Failed to read the description vector from string to list for the Product %s", prod_t.id)
            has_descvec = False
            pf_descvec = []


    mdcategories_ls = ast.literal_eval(prod_t.mdcategories)
    if len(mdcategories_ls) == 0:
        has_mdcategories = False
        pf_mdcategories = []
    else:
        has_mdcategories = True
        pf_mdcategories = list(get_categories_doc2vec_average(mdcategories_ls, doc2vec_model))

    if prod_t.kwsVectors == "NOKWSVECTORS":
        has_kwsVectors = False
        pf_kwsVectors_avg = []
    else:
        has_kwsVectors = True
        pf_kwsVectors_all = utilities.MyUtils_strings.fromlls_toarrays(prod_t.kwsVectors)
        pf_kwsVectors_avg = (np.average(pf_kwsVectors_all, axis=0)).tolist()
        if str(pf_kwsVectors_avg) == "nan":
            has_kwsVectors = False
            pf_kwsVectors_avg = []
            logging.warning("Kws vectors: %s", pf_kwsVectors_all)


    flags = [has_titlevec, has_descvec, has_mdcategories, has_kwsVectors]
    encodings = [pf_titlevec, pf_descvec, pf_mdcategories, pf_kwsVectors_avg]
    return (flags, encodings)

##########

########## Main function for products: near-copy of the one for questions,
########## determines flags and encoding for all the products, and saves them to database
def compute_encoding_products(products_filepath, out_db_filepath):
    MyUtils.init_logging("ComputeEncodingProducts.log")
    products_file = open(products_filepath, "r")
    d2v_model = D2V.Doc2Vec.load(F.D2V_MODEL)

    f = open(out_db_filepath, mode="w");
    f.close()  # clean between runs
    db_conn = sqlite3.connect(out_db_filepath)
    c = db_conn.cursor()
    c.execute('''CREATE TABLE ps_numenc(p_id varchar(63) NOT NULL,
                                        has_titlevec tinyint,    
                                        has_descvec tinyint,
                                        has_mdcategories tinyint,
                                        has_kwsVectors tinyint,
                                        encoding_titlevec varchar(8191),
                                        encoding_descvec varchar(8191),
                                        encoding_mdcategories varchar(4095),
                                        encoding_kwsVectors varchar(8191),
                                        PRIMARY KEY (p_id) )''')
    db_conn.commit()

    segment_size = 5 * 10**3
    segment_n = 1
    for input_segment in pd.read_csv(products_file, sep="_", chunksize=segment_size):
        segment_start = time()
        for prod_t in input_segment.itertuples():
            if len(prod_t.id) >=5 : #filter out undue headers
                #logging.info(quest_t)
                (q_flags, encodings) = get_num_encoding_prod(prod_t, d2v_model)
                c.execute('''INSERT INTO ps_numenc VALUES (?,?,?,?,?,?,?,?,?);''',
                          (prod_t.id, int(q_flags[0]), int(q_flags[1]), int(q_flags[2]), int(q_flags[3]),
                           str(encodings[0]), str(encodings[1]), str(encodings[2]), str(encodings[3])))
        segment_end = time()
        db_conn.commit()
        collect()
        logging.info("chunk n.%s of products encoded. Time elapsed = %s", segment_n , round(segment_end - segment_start,3))
        segment_n = segment_n+1


def exe_train():
    compute_encoding_products(F.PRODUCTS_FINAL_TRAIN, F.PRODS_NUMENCODING_DB_TRAIN)
    compute_encoding_questions(F.QUESTIONS_FINAL_TRAIN, F.QUESTS_NUMENCODING_DB_TRAIN)


def exe_valid():
    compute_encoding_products(F.PRODUCTS_FINAL_VALID, F.PRODS_NUMENCODING_DB_VALID)
    compute_encoding_questions(F.QUESTIONS_FINAL_VALID, F.QUESTS_NUMENCODING_DB_VALID)

def exe_test():
    compute_encoding_products(F.PRODUCTS_FINAL_TEST, F.PRODS_NUMENCODING_DB_TEST)
    compute_encoding_questions(F.QUESTIONS_FINAL_TEST, F.QUESTS_NUMENCODING_DB_TEST)




