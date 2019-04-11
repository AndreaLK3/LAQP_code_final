import utilities.Filenames as F
import sqlite3
import utilities.MyUtils as MyUtils
import logging
import numpy as np
import scipy.spatial.distance as S_distance
import matplotlib.pyplot as plt
import gensim.models.doc2vec as D2V
from gc import collect
import os.path

import utilities.MyUtils_dbs
import utilities.MyUtils_strings

NUM_EXAMPLE_PRODUCTS = 4000
NUM_EXAMPLE_QUESTIONS = 4000

### Utility wrappers: use the conversion functions from string to array, with the addition of an exception wrapper.
### num_levels_ls = 1 if we are extracting single vectors (eg.descvec); =2 if dealing with a lls (eg. kwsVectors)
def extract_features_arrays(num_levels_ls, arraystrings_ls):
    arrays = []
    for s in arraystrings_ls:
        try:
            if num_levels_ls == 1:
                arrays.append(utilities.MyUtils_strings.fromstring_toarray(s))
            if num_levels_ls == 2:
                arrays.append(utilities.MyUtils_strings.fromlls_toarrays(s))
        except Exception as e:
            logging.debug("Could not convert string %s to array. The selected product does not have that feature", s)
    return arrays

##### Get the random indices, through which we select elements (products or questions) from the tables in the DBs
def get_random_indices(db_cursor, n_elements):
    db_cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = db_cursor.fetchall()
    t_cardinality = db_cursor.execute("SELECT count(*) FROM {t}".format(t=tables[0][0])).fetchone()[0]
    logging.info("Cardinality of a table in the representations' database: %s", t_cardinality)

    num_products_picked_per_table = min(n_elements // len(tables) + 1, t_cardinality)
    logging.info("Number of elements picked in each table: %s ; number of tables:%s", num_products_picked_per_table,
                 len(tables))
    random_indices = np.random.choice(range(1, t_cardinality + 1), size=num_products_picked_per_table, replace=False)
    return random_indices

##### Extract the product vectors that we wish to compare using cosine similarity
def get_products_vectors(p_featurename, n_products, selected_pf_strings, doc2vec_model):

    if p_featurename.lower() == "descvec" or p_featurename.lower() == "titlevec":
        p_fs = extract_features_arrays(1, selected_pf_strings)[0:n_products] # Also applies the n_products specification

    if p_featurename.lower() == "kwsvectors":
        p_fs = list(filter ( lambda lls : len(lls) > 0, extract_features_arrays(2, selected_pf_strings) ) )
        #logging.info(np.array(p_fs).shape)
        #logging.info(p_fs[0:10])
        p_fs = p_fs[0:n_products] # Also applies the n_products specification

    if p_featurename.lower() == "mdcategories":
        p_fs_0 = [utilities.MyUtils_strings.fromstring_toarray(product_categoriesls, whitespaces_to_commas=False)
                  for product_categoriesls in selected_pf_strings]
        #logging.info("p_fs_0:%s", p_fs_0)
        p_fs_1 = utilities.MyUtils_strings.categories_to_vecs_lls(p_fs_0, doc2vec_model)
        #logging.info("p_fs_1:%s", p_fs_1)
        p_fs = list(filter(lambda lls: len(lls) > 0, p_fs_1))
        p_fs = p_fs[0:n_products]

    #logging.info("p_fs[0] : %s", p_fs[0])
    logging.info("Number of product vectors extracted: %s", len(p_fs))
    return p_fs

### Extract the product vectors that we wish to compare using cosine similarity
def get_questions_vectors(q_featurename, n_questions, selected_qf_strings):
    if q_featurename.lower() == "questionvec":
        q_fs = extract_features_arrays(1, selected_qf_strings)[0:n_questions]

    if q_featurename.lower() == "kwsvectors":
        q_fs = list(filter ( lambda lls : len(lls) > 0, extract_features_arrays(2, selected_qf_strings) ) )
        q_fs = q_fs [0:n_questions]

    logging.info("q_fs[0] : %s", q_fs[0])
    logging.info("Number of question vectors extracted: %s", len(q_fs))
    return q_fs


##### Once we have extracted the Doc2Vec vectors from the selected features for products and questions,
##### we compute a matrix of cosine similarities
def compute_matrix_cosinesims(ps_vectors, qs_vectors):
    #choice for the kwsVectors lls: either average, or unpack. Currently: average
    elements_vectorlists_0 = [ps_vectors, qs_vectors]
    elements_vectorlists = []
    for structure in elements_vectorlists_0:
        new_structure = []
        for elem_vector in structure:
            if len(np.array(elem_vector).shape) >= 2:
                avg_elem_vector = np.average(elem_vector, axis=0)
                new_structure.append(avg_elem_vector)
            else:
                new_structure.append(elem_vector)
        elements_vectorlists.append(new_structure)
    ps_vectors, qs_vectors = elements_vectorlists

    M = np.zeros(shape=(len(ps_vectors), len(qs_vectors)))
    i = 0; j = 0;
    for p_vec in ps_vectors:
        for q_vec in qs_vectors:
            cosine_sim = 1 - S_distance.cosine(p_vec, q_vec)
            M[i][j] = cosine_sim
            j = j+1
        i = i+1
        j = 0

    return M


##### Once we have the matrix M of cosine similarities, we compute a breakpoint for the given fraction (50%, 90%, etc)
def compute_breakpoint(matrix, fraction):
    flattened_matrix = matrix.flatten()
    values_sorted = sorted(flattened_matrix)
    threshold_index = int(fraction * len(values_sorted))
    sim_breakpoint = values_sorted[threshold_index]
    logging.info("At the fraction of: %s, the cosine similarity breakpoint is: %s", fraction, sim_breakpoint)

    return sim_breakpoint

##### Graphics
def show_graphic_cosinesim(M, p_featurename, q_featurename, breakpoint_x, fraction):
    plt.figure()
    plt.hist(M.flatten(), bins=400, range=(-1, 1))
    plt.title('Cosine similarity: products: {s1} - questions: {s2} | bp @ {s3}'.format(
                s1=p_featurename, s2=q_featurename,  s3=round(fraction,3)), )
    plt.grid(True)
    plt.axvline(x=breakpoint_x, color='maroon')
    plt.savefig(os.path.join("OnlineLearning", "CosineSimilarity", 'P_{s1}_Q_{s2}_{s3}.png'.format(
                s1=p_featurename, s2=q_featurename,  s3=round(fraction,3))) )
    plt.close()

##### Update database of similarity breakpoints
def update_breakpoints_db(breakpoint,p_featurename, q_featurename, n_products, n_questions, fraction):
    out_db = sqlite3.connect(F.COSINE_SIM_THRESHOLDS_DB)
    out_c = out_db.cursor()

    row_count = out_c.execute('''SELECT COUNT(*) from breakpoints 
                                 WHERE p_featurename = ?
                                     AND q_featurename = ?
                                     AND num_ps = ?
                                     AND num_qs = ? 
                                     AND threshold = ?''', (p_featurename, q_featurename,
                                                n_products, n_questions, fraction)).fetchone()[0]
    if row_count > 0:
        logging.info("Row count > 0; updating values")
        out_c.execute(''' UPDATE breakpoints
                            SET breakpoint = ?
                            WHERE p_featurename = ?
                             AND q_featurename = ?
                             AND num_ps = ?
                             AND num_qs = ? 
                             AND threshold = ?''', (breakpoint, p_featurename, q_featurename,
                                                    n_products, n_questions, fraction))
    else:
        logging.info("Row count = 0; inserting values")
        out_c.execute(''' INSERT INTO breakpoints VALUES (?,?,?,?,?,?);''',
                      (breakpoint, p_featurename, q_featurename, n_products, n_questions, fraction))
    out_db.commit()



### Main function of the module
### id, price, titlevec, descvec, mdcatagories, kwsVectors; id, questionType, questionVec, kwsVectors
def explore_cosine_similarity(n_products=100, n_questions=100, p_featurename="descvec", q_featurename="questionVec", fraction=0.75):
    MyUtils.init_logging("CosineSimilarity.log")
    logging.info("Computing a cosine similarity breakpoint at fraction %s, between P:%s and Q:%s ...",
                 fraction, p_featurename, q_featurename)

    prods_representations_db = sqlite3.connect(F.PRODUCTS_FINAL_TRAIN_DB)
    ps_c = prods_representations_db.cursor()

    quests_representations_db = sqlite3.connect(F.QUESTIONS_FINAL_TRAIN_DB)
    qs_c = quests_representations_db.cursor()

    ###### Get Doc2Vec vectors from the randomly selected products
    random_indices = get_random_indices(ps_c, n_products)
    random_indices_querystring = str(tuple(random_indices)) if len(random_indices) > 1 \
        else "(" + str(random_indices[0]) + ")"
    selected_pf_strings_ts = utilities.MyUtils_dbs.search_in_alltables_db(ps_c, "SELECT " + str(p_featurename) + " FROM",
                                                            "WHERE rowid IN " + random_indices_querystring)
    # Unpacking the tuples (each tuple is simply a container for one feature).
    selected_pf_strings = list(map(lambda t: t[0], selected_pf_strings_ts))

    d2v_model = D2V.Doc2Vec.load(F.D2V_MODEL) #it is loaded to obtain the vectors for the mdcategories
    ps_vectors = get_products_vectors(p_featurename, n_products, selected_pf_strings, d2v_model)
    del d2v_model
    collect()
    ######
    ###### Get Doc2Vec vectors from the randomly selected questions
    random_indices = get_random_indices(qs_c, n_questions)
    random_indices_querystring = str(tuple(random_indices)) if len(random_indices) > 1 \
        else "(" + str(random_indices[0]) + ")"
    selected_qf_strings_ts = utilities.MyUtils_dbs.search_in_alltables_db(qs_c, "SELECT " + str(q_featurename) + " FROM",
                                                            "WHERE rowid IN " + random_indices_querystring)

    selected_qf_strings = list(map(lambda t: t[0], selected_qf_strings_ts))
    qs_vectors = get_questions_vectors(q_featurename, n_questions, selected_qf_strings)
    ######

    M = compute_matrix_cosinesims(ps_vectors,qs_vectors)

    breakpoint = compute_breakpoint(M, fraction)

    show_graphic_cosinesim(M, p_featurename, q_featurename, breakpoint, fraction)

    update_breakpoints_db(breakpoint, p_featurename, q_featurename, n_products, n_questions, fraction)
    return breakpoint




def initialize_cosinesims_db():
    f = open(F.COSINE_SIM_THRESHOLDS_DB, 'w'); f.close()

    db = sqlite3.connect(F.COSINE_SIM_THRESHOLDS_DB)
    c = db.cursor()
    c.execute(''' CREATE TABLE breakpoints (breakpoint float,
                                            p_featurename varchar(127),
                                            q_featurename varchar(127),
                                            num_ps int,
                                            num_qs int,
                                            threshold float
                                            ) ''')
    db.commit()



def get_similarity_breakpoint(p_featurename, q_featurename, fraction):
    #MyUtils.init_logging("OnlineLearning-get_similarity_breakpoint.log")
    db = sqlite3.connect(F.COSINE_SIM_THRESHOLDS_DB)
    c = db.cursor()
    lts = c.execute('''SELECT breakpoint, num_ps, num_qs, threshold FROM breakpoints
                        WHERE p_featurename = ? AND q_featurename = ?
                        AND ABS(threshold - ?) < 0.0001''',
                    (p_featurename,q_featurename, fraction)).fetchall()

    num_bps_already_computed = len(lts)
    logging.info("number of breakpoints computed in the past:%s", num_bps_already_computed)

    if num_bps_already_computed > 0:
        lts_sorted = sorted(lts, key=lambda tpl: tpl[1]*tpl[2], reverse=True)
        logging.debug("Breakpoints, sorted by number of examples:%s", lts_sorted)
        if lts_sorted[0][1] * lts_sorted[0][2] < NUM_EXAMPLE_PRODUCTS * NUM_EXAMPLE_QUESTIONS:
            logging.info("A value was found for the estimated breakpoint, but it was created with fewer example "+
                         "products and questions. Creating new breakpoint...")
            bp = explore_cosine_similarity(n_products=NUM_EXAMPLE_PRODUCTS, n_questions=NUM_EXAMPLE_QUESTIONS,
                                           p_featurename=p_featurename, q_featurename=q_featurename,
                                           fraction=fraction)
        else:
            bp = lts_sorted[0][0] #pick what was already computed
    else:
        bp = explore_cosine_similarity(n_products=NUM_EXAMPLE_PRODUCTS, n_questions=NUM_EXAMPLE_QUESTIONS,
                                  p_featurename=p_featurename, q_featurename=q_featurename,
                                  fraction=fraction)

    logging.info("Cosine similarity breakpoint found for the features %s and %s: %s\n",
                 p_featurename, q_featurename, round(bp, 4))
    return bp
