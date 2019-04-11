import tensorflow as tf
import logging
import sys
import os
import numpy as np

import  AssociationNN.NN_Datasets_Instances as NDI
import utilities.MyUtils_dbs
import utilities.MyUtils_flags

sys.path.append(os.path.abspath('../Product Features'))
import utilities.Filenames as F
import utilities.MyUtils as MyUtils
import pandas as pd
import ast
import sqlite3
import gensim.models.doc2vec as D2V

tf.logging.set_verbosity(tf.logging.INFO)

# ### The first N/2 instances of the training dataset are positive instances, the remaining N/2 are negative.
# ### When they are created, they reflect the original order (grouped by product, ordered by ASIN)
# ### This function shuffles the two subsets '1' and '0', in order to speed up the extraction of random batches later
def shuffle_training_halves():
    MyUtils.init_logging("Shuffle_training_halves.log", logging.INFO)
    db_conn = sqlite3.connect(F.NN_TRAIN_INSTANCES_DB)
    c = db_conn.cursor()

    f = open(F.NN_TEMP_INSTANCES_DB, "w")
    f.close() #clean outdb
    outdb_conn = sqlite3.connect(F.NN_TEMP_INSTANCES_DB)
    outc = outdb_conn.cursor()
    outc.execute('''CREATE TABLE instances(   p_id varchar(63),
                                              q_id varchar(63),
                                              x varchar(8191),
                                              y tinyint
                                              )''')
    outdb_conn.commit()

    tot_num_of_rows = utilities.MyUtils_dbs.get_nn_dataset_length(utilities.MyUtils_flags.FLAG_TRAIN)
    half_mark = tot_num_of_rows // 2
    ids_pos = np.random.choice(range(1, half_mark+1), half_mark, replace=False)
    # the effect of the previous line is to permute the indices from 0 to half_mark
    ids_neg = np.random.choice(range(half_mark, tot_num_of_rows), half_mark, replace=False)

    for id_pos in ids_pos:
        picked_row = c.execute("SELECT * FROM instances WHERE rowid = " + str(id_pos)).fetchone()
        p_id = picked_row[0]
        q_id = picked_row [1]
        x = picked_row[2]
        y = picked_row[3]
        outc.execute('''INSERT INTO instances VALUES (?,?,?,?);''', (p_id, q_id, str(x), y))
    outdb_conn.commit()
    logging.info("Training set: Positive instances have been shuffled. Proceeding to shuffle negative instances...")
    for id_neg in ids_neg:
        picked_row = c.execute("SELECT * FROM instances WHERE rowid = " + str(id_neg)).fetchone()
        p_id = picked_row[0]
        q_id = picked_row[1]
        x = picked_row[2]
        y = picked_row[3]
        outc.execute('''INSERT INTO instances VALUES (?,?,?,?);''', (p_id, q_id, str(x), y))
    outdb_conn.commit()
    logging.info("Training set: Negative instances have been shuffled.")

    os.rename(src=F.NN_TEMP_INSTANCES_DB, dst=F.NN_TRAIN_INSTANCES_DB)



########  For all the input instances (P,Q) create the numerical encoding & add the appropriate label,
######## (the encoding is gathered by accessing the databases that store it)

def write_part_training_samples(positive_bool, prod_features_flags, quest_feature_flags, outdb_conn, dataset_type):
    MyUtils.init_logging("WritePartTrainingSamples.log", logging.INFO)

    if "train" in dataset_type:
        ps_db_conn = sqlite3.connect(F.PRODS_NUMENCODING_DB_TRAIN)
        qs_db_conn = sqlite3.connect(F.QUESTS_NUMENCODING_DB_TRAIN)
    elif "valid" in dataset_type:
        ps_db_conn = sqlite3.connect(F.PRODS_NUMENCODING_DB_VALID)
        qs_db_conn = sqlite3.connect(F.QUESTS_NUMENCODING_DB_VALID)
    else:  # "test"
        ps_db_conn = sqlite3.connect(F.PRODS_NUMENCODING_DB_TEST)
        qs_db_conn = sqlite3.connect(F.QUESTS_NUMENCODING_DB_TEST)

    ps_db_cursor = ps_db_conn.cursor()
    qs_db_cursor = qs_db_conn.cursor()
    outdb_cursor = outdb_conn.cursor()

    if positive_bool == True:
        instances_file = open(F.PRODSWITHQUESTS_IDS, "r")
    else:
        instances_file = open(F.PRODS_WITH_NOTASKEDQUESTS_IDS, "r")
    segment_size = 10**4
    encoding_size = 0
    for input_segment in pd.read_csv(instances_file, chunksize=segment_size, sep="_"):
        for prodid_questionsids_t in input_segment.itertuples():
            t = (prodid_questionsids_t.id,)
            ps_db_cursor.execute('SELECT * FROM ps_numenc WHERE p_id=?', t)
            row_p = ps_db_cursor.fetchone() #id, 4 flags (in the established order), 4 encodings --> +5 offset
            if row_p is None:
                logging.warning("Warning: product with id %s not found", t)
                continue
            encoding_product = []
            for i in range(len(prod_features_flags)):
                if prod_features_flags[i] == True:
                    encoding_feature = ast.literal_eval(row_p[i+5])
                    encoding_product = encoding_product + encoding_feature
            if positive_bool:
                questions_ids = ast.literal_eval(prodid_questionsids_t.questionsAsked)
                label = 1
            else:
                questions_ids = ast.literal_eval(prodid_questionsids_t.questionsNotAsked)
                label = 0
            for quest_id in questions_ids:
                t = (quest_id,)
                qs_db_cursor.execute('SELECT * FROM qs_numenc WHERE q_id=?', t)
                row_q = qs_db_cursor.fetchone()  # id, 3 flags (in the established order), 3 encodings --> +4 offset
                encoding_question = []
                if row_q is None:
                    logging.warning(row_q)
                    logging.warning("Numerical encoding not found in the database for question with id: %s ; continuing..." , quest_id)
                    continue
                for i in range(len(quest_feature_flags)):
                    if quest_feature_flags[i] == True:
                        encoding_feature = ast.literal_eval(row_q[i + 4])
                        encoding_question = encoding_question + encoding_feature
                if prod_features_flags[1] == True and quest_feature_flags[1] == True: #if question and product desc vectors are present
                    logging.debug("row_p: %s", row_p)
                    logging.debug("row_q: %s", row_q)
                    logging.debug("***")
                    encoding_vectorsproduct = np.multiply(ast.literal_eval(row_p[1+5]),ast.literal_eval(row_q[1+4])).tolist()#add the element-wise product of the two
                else:
                    encoding_vectorsproduct = []
                encoding = encoding_product + encoding_question + encoding_vectorsproduct
                if encoding_size == 0:
                    encoding_size = len(encoding)
                    logging.info("Number of features of the encoding for the NN input : %s", len(encoding))
                    logging.info("Number of product features: %s,  Number of question features: %s", len(encoding_product), len(encoding_question))
                else:
                    if len(encoding) != encoding_size:
                        logging.warning("Product: %s, Question: %s" , prodid_questionsids_t.id, quest_id)
                        logging.warning("Different size of the encoding for the NN input : %s", len(encoding))
                        logging.warning("Number of product features: %s,  Number of question features: %s",
                                     len(encoding_product), len(encoding_question))
                        continue #skip. The encoding length is not the one that should be
                outdb_cursor.execute('''INSERT INTO instances VALUES (?,?,?,?);''', (prodid_questionsids_t.id,quest_id,str(encoding), label))
        outdb_conn.commit()

    ps_db_conn.close()
    qs_db_conn.close()
    instances_file.close()


###### Utility function (for the main function)

def write_training_samples(prod_features_flags, quest_feature_flags, dataset_type):
    MyUtils.init_logging("WriteTrainingSamples.log")
    if "train" in dataset_type:
        instances_db = F.NN_TRAIN_INSTANCES_DB
    elif "valid" in dataset_type:
        instances_db = F.NN_VALID_INSTANCES_DB
    else: #"test"
        instances_db = F.NN_TEST_INSTANCES_DB

    f = open(instances_db, "w"); f.close()#clean outdb between runs
    db_conn = sqlite3.connect(instances_db)
    c = db_conn.cursor()
    c.execute('''CREATE TABLE instances(    p_id varchar(63),
                                            q_id varchar(63),
                                            x varchar(8191),
                                            y tinyint                      
                                            )''')
    db_conn.commit()

    #n: since there are no deletes, I can use the rowid as the 'index', avoiding the need for an autoincrement field

    #first all positive instances, then all negative instances.
    #The training batches are later extracted in such a way that they are random and balanced, anyway
    write_part_training_samples(True, prod_features_flags, quest_feature_flags, db_conn, dataset_type)
    write_part_training_samples(False, prod_features_flags, quest_feature_flags, db_conn, dataset_type)
    db_conn.close()


########## Main function of the module: create the training set for the NN
########## question_features_flags = [questionType, questionVec, kwsVectors]
########## product_features_flags = [titlevec, descvec, mdcategories, kwsVectors]
########## The dataset type should be one of: "train", "valid", "test"
# (The flag use_existing_file should almost always be set to false, unless I am extracting a subset of the current dataset)
def create_dataset(product_features_flags=[True] * 4, question_feature_flags=[True] * 3, set_maxnumproducts=10 ** 6,
                   dataset_type=utilities.MyUtils_flags.FLAG_TRAIN, use_existing_file =False):

    MyUtils.init_logging("CreateTrainingSet.log", logging.INFO)
    num_ps_with_matches = NDI.register_matches(product_features_flags, question_feature_flags, dataset_type, use_existing_file)
    #num_ps_with_matches = 2000 #dummy
    chosen_ids = NDI.pick_prods_ids(num_ps_with_matches, set_maxnumproducts, dataset_type)
    logging.info("Chosen products' ids: %s", chosen_ids)
    negindices_lts = NDI.get_negative_indices(chosen_ids, dataset_type)
    NDI.assign_candidate_negative_examples(chosen_ids, negindices_lts, dataset_type)
    #
    doc2vec_model =  D2V.Doc2Vec.load(F.D2V_MODEL)
    NDI.define_negative_examples(doc2vec_model, dataset_type)
    write_training_samples(product_features_flags, question_feature_flags, dataset_type)
    if dataset_type == utilities.MyUtils_flags.FLAG_TRAIN:
        shuffle_training_halves()










