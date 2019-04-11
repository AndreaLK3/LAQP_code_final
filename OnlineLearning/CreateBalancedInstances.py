import sqlite3
import numpy as np
import utilities.Filenames as F
import utilities.MyUtils
import utilities.MyUtils as MyUtils
import logging
import OnlineLearning.BalancedInstancesExtraction as IE
import pandas as pd
import ast
from gc import collect
from os import rename
import json

###### Postprocessing the out_db : 'unpacking' the lists of questions in the (p_id, q_ls) tuples,
###### and finally shuffling internally the two parts of the balanced set (positive and negative instances)
import utilities.MyUtils_dbs
import utilities.MyUtils_strings


def unpack_question_lists():
    MyUtils.init_logging("unpack_question_lists.log", logging.INFO)
    db_conn = sqlite3.connect(F.ONLINE_INSTANCEIDS_GLOBAL_DB)
    c = db_conn.cursor()

    f = open(F.ONLINE_TEMP_DB, "w")
    f.close()  # clean outdb
    outdb_conn = sqlite3.connect(F.ONLINE_TEMP_DB)
    outc = outdb_conn.cursor()
    outc.execute('''CREATE TABLE positiveinstances(p varchar(63),
                                            q varchar(63) )  ''')
    outc.execute('''CREATE TABLE negativeinstances(p varchar(63),
                                                q varchar(63) )  ''')
    outdb_conn.commit()

    p_qs_lts = c.execute("SELECT * FROM positiveinstances").fetchall()
    logging.info("(Positive examples): Unpacking the lists of questions from %s products", len(p_qs_lts))
    for p_qs_t in p_qs_lts:
        p = p_qs_t[0]
        qs_str = p_qs_t[1]
        qs_ls = json.loads(qs_str.replace("'", '"'))
        for q in qs_ls:
            outc.execute('''INSERT INTO positiveinstances VALUES (?,?);''', (str(p), str(q)) )
    outdb_conn.commit()

    p_qs_lts = c.execute("SELECT * FROM negativeinstances").fetchall()
    logging.info("(Negative examples): Unpacking the lists of questions from %s products", len(p_qs_lts))
    for p_qs_t in p_qs_lts:
        p = p_qs_t[0]
        qs_str = p_qs_t[1]
        qs_ls = json.loads(qs_str.replace("'", '"'))
        for q in qs_ls:
            outc.execute('''INSERT INTO negativeinstances VALUES (?,?);''', (str(p), str(q)) )
    outdb_conn.commit()
    rename(src=F.ONLINE_TEMP_DB, dst=F.ONLINE_INSTANCEIDS_GLOBAL_DB)


#### This function shuffles the two subsets 'positiveinstances' and 'negativeinstances', that are stored in 2 tables
def shuffle_balancedinstances_db():
    MyUtils.init_logging("Shuffle_training_halves.log", logging.INFO)
    db_conn = sqlite3.connect(F.ONLINE_INSTANCEIDS_GLOBAL_DB)
    c = db_conn.cursor()

    f = open(F.ONLINE_TEMP_DB, "w")
    f.close()  # clean outdb
    outdb_conn = sqlite3.connect(F.ONLINE_TEMP_DB)
    outc = outdb_conn.cursor()
    outc.execute('''CREATE TABLE positiveinstances(p varchar(63),
                                            qs_ls varchar(8191) )  ''')
    outc.execute('''CREATE TABLE negativeinstances(p varchar(63),
                                                qs_ls varchar(8191) )  ''')
    outdb_conn.commit()

    num_pos_instances = c.execute("SELECT COUNT(*) FROM positiveinstances").fetchone()[0]
    num_neg_instances = c.execute("SELECT COUNT(*) FROM negativeinstances").fetchone()[0]
    ids_pos = np.random.choice(a=range(1, num_pos_instances + 1), size=num_pos_instances, replace=False)
    ids_neg = np.random.choice(a=range(1, num_neg_instances +1),size=num_neg_instances, replace=False)

    for id_pos in ids_pos:
        picked_row = c.execute("SELECT * FROM positiveinstances WHERE rowid = " + str(id_pos)).fetchone()
        p = picked_row[0]
        qs = picked_row[1]
        outc.execute('''INSERT INTO positiveinstances VALUES (?,?);''', (str(p), qs))
    outdb_conn.commit()
    logging.info("Training set: Positive instances have been shuffled. Proceeding to shuffle negative instances...")
    for id_neg in ids_neg:
        picked_row = c.execute("SELECT * FROM negativeinstances WHERE rowid = " + str(id_neg)).fetchone()
        p = picked_row[0]
        qs = picked_row[1]
        outc.execute('''INSERT INTO negativeinstances VALUES (?,?);''', (str(p), qs))
    outdb_conn.commit()
    logging.info("Training set: Negative instances have been shuffled.")

    rename(src=F.ONLINE_TEMP_DB, dst=F.ONLINE_INSTANCEIDS_GLOBAL_DB)



######## Determining if products and questions have all the features. It is necessary, since we should be able to
######## apply all the different policies a_1,...,a_K on an instance (x,y)
def product_has_allfeatures(ps_db_c, prod_id_str):
    product_row = utilities.MyUtils_dbs.search_in_alltables_db(ps_db_c, "SELECT * FROM ", "WHERE id = '" + prod_id_str + "'")[0]
    prod_tuple = utilities.MyUtils.prodls_tonamedtuple(product_row)#[1:]
    if prod_tuple.descvec != "NODESCVEC" and prod_tuple.titlevec != "NOTITLEVEC"\
            and prod_tuple.kwsVectors != "NOKWSVECTORS" and len(
        utilities.MyUtils_strings.fromlls_toarrays(prod_tuple.kwsVectors)) > 0 \
            and len(ast.literal_eval(prod_tuple.mdcategories)) > 0: #eval(prod_tuple.kwsVectors, {'__builtins__':{}})
        return True
    else:
        return False

def allquestions_have_allfeatures(qs_db_c, q_ids_str):
    q_ids = eval(q_ids_str, {'__builtins__': {}})
    for q_id in q_ids:
        question_row = utilities.MyUtils_dbs.search_in_alltables_db(qs_db_c, "SELECT * FROM ", "WHERE id = '" + str(q_id) + "'")[0]
        q_tuple = utilities.MyUtils.quest_lstonamedtuple(question_row)#[1:]
        if q_tuple.questionVec != "NOQUESTVEC" and q_tuple.kwsVectors != "NOKWSVECTORS" and \
                len(utilities.MyUtils_strings.fromlls_toarrays(q_tuple.kwsVectors)) > 0:
            return True
        else:
            return False



def filter_matches_allfeatures():
    MyUtils.init_logging("OnlineLearning_DefineInstances.log")
    ps_db = sqlite3.connect(F.PRODUCTS_FINAL_TRAIN_DB)
    qs_db = sqlite3.connect(F.QUESTIONS_FINAL_TRAIN_DB)
    ps_db_c = ps_db.cursor()
    qs_db_c = qs_db.cursor()

    pqs_allmatches_file = open(F.ONLINE_PQMATCHES, "r")
    pqs_allmatches_df = pd.read_csv(pqs_allmatches_file, sep="_")
    filtered_matches = []

    for pqs_t in pqs_allmatches_df.itertuples():
        condition_p = product_has_allfeatures(ps_db_c, pqs_t.id)
        logging.info(pqs_t.id)
        condition_q = allquestions_have_allfeatures(qs_db_c, pqs_t.questionsAsked)
        if condition_p and condition_q:
            filtered_matches.append(pqs_t)
    pqs_allmatches_df = pd.DataFrame(filtered_matches)
    pqs_filteredmatches_file = open(F.ONLINE_PQMATCHES_FILTERED, "w")
    pqs_allmatches_df.to_csv(pqs_filteredmatches_file, sep="_")

    logging.info("Number of products with matching questions, that have valid values for all the features: %s", len(filtered_matches))
    pqs_allmatches_file.close()
    pqs_filteredmatches_file.close()
    del pqs_allmatches_df
#########
#########




########## Main function for the module

def create_onlinelearning_traininstances():
    MyUtils.init_logging("create_onlinelearning_traininstances.log")

    file = open(F.ONLINE_INSTANCEIDS_GLOBAL_DB, "w"); file.close()
    out_db = sqlite3.connect(F.ONLINE_INSTANCEIDS_GLOBAL_DB)
    out_c = out_db.cursor()
    out_c.execute('''CREATE TABLE positiveinstances(    p varchar(63),
                                                        qs_ls varchar(8191)                      
                                                    )''')
    out_db.commit()

    #IE.register_matches()
    #filter_matches_allfeatures()

    pqs_filteredmatches_df = pd.read_csv(F.ONLINE_PQMATCHES_FILTERED, sep="_")
    prods_ids_ls = []
    for pid_qs_t in pqs_filteredmatches_df.itertuples():
        out_c.execute("INSERT INTO positiveinstances VALUES (?,?)", (pid_qs_t.id, pid_qs_t.questionsAsked))
        prods_ids_ls.append(pid_qs_t.id)
    out_db.commit()
    logging.info("Creating balanced training instances for Online learning: Positive instances determined...")
    del pqs_filteredmatches_df
    collect()
    IE.get_negative_indices(prods_ids_ls)
    collect()
    IE.assign_candidate_negative_examples(prods_ids_ls)
    collect()
    IE.define_negative_examples()
    unpack_question_lists()
    shuffle_balancedinstances_db()
