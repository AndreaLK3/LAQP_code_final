import utilities.Filenames as F
import sqlite3
import utilities.MyUtils_dbs as MyUtils_dbs
import logging
import utilities.MyUtils as MyUtils
import utilities.MyUtils_strings as MyUtils_strings
import numpy as np
import Ranking.ComputePQdistance as CD
import os
import Ranking.GetCandidatesNN as GCN

########################
###### Ranking the candidates for a product
def sort_product_candidates(product_tuple, quests_ids, testprods_rep_c, testquests_rep_c):

    distance_list = [] #list of tuples, (p_id, q_id, distance), sorted on tpl[2]

    for quest_id in quests_ids:
        question_representation = MyUtils_dbs.search_in_alltables_db(testquests_rep_c, "SELECT * FROM ",
                                                             "WHERE id = '" + str(quest_id) + "'")[0]
        logging.debug("Question representation: %s", question_representation)
        question_tuple = MyUtils.quest_lstonamedtuple(question_representation, offset=1)
        pq_dist = CD.compute_dist_pq(product_tuple, question_tuple)
        distance_list.append((product_tuple.id, quest_id, pq_dist))

    distance_list_sorted = sorted(distance_list, key=lambda tpl : tpl[2])
    return distance_list_sorted

########################


########################
######## Given the candidates from the neural network, rank them
def sort_candidates(candidates_db_path, ranked_candidates_outdb_path, prod_reps_dbpath, quest_reps_dbpath):
    MyUtils.init_logging("Rank_candidates_nn.log")
    ### Connecting to the databases: candidates, test products, test questions
    candidates_nn_db = sqlite3.connect(candidates_db_path)
    cands_db_c = candidates_nn_db.cursor()

    testprods_rep_c = sqlite3.connect(prod_reps_dbpath).cursor()
    testquests_rep_c = sqlite3.connect(quest_reps_dbpath).cursor()

    f = open(ranked_candidates_outdb_path, "w"); f.close()
    outdb = sqlite3.connect(ranked_candidates_outdb_path)
    outdb_c = outdb.cursor()
    outdb_c.execute('''CREATE TABLE candidates(    p_id varchar(63),
                                                   q_id varchar(63),
                                                   distance int        
                                            )''')
    ###

    test_products_ids = cands_db_c.execute("SELECT DISTINCT p_id FROM candidates").fetchall()
    logging.info(test_products_ids[0])
    #logging.debug(test_products_ids)
    for tpl_pid in test_products_ids:
        pid = tpl_pid[0]
        product_representation = MyUtils_dbs.search_in_alltables_db(testprods_rep_c, "SELECT * FROM ",
                                                                    "WHERE id = '" + str(pid) + "'")[0]
        product_tuple = MyUtils.prodls_tonamedtuple(product_representation, offset=1)
        quests_ids = list(map ( lambda results_tpl : results_tpl[0], cands_db_c.execute("SELECT q_id FROM candidates WHERE p_id = ?", tpl_pid).fetchall()))
        logging.debug(quests_ids)
        product_qs_sorted = sort_product_candidates(product_tuple, quests_ids, testprods_rep_c, testquests_rep_c)
        outdb.executemany("INSERT INTO candidates VALUES (?,?,?)", product_qs_sorted)
    outdb.commit()

########################
######## Adding the text of products and questions
def attach_text_to_candidates(ranked_candidates_dbpath, prods_initial_dbpath, quests_initial_dbpath, prod_reps_dbpath, quest_reps_dbpath, final_outdb_path):
    MyUtils.init_logging("Attach_text_to_candidates.log")

    candidates_nn_db = sqlite3.connect(ranked_candidates_dbpath)
    cands_db_c = candidates_nn_db.cursor()
    f = open(F.RANKING_TEMP_DB, 'w'); f.close()
    temp_db = sqlite3.connect(F.RANKING_TEMP_DB)
    temp_db_c = temp_db.cursor()
    testprods_initial_c = sqlite3.connect(prods_initial_dbpath).cursor()
    testquests_initial_c = sqlite3.connect(quests_initial_dbpath).cursor()
    testprods_rep_c = sqlite3.connect(prod_reps_dbpath).cursor()
    testquests_rep_c = sqlite3.connect(quest_reps_dbpath).cursor()

    temp_db_c.execute('''CREATE TABLE candidates(  p_id varchar(63),
                                                   q_id varchar(63),
                                                   distance int,
                                                   p_titletext varchar(1023),
                                                   p_descriptiontext varchar(8191),
                                                   p_categorytext varchar (4095),
                                                   q_text varchar (8191)         
                                            )''')

    num_of_candidates = MyUtils_dbs.get_tot_num_rows_db(cands_db_c)
    logging.info(num_of_candidates)
    counter_questionsameid = 0
    last_prod_id = 'x'

    for rowindex in range(1, num_of_candidates + 1):
        row = cands_db_c.execute("SELECT * FROM candidates WHERE rowid = ?", (rowindex,)).fetchone()
        #logging.info("info: %s", row)
        prod_id = row[0]
        quest_id = row[1]
        distance = row[2]

        if last_prod_id != prod_id:
            product_titleinfo,product_descinfo, product_categinfo = \
                MyUtils_dbs.search_in_alltables_db(testprods_initial_c, "SELECT title, description, categories FROM",
                                                                  "WHERE asin = '" + str(prod_id) + "'")[0]
            product_representation = MyUtils_dbs.search_in_alltables_db(testprods_rep_c, "SELECT * FROM ",
                                                                        "WHERE id = '" + str(prod_id) + "'")[0]
            prod_tpl = MyUtils.prodls_tonamedtuple(product_representation, offset=1)

            counter_questionsameid = 0

        ###get question's unixTime
        if len(quest_id)< 21: #format : @nan0
            base_endpoint = 14
            question_unixTime = str(quest_id[11:base_endpoint])
        else:
            base_endpoint = 23
            question_unixTime = str(quest_id[11:base_endpoint])
        logging.debug("Question unixTime: %s", question_unixTime)

        if base_endpoint == 23: #if we have a valid unixTime specification

            possible_questions_text = MyUtils_dbs.search_in_alltables_db(testquests_initial_c, "SELECT question FROM",
                                                                  "WHERE asin = '" + str(quest_id[0:10]) + "'"
                                                              + " AND unixTime LIKE '" + question_unixTime + "%'")
        else: #if we have NULL in the unixTime field
            possible_questions_text = MyUtils_dbs.search_in_alltables_db(testquests_initial_c, "SELECT question FROM",
                                                                         "WHERE asin = '" + str(quest_id[0:10]) + "'"
                                                                         + " AND unixTime IS NULL")
        base_q_id = str(quest_id[0:23])
        possible_questions_reps = MyUtils_dbs.search_in_alltables_db(testquests_rep_c, "SELECT * FROM ",
                                                             "WHERE id LIKE '" + str(base_q_id) + "%'")
        logging.debug("possible_questions_reps: %s", possible_questions_reps)
        logging.debug("possible_questions_text:%s", possible_questions_text)

        if len(possible_questions_text) > 1:
            possible_questions_tuples = list(map ( lambda q_ls : MyUtils.quest_lstonamedtuple(q_ls, offset=1), possible_questions_reps))
            possible_questions_distances = list(map (lambda q_tpl : CD.compute_dist_pq(prod_tpl, q_tpl) , possible_questions_tuples))

            qs_dist_lts = list(zip(possible_questions_tuples, possible_questions_distances))
            qs_dist_lts_sorted = sorted( qs_dist_lts, key=lambda tpl : tpl[1])
            #logging.info("sorted question tuples: %s", qs_dist_lts_sorted)
            question_textinfo = possible_questions_text[counter_questionsameid][0]
            counter_questionsameid= counter_questionsameid+1
        else:
            question_textinfo = possible_questions_text[0][0]
        logging.debug("question_textinfo: %s", question_textinfo)

        temp_db_c.execute("INSERT INTO candidates VALUES (?,?,?,?,?,?,?)", (prod_id, quest_id, distance,
                                                                          product_titleinfo, product_descinfo, product_categinfo, question_textinfo))
        logging.debug("***")

    temp_db.commit()
    os.rename(F.RANKING_TEMP_DB , final_outdb_path)

########################

###### Main function
def rank_candidates_nn():
    GCN.apply_NN_on_testset()
    sort_candidates(F.CANDIDATES_NN_DB, F.CANDIDATES_NN_DB_RANKED, F.PRODUCTS_FINAL_TEST_DB, F.QUESTIONS_FINAL_TEST_DB)
    attach_text_to_candidates(F.CANDIDATES_NN_DB_RANKED,  F.TEST_MD_DF_DB, F.QA_TEST_DB,
                              F.PRODUCTS_FINAL_TEST_DB, F.QUESTIONS_FINAL_TEST_DB, F.CANDIDATES_NN_DB_COMPLETE)


def rank_candidates_ol(balanced):
    if balanced:
        sort_candidates(F.CANDIDATES_ONLINE_BALANCED_DB, F.CANDIDATES_ONLINE_DB_RANKED, F.PRODUCTS_FINAL_TRAIN_DB, F.QUESTIONS_FINAL_TRAIN_DB)
        attach_text_to_candidates(F.CANDIDATES_ONLINE_DB_RANKED,  F.TRAIN_MD_DF_DB, F.QA_TRAIN_DB,
                                  F.PRODUCTS_FINAL_TRAIN_DB, F.QUESTIONS_FINAL_TRAIN_DB, F.CANDIDATES_ONLINE_BALANCED_COMPLETE)
    else:
        pass



# ##### Examining the norm of Doc2Vec vectors.
# ##### We find that it changes between vectors, it is not normalized to a value (eg. 1).
# ##### Therefore, Euclidean Distance is still a possible measure of vector similarity
# def test_norm_doc2vec_vectors():
#     MyUtils.init_logging("test_norm_doc2vec_vectors.log")
#
#     candidates_nn_db = sqlite3.connect(F.CANDIDATES_NN_DB)
#     cands_db_c = candidates_nn_db.cursor()
#
#     testprods_db = sqlite3.connect(F.PRODUCTS_FINAL_TEST_DB)
#     testprods_db_c = testprods_db.cursor()
#     testquests_db = sqlite3.connect(F.QUESTIONS_FINAL_TEST_DB)
#     testquests_db_c = testquests_db.cursor()
#
#     #n: the candidates in the db are grouped by product (i.e. ordered by product asin). 2 columns: p_id, q_id
#     num_of_candidates = MyUtils_dbs.get_tot_num_rows_db(cands_db_c)
#
#     last_prod_id = 'x'
#     for rowindex in range(1,num_of_candidates+1):
#         row = cands_db_c.execute("SELECT * FROM candidates WHERE rowid = ?", (rowindex,)).fetchone()
#         #logging.debug("row: %s", row)
#         prod_id = row[1]
#         quest_id = row[2]
#         if last_prod_id != prod_id:
#             product_representation = MyUtils_dbs.search_in_alltables_db(testprods_db_c, "SELECT * FROM ",
#                                                                         "WHERE id = '" + str(prod_id) + "'")[0]
#             logging.debug("Product representation: %s", product_representation)
#             product_tuple = MyUtils.prodls_tonamedtuple(product_representation, offset=1)
#             last_prod_id = prod_id
#             logging.info("***New product; product id: %s", prod_id)
#
#         question_representation =  MyUtils_dbs.search_in_alltables_db(testquests_db_c, "SELECT * FROM ", "WHERE id = '"+str(quest_id) + "'")[0]
#         logging.debug("Question representation: %s",question_representation)
#         question_tuple = MyUtils.quest_lstonamedtuple(question_representation, offset=1)
#         logging.info("Question id: %s", quest_id)
#
#         #P title vector, P description vector, Q text vector
#         #applied only to those products that have both title and description
#         try:
#             P_titlevec = MyUtils_strings.fromstring_toarray(product_tuple.descvec)
#             P_descvec = MyUtils_strings.fromstring_toarray(product_tuple.titlevec)
#             Q_questionvec = MyUtils_strings.fromstring_toarray(question_tuple.questionVec)
#             norm_P_titlevec = np.linalg.norm(P_titlevec)
#             norm_P_descvec = np.linalg.norm(P_descvec)
#             norm_Q_questionvec = np.linalg.norm(Q_questionvec)
#             logging.info("Norms of: product title vector = %s; product description vector = %s; question text vector = %s; ",
#                          round(norm_P_titlevec,6) , round(norm_P_descvec,6), round(norm_Q_questionvec,6))
#         except NameError:
#             pass#some feature is missing. Proceed to next (P,Q)








