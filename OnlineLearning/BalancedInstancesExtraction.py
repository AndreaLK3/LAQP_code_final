import sqlite3

import utilities.MyUtils
import utilities.MyUtils as MyUtils
from sys import exc_info
import logging
from time import time
import utilities.Filenames as F
import csv
import  numpy as np
import pandas as pd
import ast
import json
from gc import collect

import utilities.MyUtils_dbs

NUM_NEGATIVE_CANDIDATES =20

### Simplified version of the code found in NN_Datasets_instances. I am exploiting the fact that the P_FINAL and Q_FINAL
### files are ordered, to register all the matches between Ps and Qs (i.e. questions asked for the products)
def register_matches():

    prods_filepath = F.PRODUCTS_FINAL_TRAIN
    quests_filepath = F.QUESTIONS_FINAL_TRAIN

    MyUtils.init_logging("OnlineLearning_RegisterMatches.log", logging.INFO)
    start = time()
    f = open(F.ONLINE_PQMATCHES, "w"); f.close()#clean outfile between runs
    ids_outfile = open(F.ONLINE_PQMATCHES, "a")
    ids_outfile.write("id_questionsAsked\n")

    prods_filehandler =  open(prods_filepath, "r", newline='')
    quests_filehandler = open(quests_filepath, "r", newline='')
    reader_1 = csv.reader(prods_filehandler, delimiter='_', quotechar='"')
    reader_2 = csv.reader(quests_filehandler, delimiter='_', quotechar='"')

    num_prods_withmatches = 0
    num_products_reviewed = 0
    num_questions_reviewed = 0
    last_prod_id = "x"
    questionsasked_ids_ls = []
    ### init:
    reader_1.__next__(); reader_2.__next__() ; reader_1.__next__(); reader_2.__next__()  #skip headers
    p_ls = reader_1.__next__()
    q_ls = reader_2.__next__()
    prod_t = MyUtils.prodls_tonamedtuple(p_ls, offset=0)
    quest_t = utilities.MyUtils.quest_lstonamedtuple(q_ls, offset=0)
    q_prod = (quest_t.id)[0:10]
    #loop:
    while True:
        try:
            match = False
            while not(match):
                while q_prod > prod_t.id or (len(q_prod) > len(prod_t.id)):
                    logging.debug("%s < %s", prod_t.id , q_prod)
                    p_ls = reader_1.__next__() #advance product
                    num_products_reviewed = num_products_reviewed + 1
                    prod_t = utilities.MyUtils.prodls_tonamedtuple(p_ls, offset=0)

                while q_prod < prod_t.id or (len(q_prod) < len(prod_t.id)):
                    logging.debug("%s > %s", prod_t.id, q_prod)
                    q_ls = reader_2.__next__() #advance question
                    num_questions_reviewed = num_questions_reviewed + 1
                    quest_t = utilities.MyUtils.quest_lstonamedtuple(q_ls, offset=0)
                    q_prod = (quest_t.id)[0:10]

                if q_prod == prod_t.id:
                    match = True
                    #barrier: feature filtering on products and questions; DB lookup:
                    logging.info("Match: product: %s , \t question: %s", prod_t.id, quest_t.id)
                    #positive_qs_ids_file.write(str(quest_t.id) + "\n")#store the question id (positive example)
                    if len(prod_t.id) > 5:
                        if prod_t.id != last_prod_id:
                            if len(last_prod_id) > 5:
                                ids_outfile.write(str(last_prod_id) + "_" + str(questionsasked_ids_ls) + "\n")#write the previous p and qs
                            questionsasked_ids_ls = [] #reset, and then append
                            questionsasked_ids_ls.append(quest_t.id)
                            last_prod_id = prod_t.id
                            num_prods_withmatches = num_prods_withmatches +1 #n: matches = number of products that have questions
                        else:
                            logging.info("***")
                            questionsasked_ids_ls.append(quest_t.id)#same product as previously; only append
                    #on to the next question:
                    q_ls = reader_2.__next__()
                    quest_t = utilities.MyUtils.quest_lstonamedtuple(q_ls, offset=0)
                    q_prod = (quest_t.id)[0:10]

        except StopIteration:
            logging.warning("Exception information: %s", exc_info())
            break
    logging.info("Total number products that have matching questions: %s", num_prods_withmatches)
    logging.info("Products reviewed: %s", num_products_reviewed)
    logging.info("Questions reviewed: %s", num_questions_reviewed)

    end = time()
    logging.info("Time elapsed: %s", round(end - start,4))
    ids_outfile.close()
    prods_filehandler.close()
    quests_filehandler.close()
    #positive_qs_ids_file.close()
    return num_prods_withmatches




####### Assigning to each product N indices (corresponding to the questions that are candidate negative examples),
####### and returning a list of tuples [(0, ['B0009TPLJC']), (9, ['B000HS2L3O', 'B000RDQCDE']), ... ]

def get_negative_indices(chosen_prods_ids_ls):
    logging.info("Extracting the random indices for the questions of the negative examples...")
    #questions_final_filepath = F.QUESTIONS_FINAL_TRAIN

    negative_indices_dict = {}

    #num_of_questions = 0
    #for input_segment in pd.read_csv(questions_final_filepath, chunksize=10**5, sep="_"):
    #    num_of_questions = num_of_questions + len(input_segment)
    questions_final_db = sqlite3.connect(F.QUESTIONS_FINAL_TRAIN_DB)
    q_c = questions_final_db.cursor()
    len_qstables_ls = list(map(lambda result_tpl: result_tpl[0], utilities.MyUtils_dbs.search_in_alltables_db(q_c, "SELECT COUNT(*) FROM ", "")))
    logging.info(len_qstables_ls)
    num_of_questions = sum(len_qstables_ls)
    logging.info(len_qstables_ls)
    questions_final_db.close()

    logging.info("Number of questions in the current dataset: %s", num_of_questions)  # 111171
    logging.info(max(len_qstables_ls))

    random_indices_intable = np.random.choice(a=range(1, max(len_qstables_ls)+1), size=len(chosen_prods_ids_ls)*NUM_NEGATIVE_CANDIDATES,
                                                  replace=True, p=None)
    random_indices_tablenumber = np.random.choice(a=range(1, len(len_qstables_ls)+1), size=len(chosen_prods_ids_ls)*NUM_NEGATIVE_CANDIDATES,
                                                  replace=True, p=None)
    all_random_indices = list(zip(random_indices_tablenumber, random_indices_intable))
    #logging.info(all_random_indices[0:10])
    for i in range(len(chosen_prods_ids_ls)):
        prod_id = chosen_prods_ids_ls[i]
        product_random_indices = all_random_indices[i*NUM_NEGATIVE_CANDIDATES:(i+1)*NUM_NEGATIVE_CANDIDATES]
        #logging.info(product_random_indices)
        if i % (max(len(chosen_prods_ids_ls) // 10, 1)) == 0:
            logging.info("Product: %s / %s", i, len(chosen_prods_ids_ls))
        for rand_tuple in product_random_indices:
            if rand_tuple not in negative_indices_dict:
                negative_indices_dict[rand_tuple] = []
                negative_indices_dict[rand_tuple].append(prod_id)
            else:
                negative_indices_dict[rand_tuple].append(prod_id)
    #logging.info(negative_indices_dict)
    negindices_lts = list(map(lambda lts_tuple: (lts_tuple[0][0], lts_tuple[0][1], lts_tuple[1]) , sorted(negative_indices_dict.items()) ))
    del negative_indices_dict
    logging.info(negindices_lts[0:20])
    lts_df = pd.DataFrame(negindices_lts, columns=["table_number","internal_index","product_ids"])
    lts_df.to_csv(F.ONLINE_NEGINDICES_LTS,sep='_', index_label="")



####### IV : To each product, we assign N question ids, the identifiers of the candidate negative examples
####### Operating: from the reverse dictionary and the questions' final file --> to a sql database

def assign_candidate_negative_examples(chosen_prods_ids_ls):
    logging.info("Assigning preliminary candidate negative examples to products...")
    collect() #clean memory
    start= time()
    #negativeexamples_sourcefile = F.QUESTIONS_FINAL_TRAIN
    negativeexamples_sourcedb = sqlite3.connect(F.QUESTIONS_FINAL_TRAIN_DB)
    neg_exs_c = negativeexamples_sourcedb.cursor()

    db_conn = sqlite3.connect(F.ONLINE_INSTANCEIDS_GLOBAL_DB)
    c = db_conn.cursor()
    c.execute('''CREATE TABLE negativeinstances(p varchar(63),
                                                qs_ls varchar(8191) )                     
                                                ''')

    #initializing with the chosen products' ids
    for the_id in chosen_prods_ids_ls:
        c.execute('''INSERT INTO negativeinstances VALUES (?,?);''', (the_id,'',))
    db_conn.commit()
    c.execute("CREATE INDEX index_prod_ids ON negativeinstances (p);")


    #neg_indices = list(map( lambda neg_ind_t: neg_ind_t[0], neg_indices_lts))
    for in_lts_segment in pd.read_csv(F.ONLINE_NEGINDICES_LTS, sep='_', chunksize=5 * 10**4):
        start = time()
        for lts_t in in_lts_segment.itertuples():
            neg_table = lts_t.table_number
            neg_internalindex = lts_t.internal_index
            quest_id_t = neg_exs_c.execute("SELECT id FROM elements"+str(neg_table)+" WHERE rowid = " + str(neg_internalindex)).fetchone()
            #logging.info(quest_id_t)
            if quest_id_t is not None: #the last table of questions in the final_db may have fewer elements
                quest_id = quest_id_t[0]
                product_ids = (lts_t.product_ids).replace("'", '"')
                for prod_id in json.loads(product_ids):
                    t = (prod_id,)
                    c.execute('SELECT * FROM negativeinstances WHERE p=?', t)
                    row = c.fetchone()
                    negqs_string = row[1]
                    negqs_string = negqs_string + '"' + str(quest_id) + '"' + ","
                    c.execute('''UPDATE negativeinstances
                                    SET qs_ls = ?
                                WHERE p = ? ''', (negqs_string, prod_id))

        logging.info("Segment of %s candidate negative questions (over %s) processed, in %s",
                     5*10**4, len(chosen_prods_ids_ls)*NUM_NEGATIVE_CANDIDATES, round(time()-start,5))

    db_conn.commit()

    logging.info("Completed the assignment of candidate negative examples")


######## V : We exclude the candidate negative examples that are actually positive examples (asked for the product),
########     and then we pick the K candidates that will constitute the actual negative examples for the product.
########     Operating: from PRODSWITHQUESTS_IDS & CandidateNegativeQs.db --> to PRODS_WITH_NOTASKEDQUESTS_IDS


def define_negative_examples():
    MyUtils.init_logging("OnlineLearning-define_negative_examples.log")
    logging.info("Defining negative instances for the dataset...")

    ### Connect with the database to read from: candidate negative examples
    db_conn = sqlite3.connect(F.ONLINE_INSTANCEIDS_GLOBAL_DB)
    c = db_conn.cursor()

    segment_size = 10**4
    for input_segment in pd.read_csv(F.ONLINE_PQMATCHES_FILTERED, sep="_", chunksize=segment_size):
        for id_askedqs_t in input_segment.itertuples():
            prod_id = id_askedqs_t.id

            asked_qs = ast.literal_eval(id_askedqs_t.questionsAsked)
            #logging.info(asked_qs)
            t = (prod_id,)
            row = c.execute('SELECT * FROM negativeinstances WHERE p=?', t).fetchone()
            #logging.info("Row: %s", row)
            if row is None: #i.e. if the product in the file PRODSWITHQUESTS_IDS was excluded from the previous random subsampling
                continue
            candidatenegativeqs_rawstring = row[1]
            candidatenegativeqs_string = "[" + candidatenegativeqs_rawstring[:-1] + "]"

            candidatenegativeqs_string.replace("'",'"')
            candidatenegativeqs_ls = json.loads(candidatenegativeqs_string)
            candidatenegativeqs_ls1 = [q_id for q_id in candidatenegativeqs_ls if q_id not in asked_qs]
            random_indices = sorted(np.random.choice(a=range(len(candidatenegativeqs_ls1)),
                                                     size=min(len(candidatenegativeqs_ls1), len(asked_qs)),
                                                     replace=False, p=None))

            negativeqs_ls = [candidatenegativeqs_ls1[i] for i in random_indices]
            c.execute('''UPDATE negativeinstances
                            SET qs_ls = ?
                            WHERE p = ? ''', (str(negativeqs_ls), prod_id))
        db_conn.commit()
        logging.info("Definition of negative instances for the dataset completed")