import ast
import csv
import logging
import os
import sqlite3
import sys
from time import time

import numpy as np
import pandas as pd
from pympler import asizeof as mem

import utilities.MyUtils as MyUtils
import utilities.MyUtils_dbs as MyUtils_dbs
import utilities.Filenames as F
import utilities.MyUtils_flags as MyUtils_flags
import AssociationNN.ExploreSimilarity as ES
import AssociationNN.ProductSimilarity as PS
from shutil import copy

NUM_NEGATIVE_CANDIDATES = 30


######### Utility functions: checking if a product/questions can be included, based on which features we chose
def featurefilter_prod(prod_id, prod_featureflags, dbcursor):
    #has_titlevec, descvec, mdcategories, kwsVectors
    t = (prod_id,)
    dbcursor.execute('SELECT * FROM ps_numenc WHERE p_id=?', t)
    row = dbcursor.fetchone()
    for i in range(len(prod_featureflags)):
        if prod_featureflags[i] == True:
            #check:
            if row[i+1] == 0 :
                return False

    return True

def featurefilter_quest(quest_id, quest_featureflags, dbcursor):
    #has_questionType, questionVec, kwsVectors
    t = (quest_id,)
    dbcursor.execute('SELECT * FROM qs_numenc WHERE q_id=?', t)
    row = dbcursor.fetchone()
    for i in range(len(quest_featureflags)):
        if quest_featureflags[i] == True:
            #check:
            if row[i+1] == 0 :
                return False

    return True



########## I : Registering the products that have questions asked about them

def register_matches(product_featureflags, quest_featureflags, dataset_type, use_existing_file):
    allmatches_filepath = F.PRODSWITHQUESTS_IDS_ALL + dataset_type
    if use_existing_file:
        if os.path.exists(allmatches_filepath):
            if os.path.getsize(allmatches_filepath) > 0:
                logging.info("The P-Q matches for the requested dataset were already found. They are located in the file:%s",
                             allmatches_filepath)
                last_prod_id = "x"
                allmatches_file = open(file=allmatches_filepath, mode="r", newline='')
                reader = csv.reader(allmatches_file, delimiter='_', quotechar='"')
                reader.__next__() #skip header
                count_ps_withmatches = 0
                while True:
                    try:
                        p_ls = reader.__next__()
                        prod_id = p_ls[0]
                        if prod_id != last_prod_id:
                            count_ps_withmatches = count_ps_withmatches+1
                            last_prod_id = prod_id
                    except StopIteration:
                        break
                allmatches_file.close()
                return count_ps_withmatches

    if dataset_type == MyUtils_flags.FLAG_VALID:
        ps_db_filepath= F.PRODS_NUMENCODING_DB_VALID
        qs_db_filepath= F.QUESTS_NUMENCODING_DB_VALID
        prods_filepath = F.PRODUCTS_FINAL_VALID
        quests_filepath = F.QUESTIONS_FINAL_VALID
    elif dataset_type == MyUtils_flags.FLAG_TEST:
        ps_db_filepath= F.PRODS_NUMENCODING_DB_TEST
        qs_db_filepath= F.QUESTS_NUMENCODING_DB_TEST
        prods_filepath = F.PRODUCTS_FINAL_TEST
        quests_filepath = F.QUESTIONS_FINAL_TEST
    else: #"train"
        ps_db_filepath = F.PRODS_NUMENCODING_DB_TRAIN
        qs_db_filepath = F.QUESTS_NUMENCODING_DB_TRAIN
        prods_filepath = F.PRODUCTS_FINAL_TRAIN
        quests_filepath = F.QUESTIONS_FINAL_TRAIN

    MyUtils.init_logging("RegisterMatches.log", logging.INFO)
    start = time()
    f = open(F.PRODSWITHQUESTS_IDS, "w"); f.close()#clean outfile between runs
    ids_outfile = open(F.PRODSWITHQUESTS_IDS, "a")
    ids_outfile.write("id_questionsAsked\n")

    #connecting with the products, to filter them, based on the features we chose to include
    ps_db_conn = sqlite3.connect(ps_db_filepath)
    ps_db_cursor = ps_db_conn.cursor()
    # connecting with the questions, to filter them, based on the features we chose to include
    qs_db_conn = sqlite3.connect(qs_db_filepath)
    qs_db_cursor = qs_db_conn.cursor()

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
    quest_t = MyUtils.quest_lstonamedtuple(q_ls, offset=0)
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
                    prod_t = MyUtils.prodls_tonamedtuple(p_ls, offset=0)

                while q_prod < prod_t.id or (len(q_prod) < len(prod_t.id)):
                    logging.debug("%s > %s", prod_t.id, q_prod)
                    q_ls = reader_2.__next__() #advance question
                    num_questions_reviewed = num_questions_reviewed + 1
                    quest_t = MyUtils.quest_lstonamedtuple(q_ls, offset=0)
                    q_prod = (quest_t.id)[0:10]

                if q_prod == prod_t.id:
                    match = True
                    #barrier: feature filtering on products and questions; DB lookup:
                    if featurefilter_prod(prod_t.id, product_featureflags, ps_db_cursor) == True and \
                       featurefilter_quest(quest_t.id, quest_featureflags, qs_db_cursor) == True:
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
                    else:
                        pass
                    #on to the next question:
                    q_ls = reader_2.__next__()
                    quest_t = MyUtils.quest_lstonamedtuple(q_ls, offset=0)
                    q_prod = (quest_t.id)[0:10]

        except StopIteration:
            exc_info = sys.exc_info()
            logging.warning("Exception information: %s", exc_info)
            break
    logging.info("Total number products that have matching questions: %s", num_prods_withmatches)
    logging.info("Products reviewed: %s", num_products_reviewed)
    logging.info("Questions reviewed: %s", num_questions_reviewed)

    copy(src=F.PRODSWITHQUESTS_IDS, dst=F.PRODSWITHQUESTS_IDS_ALL + dataset_type)

    end = time()
    logging.info("Time elapsed: %s", round(end - start,4))
    ids_outfile.close()
    prods_filehandler.close()
    quests_filehandler.close()
    #positive_qs_ids_file.close()
    return num_prods_withmatches



########## II : picking a number of ids of the 'products with questions'

def pick_prods_ids(num_prods_withqs, trainset_maxproducts, dataset_type):

    f = open(F.PRODSWITHQUESTS_IDS_TEMP,"w") #clean outfile
    f.close()

    if num_prods_withqs <= trainset_maxproducts:
        chosen_prods_ids = pd.read_csv(F.PRODSWITHQUESTS_IDS_ALL + dataset_type, delimiter="_")["id"]  # pick all of
        logging.info("Picking all the %s products with questions to form the dataset", len(chosen_prods_ids))
        return list(chosen_prods_ids)
    else:  # pick trainset_maxhalfcardinality ids, at random
        # logging.info("Branch 2: picking products at random:")
        chosen_indices = sorted(
            np.random.choice(a=range(num_prods_withqs), size=trainset_maxproducts, replace=False, p=None))
        chosen_ps_ls = []
        logging.info("Indices : %s", chosen_indices)
        ps_file = open(file=F.PRODSWITHQUESTS_IDS_ALL + dataset_type, mode="r", newline='')
        ps_reader = csv.reader(ps_file, quotechar='"',  delimiter='_')
        new_ps_file = open(file=F.PRODSWITHQUESTS_IDS_TEMP, mode="a", newline='')
        new_ps_writer = csv.writer(new_ps_file, delimiter='_')
        headers = ps_reader.__next__()
        new_ps_writer.writerow(headers)
        current_index = 0
        while True:
            try:
                line = ps_reader.__next__()
                p_id = line[0]
                current_index = current_index + 1
                if current_index in chosen_indices:
                    chosen_ps_ls.append(p_id)
                    new_ps_writer.writerow(line)
                    #logging.info("Included product:%s", p_id)
            except StopIteration:
                logging.info("Subset of %s products picked. Iteration stopped at: %s", trainset_maxproducts, current_index)
                #chosen_prods_ids = pd.DataFrame(chosen_ps_ls, columns=["id"])  # end
                break
    #logging.info(list(chosen_prods_ids))
    os.rename(src=F.PRODSWITHQUESTS_IDS_TEMP, dst=F.PRODSWITHQUESTS_IDS)


    return chosen_ps_ls



####### III : Assigning to each product N indices (corresponding to the questions that are candidate negative examples),
#######       and returning a list of tuples [(0, ['B0009TPLJC']), (9, ['B000HS2L3O', 'B000RDQCDE']), ... ]

def get_negative_indices(chosen_prods_ids_ls, dataset_type):

    if dataset_type == MyUtils_flags.FLAG_TRAIN:
        questions_final_filepath = F.QUESTIONS_FINAL_TRAIN
    if dataset_type == MyUtils_flags.FLAG_VALID:
        questions_final_filepath = F.QUESTIONS_FINAL_VALID
    if dataset_type == MyUtils_flags.FLAG_TEST:
        questions_final_filepath = F.QUESTIONS_FINAL_TEST

    negative_indices_dict = {}

    num_of_questions = 0
    for input_segment in pd.read_csv(questions_final_filepath, chunksize=2 ** 10, sep="_"):
        num_of_questions = num_of_questions + len(input_segment)
    logging.info("Number of questions in the current dataset: %s", num_of_questions)  # 111171

    for prod_id in chosen_prods_ids_ls:
        #logging.info("Debug: %s", prod_id)
        random_indices = np.random.choice(a=range(num_of_questions), size=NUM_NEGATIVE_CANDIDATES, replace=False, p=None)
        for rand_index in random_indices:
            if rand_index not in negative_indices_dict:
                negative_indices_dict[rand_index] = []
                negative_indices_dict[rand_index].append(prod_id)
            else:
                negative_indices_dict[rand_index].append(prod_id)

    #with open(F.PRODS_NEGATIVEINDICES, "wb") as neg_indices_dict_file: ##save the structure to check it
    #    pickle.dump(obj=negative_indices_dict, file=neg_indices_dict_file, )
    #Python dictionaries are not ordered --> need to use... a list of
    #logging.info(negative_indices_dict)
    logging.info("Size in memory of the dictionary of random indices: %s KB",mem.asizeof(negative_indices_dict) // 2**10)
    negindices_lts = sorted(negative_indices_dict.items())

    return negindices_lts

####### IV : To each product, we assign N question ids, the identifiers of the candidate negative examples
####### Operating: from the reverse dictionary and the questions' final file --> to a sql database

def assign_candidate_negative_examples(chosen_prods_ids_ls, neg_indices_lts, dataset_type):
    if dataset_type == MyUtils_flags.FLAG_TRAIN:
        negativeexamples_sourcefile = F.QUESTIONS_FINAL_TRAIN
    if dataset_type == MyUtils_flags.FLAG_VALID:
        negativeexamples_sourcefile = F.QUESTIONS_FINAL_VALID
    if dataset_type == MyUtils_flags.FLAG_TEST:
        negativeexamples_sourcefile = F.QUESTIONS_FINAL_TEST


    start = time()
    f = open(F.CANDIDATE_NEGQS_DB, mode="w");
    f.close()  # clean between runs
    db_conn = sqlite3.connect(F.CANDIDATE_NEGQS_DB)
    c = db_conn.cursor()
    c.execute('''CREATE TABLE prodnegatives(prod_id varchar(63) NOT NULL,
                                        cand_qneg_ids varchar(2047) DEFAULT '',                      
                                        PRIMARY KEY (prod_id)
                                        )''')
    #initializing the  with the chosen products' ids
    #logging.info(chosen_prods_ids_ls)
    for the_id in chosen_prods_ids_ls:
        #logging.info("Inserting id: %s", the_id)
        c.execute('''INSERT INTO prodnegatives VALUES (?,?);''', (the_id, ''))
    db_conn.commit()

    # The loop:
    segment_size = 10**4
    current_index = 0
    negindices_counter = 0
    num_of_neg_qs = len(neg_indices_lts)
    with open(negativeexamples_sourcefile, "r") as questions_final_file:
        for segment in pd.read_csv(questions_final_file, chunksize=segment_size, sep="_"):
            start_segment = time()
            for quest_t in segment.itertuples():
                if negindices_counter < num_of_neg_qs: #until there are still matches with the negative qs to find
                    if current_index == neg_indices_lts[negindices_counter][0]:
                        #negativeqs_ids_file.write(str(quest_t.id + "\n"))
                        #question index matching a question that was chosen as a random negative example
                        for prod_id in neg_indices_lts[negindices_counter][1]:
                            t = (prod_id,)
                            c.execute('SELECT * FROM prodnegatives WHERE prod_id=?', t)
                            row = c.fetchone()
                            negqs_string = row[1]
                            negqs_string = negqs_string + '"' + str(quest_t.id) + '"' + ","

                            c.execute('''UPDATE prodnegatives
                                            SET cand_qneg_ids = ?
                                        WHERE prod_id = ? ''', (negqs_string, prod_id))
                        negindices_counter = negindices_counter +1
                        #logging.info("Row: %s", row)
                    current_index = current_index+1
            db_conn.commit()
            end_segment = time()
            #logging.info("Assigning the ids of not-asked questions to products... segment time: %s seconds", round(end_segment - start_segment,4))

    db_conn.commit()
    #negativeqs_ids_file.close()
    end = time()
    logging.info("Completed the assignment of negative examples; time elapsed : %s seconds", round(end-start,3))


######## V : We exclude the candidate negative examples that are actually positive examples (asked for the product),
########     and then we pick the K candidates that will constitute the actual negative examples for the product.
########     Operating: from PRODSWITHQUESTS_IDS & CandidateNegativeQs.db --> to PRODS_WITH_NOTASKEDQUESTS_IDS


def define_negative_examples(doc2vec_model, dataset_typeflag):
    MyUtils.init_logging("NN_Dataset_Instances-define_negative_examples.log", logging.INFO)

    f = open(F.PRODS_WITH_NOTASKEDQUESTS_IDS, "w"); f.close()
    prodsnegativeqs_outfile = open(F.PRODS_WITH_NOTASKEDQUESTS_IDS, "a")
    prodsnegativeqs_outfile.write("id_questionsNotAsked\n")

    ### Connect with the database to read from: candidate negative examples
    db_conn = sqlite3.connect(F.CANDIDATE_NEGQS_DB)
    c = db_conn.cursor()

    ### IF we are working to create the training dataset,
    ### then we before allowing a question Q asked for P2 to be a negative example for P1,
    ### we check the similarity between P1 and P2 (it must not be too high)
    if dataset_typeflag == MyUtils_flags.FLAG_TRAIN:

        ### Determining the maximum allowed similarity between products. Creates the similarity db if it does not exist
        if os.path.exists(F.SIMILARITY_PRODUCTS_DB) == True:
            p_sim_breakpoint = ES.get_products_similarity_breakpoint(fraction=0.97)
        else:
            p_sim_breakpoint = ES.explore_products_similarity(N=500, fraction=0.97)

        ### Connect with the databases of product and questions representations, to be able to pick the products P1 and P2
        product_reps_dbconn = sqlite3.connect(F.PRODUCTS_FINAL_TRAIN_DB)
        product_reps_c = product_reps_dbconn.cursor()

    segment_size = 10**4
    for input_segment in pd.read_csv(F.PRODSWITHQUESTS_IDS, sep="_", chunksize=segment_size):
        for id_askedqs_t in input_segment.itertuples():
            prod_id = id_askedqs_t.id
            #logging.debug("Reading from F.PRODSWITHQUESTS_IDS, the product.id is: %s", prod_id)
            asked_qs = ast.literal_eval(id_askedqs_t.questionsAsked)
            t = (prod_id,)
            c.execute('SELECT * FROM prodnegatives WHERE prod_id=?', t)
            row = c.fetchone()
            if row is None: #i.e. if the product in the file PRODSWITHQUESTS_IDS was excluded from the previous random subsampling
                continue
            candidatenegativeqs_rawstring = row[1]
            candidatenegativeqs_string = "[" + candidatenegativeqs_rawstring[:-1] + "]"

            candidatenegativeqs_ls = ast.literal_eval(candidatenegativeqs_string)
            candidatenegativeqs_ls1 = [q_id for q_id in candidatenegativeqs_ls if q_id not in asked_qs]

            if dataset_typeflag == MyUtils_flags.FLAG_TRAIN:
                p1_row = MyUtils_dbs.search_in_alltables_db(dbcursor=product_reps_c, query_pretext="SELECT * FROM",
                                                                      query_aftertext=" WHERE id='"+str(prod_id)+"'")[0]
                candidatenegativeqs_asins = list(map(lambda q_id : q_id[0:10] , candidatenegativeqs_ls1))

                p2_rows = MyUtils_dbs.search_in_alltables_db(dbcursor=product_reps_c, query_pretext="SELECT * FROM",
                                                                       query_aftertext="WHERE id IN "+ str(tuple(candidatenegativeqs_asins)))
                qids_and_p2rows = list(zip(candidatenegativeqs_ls1, p2_rows))


                for q_id, p2_row in qids_and_p2rows:
                    #logging.debug("p1_row : %s", p1_row)
                    if p2_row is not None and len(p2_row)>0:
                        #there are questions without corresponding products, in which case no similarity check is to be done

                        p1_tuple = MyUtils.prodls_tonamedtuple(p1_row)#[1:]?
                        p2_tuple = MyUtils.prodls_tonamedtuple(p2_row)
                        p1_p2_sim, _simparts = PS.compute_2products_similarity_singleprocess(prod1_tuple=p1_tuple, prod2_tuple=p2_tuple,
                                                                          d2v_model=doc2vec_model)
                        if p1_p2_sim > p_sim_breakpoint:
                            candidatenegativeqs_ls1.remove(q_id)
                            logging.info("Removing question from the candidate negative examples, " +
                                         "because the similarity between %s and %s is > %s",
                                          prod_id, p2_tuple.id, p_sim_breakpoint)
                logging.info("Choosing negative examples: P-to-p similarity checks done for product: %s", prod_id)



            random_indices = sorted(np.random.choice(a=range(len(candidatenegativeqs_ls1)),
                                                     size=min(len(candidatenegativeqs_ls1),len(asked_qs)), replace=False, p=None))
            #logging.info(candidatenegativeqs_ls1)
            negativeqs_ls = [ candidatenegativeqs_ls1[i] for i in random_indices]
            #logging.info(negativeqs_ls)
            prodsnegativeqs_outfile.write(prod_id + "_" + str(negativeqs_ls) + "\n")

    prodsnegativeqs_outfile.close()