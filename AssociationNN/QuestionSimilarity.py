import sys
import os

from utilities.MyUtils import quest_lstonamedtuple

sys.path.append(os.path.abspath('../Product Features'))
#print(sys.path)
import utilities.MyUtils as MyUtils
import utilities.Filenames as F
import logging
import sqlite3
from time import time
import AssociationNN.ProductSimilarity as PS
import csv
from itertools import islice
import matplotlib.pyplot as plt

def compute_2questions_similarity_singleprocess(quest1_t, quest2_t):
    if quest1_t.questionVec == "NOQUESTVEC" or quest2_t.questionVec == "NOQUESTVEC":
        return -2
    else:
        qvec_cosine_sim = PS.compute_cosinesim_2arraystrings(quest1_t.questionVec, quest2_t.questionVec)
        logging.info("Qvec cosine sim: %s", round(qvec_cosine_sim,5))
        kws_sim = PS.get_kws_similarity(quest1_t.kwsVectors, quest2_t.kwsVectors)
        logging.info("Kws cosine sim: %s", round(kws_sim, 5))

    if quest1_t.questionType == quest2_t.questionType:  # 'open-ended', 'yes/no'
        same_qtype = 1
    else:
        same_qtype = -1
    logging.info("Same qType sim: %s", round(same_qtype, 5))
    sim_value = 0.65 * qvec_cosine_sim + 0.25 * kws_sim + 0.1 * same_qtype
    transposed_sim_value = (sim_value + 1) / 2 #from [-1,1] to [0,1]
    sim_parts = (qvec_cosine_sim, kws_sim, same_qtype) #for exploration purposes
    return (transposed_sim_value, sim_parts)





#### testing the (excessively long) loop to compute all similarities (Qi, Qj)
#### after having processed threshold_elem1/2 elements as Qi/j, stops;
#### the objective is to show the range&frequency results of some features
def test_csv(threshold_elem1 = 20, threshold_elem2 = 100):
    MyUtils.init_logging("QuestionSimilarity.log")
    f = open("QuestionSimilarity.db", mode="w");
    f.close()  # clean between runs
    db_conn = sqlite3.connect('QuestionSimilarity.db')
    c = db_conn.cursor()
    c.execute('''CREATE TABLE questsim(idquest1 varchar(63) NOT NULL,
                                       idquest2 varchar(63) NOT NULL, 
                                       sim float(53),
                                       PRIMARY KEY (idquest1, idquest2) )''')
    db_conn.commit()

    c.execute('''SELECT * FROM questsim''')
    all_rows = c.fetchall()

    # num_qs = 0
    # for input_segment in pd.read_csv(F.QUESTIONS_FINAL_FILENAME, chunksize=2 ** 10, sep="_"):
    #     num_qs = num_qs + len(input_segment)
    # logging.info("Number of questions in the current training (sub)set: %s", num_qs) #111171

    segment_size = 10 ** 3
    #THE FOR CYCLE
    counter_1 = 0
    counter_2 = 0
    q_filehandler1 =  open(F.QUESTIONS_FINAL_TRAIN, "r", newline='')
    q_filehandler2 = open(F.QUESTIONS_FINAL_TRAIN, "r", newline='')
    reader_1 = csv.reader(q_filehandler1, delimiter='_', quotechar='"')
    reader_2 = csv.reader(q_filehandler2, delimiter='_', quotechar='"')
    array_questvec_sims = []
    array_kws_sims = []
    array_similarity = []

    while True:
        try:
            q1_ls = reader_1.__next__()
            q1_t = quest_lstonamedtuple(q1_ls)
            if len(q1_t.id) < 10: # (filtering out any undue headers)
                continue
            counter_2 = 0
            counter_1 = counter_1 + 1
            if counter_1 > threshold_elem1:
                break  # manual method to break out of the loop, so that we can examine the ranges of the different contributions to similarity
        except StopIteration:
            logging.warning("Loop completed on element 1")
            break #end
        while True:
            try:
                q2_ls = reader_2.__next__()
                q2_t = quest_lstonamedtuple(q2_ls)
                #logging.info("Q1.id: %s", q1_t.id)
                #logging.info("Q2.id: %s", q2_t.id)
                if len(q1_t.id) >= 8 and len(q2_t.id) >= 8:  # (filtering out any undue headers)
                    counter_2 = counter_2 + 1
                    if counter_2 > threshold_elem2:
                        break
                    prod_start = time()
                    (sim, sim_parts) = compute_2questions_similarity_singleprocess(q1_t, q2_t)
                    array_questvec_sims.append(sim_parts[0])
                    array_kws_sims.append(sim_parts[1])
                    array_similarity.append(sim)
                    c.execute('''INSERT INTO questsim
                                        VALUES (?,?,?);
                    ''', (q1_t.id, q2_t.id, sim))
                    db_conn.commit()
                    prod_end = time()
                    logging.info("Similarity: %s", sim)
                    logging.info("Question similarity: total time: %s seconds\n", round(prod_end - prod_start, 6))
            except StopIteration:
                q_filehandler2 = open(F.QUESTIONS_FINAL_TRAIN, "r", newline='')
                reader_2 = csv.reader(islice(q_filehandler2, counter_1, None), delimiter='_', quotechar='"') #restart for element 2
                exc_info = sys.exc_info()
                logging.warning("Exception information: %s", exc_info)
                logging.warning("Resetting the loop for element 2")
                break

    db_conn.commit()
    logging.info("Number of cosine similarity values computed: %s", len(array_questvec_sims))
    plt.figure()
    plt.hist(array_questvec_sims, bins=200, range=(-1,1))
    plt.title(r'Questvec cosine similarity')
    plt.grid(True)

    plt.figure()
    plt.hist(array_kws_sims, bins=200, range=(-1, 1))
    plt.title('Keywords "average from best-match matrix" similarity')
    plt.grid(True)
    plt.show()

    plt.figure()
    plt.hist(array_similarity, bins=200, range=(0, 1))
    plt.title('QUESTIONS SIMILARITY')
    plt.grid(True)
    plt.show()





	
