import logging
import sqlite3
from utilities import Filenames as F
import utilities.MyUtils_flags as MyUtils_flags

###########
###### Search in DB with multiple tables #####
def search_in_alltables_db(dbcursor, query_pretext, query_aftertext, logqueries=False):
    tables = [row[0] for row in dbcursor.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    answers = []
    for table in tables:
        query_text = query_pretext +" "+ str(table) +" " + query_aftertext
        if logqueries:
            logging.info(query_text)
        answers.extend(dbcursor.execute(query_text).fetchall())
    answers_not_empty = [answer for answer in answers if len(answer)>0]
    #logging.info(answers_not_empty)
    return answers_not_empty


###########
######  Obtain the total number of rows in a multi-table dataset:
def get_tot_num_rows_db(db_cursor):
    results = search_in_alltables_db(db_cursor, "SELECT COUNT(*) FROM" , "")
    results_ls = [tpl[0] for tpl in results]
    tot_rows = sum(results_ls)
    return tot_rows

##########
###### Get the number of instances of a dataset (training, validation, test)
def get_nn_dataset_length(dataset_typeflag):
    if dataset_typeflag == MyUtils_flags.FLAG_TRAIN:
        db_conn = sqlite3.connect(F.NN_TRAIN_INSTANCES_DB)
    elif dataset_typeflag == MyUtils_flags.FLAG_VALID:
        db_conn = sqlite3.connect(F.NN_VALID_INSTANCES_DB)
    else: #test
        db_conn = sqlite3.connect(F.NN_TEST_INSTANCES_DB)
    db_cursor = db_conn.cursor()
    db_cursor.execute('''SELECT COUNT(x)
                                FROM instances''')
    dataset_length = db_cursor.fetchone()[0]

    db_conn.close()
    return dataset_length