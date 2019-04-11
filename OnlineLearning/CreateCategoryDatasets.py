import OnlineLearning.OrganizeCategories as OC
import os.path
import utilities.MyUtils as MyUtils
import sqlite3
import utilities.MyUtils_flags
import utilities.MyUtils_dbs as MyUtils_dbs
import utilities.MyUtils_filesystem as MyUtils_filesystem
import utilities.MyUtils_flags as MyUtils_flags
import logging
import numpy as np
import pandas as pd
from OnlineLearning.CreateBalancedInstances import product_has_allfeatures, allquestions_have_allfeatures

### Auxiliary functions

### Shuffle database table
def shuffle_db_table(db_path):
    db_conn = sqlite3.connect(db_path)
    c = db_conn.cursor()

    temp_db = db_path + "_temp"
    f = open(temp_db, "w")
    f.close() #clean outdb
    outdb_conn = sqlite3.connect(temp_db)
    outc = outdb_conn.cursor()
    outc.execute('''CREATE TABLE instances(p varchar(63),
                                           q varchar(63),
                                           y tinyint)  ''')
    outdb_conn.commit()

    tot_num_of_rows = MyUtils_dbs.get_tot_num_rows_db(c)
    rand_indices = np.random.choice(range(1, tot_num_of_rows+1), tot_num_of_rows, replace=False)

    for ind in rand_indices:
        picked_row = c.execute("SELECT * FROM instances WHERE rowid = " + str(ind)).fetchone()
        p = picked_row[0]
        q = picked_row[1]
        y = picked_row[2]
        outc.execute('''INSERT INTO instances VALUES (?,?,?);''', (str(p), str(q), str(y)))
    outdb_conn.commit()
    logging.info("Instances have been shuffled.")

    os.rename(src=temp_db, dst=db_path)


###########
######## Creating the instances of a category dataset
def write_positive_instances(num_ps_rows, prods_db_c, quests_db_c, outc):
    ###Iterate over the products:
    for rowid in (range(1,num_ps_rows+1)):
        p_id = MyUtils_dbs.search_in_alltables_db(prods_db_c, "SELECT id FROM", "WHERE rowid = "+str(rowid))[0][0]

        ###Get all the Qs asked for the selected Ps; they will always be part of the dataset,since there are so few Ps
        q_ids_results = MyUtils_dbs.search_in_alltables_db(quests_db_c, "SELECT id FROM", "WHERE id LIKE '"+ str(p_id) +"%'")
        q_ids_ls = [tpl[0] for tpl in q_ids_results]
        #filter: p and qs must have all features
        if product_has_allfeatures(prods_db_c, p_id) and allquestions_have_allfeatures(quests_db_c, str(q_ids_ls)):
            insertion_sequence = [(p_id, q_id, 1) for q_id in q_ids_ls]
            outc.executemany("INSERT INTO instances VALUES (?,?,?)", insertion_sequence)
        else:
            logging.info("Product %s excluded from the instances due to not having all the features", p_id)


def write_negative_instances(num_ps_rows, num_qs_rows, prods_db_c, num_random_qs_per_prod, quests_db_c, outc, outdb):

    for rowid in (range(1, num_ps_rows + 1)):
        p_id = MyUtils_dbs.search_in_alltables_db(prods_db_c, "SELECT id FROM", "WHERE rowid = " + str(rowid))[0][0]
        neg_qs_ids = []
        neg_qs_indices = np.random.choice(a=range(1, num_qs_rows), size=num_random_qs_per_prod, replace=False)
        for neg_qs_index in neg_qs_indices:
            neg_qs_ids.append(MyUtils_dbs.search_in_alltables_db(quests_db_c, "SELECT id FROM",
                                                                 "WHERE `index` = " + str(neg_qs_index))[0][0])

        if product_has_allfeatures(prods_db_c, p_id) and allquestions_have_allfeatures(quests_db_c, str(neg_qs_ids)):
            insertion_sequence = [(p_id, q_id, 0) for q_id in neg_qs_ids]
            outc.executemany("INSERT INTO instances VALUES (?,?,?)", insertion_sequence)
        else:
            logging.info("Product %s excluded from the instances due to not having all the features", p_id)

        if rowid % (num_ps_rows // 10) == 0:
            logging.info("Working on category: +10%%...")
            outdb.commit()



def obtain_category_instances(category_dirpath, categ_products_db, categ_questions_db, max_neg_cardinality):
    logging.info("Extracting instances for the category: %s", os.path.basename(category_dirpath))
    prods_db_c = categ_products_db.cursor()
    quests_db_c = categ_questions_db.cursor()

    outdbname = MyUtils_flags.FLAG_INSTANCEIDS + ".db"
    f = open(os.path.join(category_dirpath, outdbname),"w"); f.close()
    outdb = sqlite3.connect(os.path.join(category_dirpath, outdbname))
    outc = outdb.cursor()
    outc.execute('''CREATE TABLE instances(p varchar(63),
                                           q varchar(63),
                                           y tinyint)  ''')
    outdb.commit()

    ### Get the number of Ps and Qs. Generally, |Ps| << |Qs| (eg. 119,43608)
    num_ps_rows = MyUtils_dbs.get_tot_num_rows_db(prods_db_c)
    logging.info("Number of products in category: %s", num_ps_rows)
    num_qs_rows = MyUtils_dbs.get_tot_num_rows_db(quests_db_c)
    logging.info("Number of questions in category: %s", num_qs_rows)
    num_possible_instances = num_ps_rows * num_qs_rows
    logging.info("Potential total number of instances from the category: %s", num_possible_instances)
    cardinality = min(num_possible_instances, max_neg_cardinality)
    logging.info("Considering the upper boundary, the number of negative instances to include in the category dataset is: %s ",
                 cardinality)
    num_random_qs_per_prod = cardinality // num_ps_rows
    logging.info("Number of random negative examples per product: %s", num_random_qs_per_prod)

    write_positive_instances(num_ps_rows, prods_db_c, quests_db_c, outc)
    outdb.commit()
    write_negative_instances(num_ps_rows, num_qs_rows, prods_db_c, num_random_qs_per_prod, quests_db_c, outc, outdb)
    outdb.commit()
    shuffle_db_table(os.path.join(category_dirpath, outdbname))

    categ_products_db.close()
    categ_questions_db.close()
    outdb.close()





###### Main execution functions
def create_category_dataset(category_dirpath, max_neg_cardinality):

    for filename in os.listdir(category_dirpath):
        if ".db" in filename and utilities.MyUtils_flags.FLAG_PRODUCTS in filename:
            categ_products_db = sqlite3.connect(os.path.join(category_dirpath, filename))
        if ".db" in filename and not(utilities.MyUtils_flags.FLAG_PRODUCTS in filename) and not( MyUtils_flags.FLAG_INSTANCEIDS in filename):
            categ_questions_db = sqlite3.connect(os.path.join(category_dirpath, filename))

    obtain_category_instances(category_dirpath, categ_products_db, categ_questions_db, max_neg_cardinality)


def create_categories_datasets(max_neg_cardinality = 10**5):
    MyUtils.init_logging("CreateCategoryDatasets.log")
    category_dirpaths = MyUtils_filesystem.get_category_dirpaths()

    for c_dirpath in category_dirpaths:
        create_category_dataset(c_dirpath, max_neg_cardinality)



