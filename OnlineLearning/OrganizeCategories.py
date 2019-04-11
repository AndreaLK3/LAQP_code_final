import os
import utilities.Filenames as F
import utilities.MyUtils
import utilities.MyUtils as MyUtils
import numpy as np
from time import time
import pandas as pd
from gc import collect
import logging
import csv
import sqlite3
import Create_PQs.Represent_Common as RC
import utilities.MyUtils_dbs
import utilities.MyUtils_filesystem
import utilities.MyUtils_flags
import utilities.MyUtils_strings


def get_csvs_filepaths():
    fn_ls = []
    for filename in os.listdir("" + F.QA_DIR_PATH):
        if filename.endswith(".csv") and (not(F.QA_OUTFILE_CORENAME in filename)):
            fn_ls.append(filename)

    fps_ls = [os.path.join(F.QA_DIR_PATH,fname) for fname in fn_ls]
    return fps_ls



### Using the Database of training products to register the matches (better speed?)
def register_products_db(prods_db_filepath, quests_filepath, out_category_prods_filepath, append=False):
    segment_size = 10**4
    allprods_db = sqlite3.connect(prods_db_filepath)
    p_cursor = allprods_db.cursor()

    if not (append):
        f = open(out_category_prods_filepath, "w");
        f.close()  # clean outfile between runs
    prods_outfile = open(out_category_prods_filepath, "a")
    if not (append):
        prods_outfile.write("id_price_titlevec_descvec_mdcategories_kwsVectors\n")

    chunk_id = 0
    last_product_asin = 'x'
    for segment in pd.read_csv(quests_filepath, sep='_', chunksize=segment_size):
        segment_start = time()
        products_out_chunk = []
        chunk_id = chunk_id+1
        for qs_tpl in segment.itertuples():
            if qs_tpl.asin != last_product_asin:
                #note: I am operating with the questions in their initial form, not final/processed. So they have asin and unixTime, not id
                search_result = utilities.MyUtils_dbs.search_in_alltables_db(p_cursor, "SELECT * FROM", "WHERE id = '" + str(qs_tpl.asin) + "'")
                #logging.debug(search_result)
                if len(search_result) > 0:
                    qs_product = utilities.MyUtils.prodls_tonamedtuple(search_result[0], offset=1)
                    #logging.info(qs_product)
                    product_representation = [qs_product.id, str(qs_product.price), qs_product.titlevec,
                                           qs_product.descvec, qs_product.mdcategories, qs_product.kwsVectors]
                    products_out_chunk.append(product_representation)
                    last_product_asin = qs_product.id
        RC.dump_chunk(products_out_chunk, chunk_id, "products", out_category_prods_filepath)
        chunk_id = chunk_id + 1

        logging.info("Time used to find matching products for a segment of category questions: %s", round(time()-segment_start,4))
        collect()
    parts_dir_path = os.path.dirname(out_category_prods_filepath)
    RC.reuniteandsort("products", parts_dir_path, out_category_prods_filepath)
    logging.info("Products found for category file : %s", quests_filepath)


#Phase 1: copy and transfer the .csv files for each category of questions
def organize_questions(qs_csv_fpaths, eliminate_prev_results=False):
    category_dir_paths = []
    for qs_csv_fpath in qs_csv_fpaths:
        qs_csv_df = pd.read_csv(qs_csv_fpath, sep='_',dtype='object')

        new_csv_fname = utilities.MyUtils_flags.FLAG_INITIAL + '_' + os.path.basename(os.path.normpath(qs_csv_fpath))
        category_subdir = utilities.MyUtils_strings.remove_string_end(os.path.basename(os.path.normpath(qs_csv_fpath)), ".csv")
        category_dir_path = os.path.join(F.ONLINE_QUESTIONS_CSVS_DIR,category_subdir)
        category_dir_paths.append(category_dir_path)
        if not os.path.exists(category_dir_path):
            os.makedirs(category_dir_path)
        if eliminate_prev_results:
            utilities.MyUtils_filesystem.clean_directory(category_dir_path)

        qs_csv_df.to_csv(path_or_buf=os.path.join(category_dir_path, new_csv_fname), sep='_')

        logging.info("Organizing questions datasets by category. Processed: %s",new_csv_fname)
        collect()
    return category_dir_paths


#Phase 2: Find and store all the products that belong to a category
def attach_category_products(category_dir_path=None):
    MyUtils.init_logging("OnlineLearning-attach_category_products.log", logging.INFO)
    segment_size = 10**4
    #the files PRODUCTS_FINAL_TRAIN and PRODUCTS_FINAL_VALID are already sorted
    products_train_fpath = F.PRODUCTS_FINAL_TRAIN
    products_valid_fpath = F.PRODUCTS_FINAL_VALID

    for filename in os.listdir(category_dir_path):
        if filename.endswith(".csv") and utilities.MyUtils_flags.FLAG_INITIAL in filename:
            category_products_filepath = os.path.join(category_dir_path, utilities.MyUtils_flags.FLAG_PRODUCTS + utilities.MyUtils_strings.remove_string_start(filename,
                                                                                                                                                               utilities.MyUtils_flags.FLAG_INITIAL))
            logging.info("File in which to store the products belonging to the category:%s", category_products_filepath)
            category_qs_fpath = os.path.join(category_dir_path,filename)
            #logging.info("%s", category_qs_fpath)
            #register_products(products_train_fpath, category_qs_fpath, category_products_filepath, append=False)
            register_products_db(F.PRODUCTS_FINAL_TRAIN_DB, category_qs_fpath, category_products_filepath, append=False)
            register_products_db(F.PRODUCTS_FINAL_VALID_DB, category_qs_fpath, category_products_filepath, append=True)


##### Main execution function
def organize_category_datasets():
    MyUtils.init_logging("OnlineLearning-organize_category_datasets.log")

    qs_csv_fpaths = get_csvs_filepaths()
    category_dir_paths = organize_questions(qs_csv_fpaths)
    for category_dir_p in category_dir_paths:
        attach_category_products(category_dir_p)



#extract 90% to send it to a "training" set, and 10% to a test set... but this would fragment the questions asked for the same product
    #it is better if any division is made on the products of the category...
    # for qs_csv_fpath in qs_csv_fpaths:
    #     csv_num_lines = MyUtils_2.get_csv_length(qs_csv_fpath)
    #     train_indices = sorted(np.random.choice(a=range(csv_num_lines), size=int(0.9 * csv_num_lines), replace=False))
    #     test_indices = sorted(set(range(csv_num_lines)) - set(train_indices))
    #     qs_csv_df = pd.read_csv(qs_csv_fpath, sep='_',dtype='object')
    #     train_df = qs_csv_df.iloc[train_indices]
    #     test_df = qs_csv_df.iloc[test_indices]
    #
    #     new_csv_fname = 'ol_'+ os.path.basename(os.path.normpath(qs_csv_fpath))
    #     category_subdir = MyUtils.remove_string_end(os.path.basename(os.path.normpath(qs_csv_fpath)), ".csv")
    #     category_dir_path = os.path.join(F.ONLINE_QUESTIONS_CSVS_DIR,category_subdir)
    #     if not os.path.exists(category_dir_path):
    #     MyUtils_2.clean_directory(category_dir_path)
    #
    #     train_df.to_csv(path_or_buf=os.path.join(category_dir_path, MyUtils.FLAG_TRAIN+'_'+new_csv_fname), sep='_')
    #     test_df.to_csv(path_or_buf=os.path.join(category_dir_path, MyUtils.FLAG_TEST+'_'+new_csv_fname), sep='_')
    #
    #     logging.info("Organizing questions datasets by category. Processed: %s",new_csv_fname)
    #     collect()

# ##### Yet another modified version of the method to find matches between products and questions in 2 sorted csv files
# def register_products(prods_filepath, quests_filepath, out_category_prods_filepath, append=False):
#     start = time()
#     if not (append):
#         f = open(out_category_prods_filepath, "w");
#         f.close()  # clean outfile between runs
#
#     prods_outfile = open(out_category_prods_filepath, "a")
#     if not (append):
#         prods_outfile.write("id_price_titlevec_descvec_mdcategories_kwsVectors\n")
#
#     prods_filehandler = open(prods_filepath, "r", newline='')
#     quests_filehandler = open(quests_filepath, "r", newline='')
#     reader_1 = csv.reader(prods_filehandler, delimiter='_', quotechar='"')
#     reader_2 = csv.reader(quests_filehandler, delimiter='_', quotechar='"')
#
#     num_prods_withmatches = 0
#     num_products_reviewed = 0
#     num_questions_reviewed = 0
#     last_prod_id = "x"
#     questionsasked_ids_ls = []
#     ### init:
#     reader_1.__next__();
#     reader_2.__next__();
#     #reader_1.__next__();
#     #reader_2.__next__()  # skip headers
#     p_ls = reader_1.__next__()
#     q_ls = reader_2.__next__()
#     logging.info(p_ls)
#     prod_t = MyUtils_2.prodls_tonamedtuple(p_ls, offset=0)
#     logging.info(q_ls)
#     quest_t = MyUtils_2.quest_lstonamedtuple(q_ls)
#     q_prod = (quest_t.id)[0:10]
#     # loop:
#     while True:
#         try:
#             match = False
#             while not (match):
#                 while q_prod > prod_t.id or (len(q_prod) > len(prod_t.id)):
#                     logging.debug("%s < %s", prod_t.id, q_prod)
#                     p_ls = reader_1.__next__()  # advance product
#                     num_products_reviewed = num_products_reviewed + 1
#                     prod_t = MyUtils_2.prodls_tonamedtuple(p_ls, offset=0)
#
#                 while q_prod < prod_t.id or (len(q_prod) < len(prod_t.id)):
#                     logging.debug("%s > %s", prod_t.id, q_prod)
#                     q_ls = reader_2.__next__()  # advance question
#                     num_questions_reviewed = num_questions_reviewed + 1
#                     quest_t = MyUtils_2.quest_lstonamedtuple(q_ls)
#                     q_prod = (quest_t.id)[0:10]
#
#                 if q_prod == prod_t.id:
#                     match = True
#                     # barrier: feature filtering on products and questions; DB lookup:
#                     logging.info("Match: product: %s , \t question: %s", prod_t.id, quest_t.id)
#                     # positive_qs_ids_file.write(str(quest_t.id) + "\n")#store the question id (positive example)
#                     if len(prod_t.id) > 5:
#                         if prod_t.id != last_prod_id:
#                             prods_outfile.write("_".join([prod_t.id, prod_t.price, prod_t.titlevec, prod_t.descvec,
#                                                           prod_t.mdcategories, prod_t.kwsVectors]))
#                             last_prod_id = prod_t.id
#                             num_prods_withmatches = num_prods_withmatches + 1  # n: matches = number of products that have questions
#                         else:
#                             logging.info("***")
#                             # same product as previously; do nothing
#                     # on to the next question:
#                     q_ls = reader_2.__next__()
#                     quest_t = MyUtils_2.quest_lstonamedtuple(q_ls)
#                     q_prod = (quest_t.id)[0:10]
#
#         except StopIteration as e:
#             logging.warning("Exception: %s", e)
#             break
#     logging.info("Total number products of the category that have matching question: %s", num_prods_withmatches)
#     logging.info("Products reviewed: %s", num_products_reviewed)
#     logging.info("Questions reviewed: %s", num_questions_reviewed)
#
#     end = time()
#     logging.info("Time elapsed: %s", round(end - start, 4))
#     prods_outfile.close()
#     prods_filehandler.close()
#     quests_filehandler.close()
#     return num_prods_withmatches
