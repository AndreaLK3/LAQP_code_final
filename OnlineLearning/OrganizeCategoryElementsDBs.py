import Create_PQs.Represent_Common as RC
import utilities.MyUtils as MyUtils
import Create_PQs.MyRAKE as MyRAKE
import os
import logging
from gc import collect
import Create_PQs.RepresentQuestions as RQ
import Create_PQs.RepresentationsDatabases as RD

####### Auxiliary functions
import utilities.MyUtils_flags
import utilities.MyUtils_strings
from utilities.MyUtils_filesystem import get_category_dirpaths


def clean_category_dir(category_dirpath):
    for filename in os.listdir(category_dirpath):
        if utilities.MyUtils_flags.FLAG_KEYWORDS in filename or "withduplicateids" in filename:
            os.remove(os.path.join(category_dirpath, filename))



####### For one category: from the .csv of unprocessed questions in their initial form, create their Representation
def create_category_qs_representations(category_dirpath, d2v_model, phrases_model):

    categ_id = os.path.basename(os.path.normpath(category_dirpath))
    logging.info(categ_id)
    for filename in os.listdir(category_dirpath):
        if filename.endswith(".csv") and utilities.MyUtils_flags.FLAG_INITIAL in filename:
            questions_filename = filename
    logging.info(questions_filename)
    keywords_text_filepath = os.path.join(category_dirpath,"keywords_text_" + categ_id)
    keywords_vectors_filepath = os.path.join(category_dirpath,"keywords_vectors_" + categ_id)
    category_qs_representation_filepath = os.path.join(category_dirpath, utilities.MyUtils_strings.remove_string_start(questions_filename, utilities.MyUtils_flags.FLAG_INITIAL + '_'))

    if not (os.path.exists(category_qs_representation_filepath)):
        if not(os.path.exists(keywords_text_filepath)):
            MyRAKE.my_rake_exe(in_df_filepath=os.path.join(category_dirpath, questions_filename), elementTextAttribute="question",
                               threshold_fraction=2/3, out_kwsdf_filepath=keywords_text_filepath)
            collect()
        if not(os.path.exists(keywords_vectors_filepath)):
            MyRAKE.vectorize_keywords(in_kwsdf_filepath=keywords_text_filepath, phrases_model=phrases_model,
                                      d2v_model=d2v_model, out_kwvecs_filepath=keywords_vectors_filepath)
            collect()

        RQ.represent_questions(questions_df_filepath=os.path.join(category_dirpath, questions_filename), d2v_model=d2v_model,
                           kwsvecs_filepath=keywords_vectors_filepath,
                           outfilepath=category_qs_representation_filepath)
    else:
        logging.info("Questions' representations file for %s already made.", (os.path.basename(category_qs_representation_filepath)))

    clean_category_dir(category_dirpath)
###############


#### Phase 1: From the .csv of unprocessed questions in their initial form, create their Representation
def create_questions_representations():
    MyUtils.init_logging("OnlineLearning_create_questions_representations.log")
    categ_dirpaths = get_category_dirpaths()
    (d2v_model, phrases_model) = RC.load_the_models()
    for categ_dir in categ_dirpaths:
        create_category_qs_representations(categ_dir, d2v_model, phrases_model)
        collect()


#### Phase 2: From the csvs storing the Representations of Products and Questions, create the corresponding Databases
#### (the db-s will be later used to extract unbalanced instances for the Online Learning Tool)
def create_categories_dbs():
    categ_dirpaths = get_category_dirpaths()
    MyUtils.init_logging("OnlineLearning_create_categories_dbs.log")

    for categ_dir_p in categ_dirpaths:
        base_name = (os.path.basename(categ_dir_p))
        RD.clean_representations_dbs(categ_dir_p)

        for filename in os.listdir(categ_dir_p):
            if not(utilities.MyUtils_flags.FLAG_PRODUCTS in filename) and not(
                    utilities.MyUtils_flags.FLAG_INITIAL in filename):
                quests_csv_fname = filename
                quests_csv_path = os.path.join(categ_dir_p,quests_csv_fname)
                logging.info("Questions csv file: %s", quests_csv_path)
            elif (utilities.MyUtils_flags.FLAG_PRODUCTS in filename):
                prods_csv_fname = filename
                prods_csv_path = os.path.join(categ_dir_p,prods_csv_fname)
                logging.info("Products csv file: %s", prods_csv_path)

        quests_db_path = os.path.join(categ_dir_p, utilities.MyUtils_strings.remove_string_end(quests_csv_fname, '.csv') + '.db')
        RD.create_representations_db(quests_csv_path, quests_db_path)
        logging.info("Category: %s .Created database for the questions. Proceeding to create the db for products...", base_name)
        prods_db_path = os.path.join(categ_dir_p, utilities.MyUtils_strings.remove_string_end(prods_csv_fname, '.csv') + '.db')
        RD.create_representations_db(prods_csv_path, prods_db_path)
        logging.info("Category: %s .Created database for the products",base_name)


def full_exe():
    create_questions_representations()
    create_categories_dbs()


def debug_products_test():
    dirpath = 'OnlineLearning/CategoryDatasets/qa_Office_Products'
    prods_csv_fname = 'products_qa_Office_Products.csv'
    prods_csv_path = os.path.join(dirpath, prods_csv_fname)
    prods_db_path = os.path.join(dirpath, utilities.MyUtils_strings.remove_string_end(prods_csv_fname, '.csv') + '.db')
    RD.create_representations_db(prods_csv_path, prods_db_path)

def debug_questions_test():
    dirpath = 'OnlineLearning/CategoryDatasets/qa_Office_Products'
    quests_csv_fname = 'qa_Office_Products.csv'
    quests_csv_fpath = os.path.join(dirpath, quests_csv_fname)
    quests_db_path = os.path.join(dirpath, utilities.MyUtils_strings.remove_string_end(quests_csv_fpath, '.csv') + '.db')
    RD.create_representations_db(quests_csv_fpath, quests_db_path)


