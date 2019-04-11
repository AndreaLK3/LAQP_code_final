import numpy
from time import time
import os
import pandas as pd
import sys
import gzip

import utilities.MyUtils_strings

sys.path.append(os.path.abspath('..'))
from Create_PQs.ReadMetadata import create_dict_from_data
from Create_PQs.ReadMetadata import READKEYWORD_TRAIN, READKEYWORD_TRAINSUBSET, READKEYWORD_TEST, READKEYWORD_VALIDATION
import logging
import os.path
import utilities.Filenames as F
import utilities.MyUtils as MyUtils
import csv
from collections import namedtuple
from math import nan

QA_TRAIN_DFPATH = os.path.join(F.QA_DIR_PATH, F.QA_OUTFILE_CORENAME + "_" + READKEYWORD_TRAIN + "_df.csv")
QA_TRAINSUBSET_DFPATH = os.path.join(F.QA_DIR_PATH, F.QA_OUTFILE_CORENAME + "_" + READKEYWORD_TRAINSUBSET + "_df.csv")
QA_TEST_DFPATH = os.path.join(F.QA_DIR_PATH, F.QA_OUTFILE_CORENAME + "_" + READKEYWORD_TEST + "_df.csv")
QA_VALIDATION_DFPATH = os.path.join(F.QA_DIR_PATH, F.QA_OUTFILE_CORENAME + "_" + READKEYWORD_VALIDATION + "_df.csv")

Question = namedtuple('Question',
                          'asin answer answerTime answerType question questionType unixTime')

########## Utilities: Get the filenames we seek, and clean the folders

def get_filenames():
    fn_ls = []
    for filename in os.listdir("" + F.QA_DIR_PATH):
        if filename.endswith(".json.gz"):
            fn_ls.append(filename)
    return fn_ls

###Eliminate the old csvs
def clean_old_files(all):

    for filename in os.listdir("" + F.QA_DIR_PATH):
        if all:
            if filename.endswith(".csv"):
                filepath = "" + F.QA_DIR_PATH + "/" + filename
                logging.info("Going to clean the file: %s", str(filepath))
                f = open(filepath, "w");
                f.close();
        else:
            if filename.endswith("df.csv"):
                filepath = "" + F.QA_DIR_PATH + "/" + filename
                logging.info("Going to clean the file: %s", str(filepath))
                f = open(filepath, "w");
                f.close();

#eliminate the 0bytes files created by Python during the process
def clean_empty_files(subdir_path="./"):
    for filename in os.listdir(subdir_path):
        the_path = os.path.join(subdir_path, filename)
        if os.path.getsize(the_path) == 0:
            os.remove(the_path)

########## Initial reading of the .json.gz file, transformed into .csv-s

def readfirst_qa():
    MyUtils.init_logging("ReadQuestions.log")
    clean_old_files(all=True)
    filenames = get_filenames()
    core_filenames = list(map(lambda s: utilities.MyUtils_strings.remove_string_end(s, ".json.gz"), filenames))
    nameslist = list(zip(filenames, core_filenames))
    for (fname, core_fname) in nameslist:

        with gzip.open(F.QA_DIR_PATH + "/" + fname, 'rb') as qa_file:  # use gzip.open and rb if the file is compressed
                chunk = qa_file.readlines()  # returns a list of strings, one for each line
                qa_df = pd.DataFrame(create_dict_from_data(chunk))
                qa_df = qa_df.set_index(
                    keys="asin")  # This sets the 'asin' as the index (n: but also drops the column)

                qa_df.to_csv(F.QA_DIR_PATH + "/" + core_fname + ".csv", sep="_")
                logging.info("Did read questions subset: %s", str(fname))
    clean_empty_files()
    clean_empty_files(F.QA_DIR_PATH)


########## Organize the subfiles into the training, validation, test datasets,
########## based on the products that have been inserted into those sets

#### auxiliary function: returns a Dataframe with the selected questions
def get_questions_fordataset(df_fname, products_dataframe_filepath):

    # qa_df = pd.read_csv(df_fname, sep="_", engine='c')
    logging.info("Reading the questions datasets: Organizing subset : %s ...", df_fname)
    df_file = open(df_fname, "r", newline='')
    # logging.info(qa_df.shape)
    # tot_rows = qa_df.shape[0]
    Question = namedtuple('Question', 'asin answer answerTime answerType question questionType unixTime')

    prods_train_file = open(products_dataframe_filepath, "r", newline='')
    prods_reader = csv.reader(prods_train_file, delimiter='_', quotechar='"')
    quests_reader = csv.reader(df_file, delimiter='_', quotechar='"')
    prods_train_file.__next__()
    q_ls = quests_reader.__next__()  # skip headers

    p_ls = prods_reader.__next__()
    p_asin = p_ls[0]
    q_ls = quests_reader.__next__()
    q_asin = q_ls[0]

    selected_questions_ls = []
    quests_without_matchingprods_ls = []
    # loop:
    while True:
        try:
            match = False
            while (not match):
                while (p_asin < q_asin):
                    #logging.debug("No match; p_asin=%s , q_asin=%s", p_asin, q_asin)
                    p_ls = prods_reader.__next__()  # advance product
                    p_asin = p_ls[0]
                while (p_asin > q_asin):
                    #logging.debug("No match; p_asin=%s , q_asin=%s", p_asin, q_asin)
                    quests_without_matchingprods_ls.append(q_ls)#add to the list of questions with no matches; to be divided and assigned later
                    q_ls = quests_reader.__next__()  # advance question
                    q_asin = q_ls[0]
                if p_asin == q_asin:
                    match = True
                    logging.debug("Match; p_asin=%s ", p_asin)
                    # restore the named tuple, and add it to the list that will be transformed into a dataframe
                    q_t = Question(asin=q_ls[0], answer=q_ls[1], answerTime=q_ls[2], answerType=q_ls[3],
                                   question=q_ls[4], questionType=q_ls[5], unixTime=q_ls[6])
                    selected_questions_ls.append(q_t)
                    q_ls = quests_reader.__next__() #on to the next question
                    q_asin = q_ls[0]
        except StopIteration:
            break
        except Exception:
            exc_info = sys.exc_info()
            #silently handle the case when quote characters are unbalanced and a field is perceived as "too long".
            #The case is rare, 1 to 4 products in each dataset
            #logging.warning("Exception information: %s", exc_info)
    sel_qs_df = pd.DataFrame(selected_questions_ls)
    prods_train_file.close()

    return sel_qs_df


####auxiliary function: additional part: distribute (0.8, 0.2*0.8, 0.1, 0.1) the questions that have no matching products.
### They are still useful to build the language models, and later to provide random negative examples for the NN
def distribute_quests_withoutps(quests_without_matchingprods_df):

    num_qsnop = quests_without_matchingprods_df.shape[0]
    training_fraction = 0.8
    training_indices_set = set(numpy.random.choice(num_qsnop, int(training_fraction * num_qsnop), replace=False))
    training_indices = sorted(list(training_indices_set))
    training_subset_indices = sorted(
        numpy.random.choice(training_indices, int(0.2 * len(training_indices)), replace=False))

    remaining_indices_set = set(list(range(0, num_qsnop))) - training_indices_set
    remaining_indices_ls = list(remaining_indices_set)
    validation_indices_set = set(
        numpy.random.choice(remaining_indices_ls, int(0.5 * len(remaining_indices_ls)), replace=False))
    validation_indices = sorted(list(validation_indices_set))
    test_indices_set = remaining_indices_set - validation_indices_set
    test_indices = sorted(list(test_indices_set))

    datasets_indices = [training_indices, training_subset_indices, validation_indices, test_indices]
    datasets_files = [QA_TRAIN_DFPATH, QA_TRAINSUBSET_DFPATH, QA_VALIDATION_DFPATH, QA_TEST_DFPATH]
    d_indices_and_files = zip(datasets_indices, datasets_files)

    for subset_indices, subset_filepath in d_indices_and_files:
        subset_df = quests_without_matchingprods_df.iloc[subset_indices]
        subset_file = open(subset_filepath, "a")
        subset_df.to_csv(subset_file, sep="_")
        subset_file.close()



###### Function to organize a subfile of questions (i.e. to distribute into the datasets the questions for a category)
def organize_qa_subfile(core_filename):
    df_fname = F.QA_DIR_PATH + "/" + core_filename + ".csv"

    trainsub_quests_df = get_questions_fordataset(df_fname, F.TRAINSUBSET_MD_DF_FILEPATH)
    logging.info(trainsub_quests_df.shape)
    trainsub_file = open(QA_TRAINSUBSET_DFPATH, "a")
    trainsub_quests_df.to_csv(trainsub_file, sep="_", header=bool(int(os.path.getsize(QA_TRAINSUBSET_DFPATH)) <= 0))
    #I am now dealing elsewhere with duplicates, adding a counter number to the end of the asin.unixTime expression
    #trainsub_quests_df = trainsub_quests_df.drop_duplicates(subset=["asin", "unixTime"], keep='first', inplace=False)
    trainsub_file.close()

    train_quests_df = get_questions_fordataset(df_fname, F.TRAIN_MD_DF_FILEPATH)
    logging.info(train_quests_df.shape)
    #train_quests_df = train_quests_df.drop_duplicates(subset=["asin", "unixTime"], keep='first', inplace=False)
    train_file = open(QA_TRAIN_DFPATH, "a")
    train_quests_df.to_csv(train_file, sep="_", header=bool(int(os.path.getsize(QA_TRAIN_DFPATH))<= 0))
    train_file.close()

    valid_quests_df = get_questions_fordataset(df_fname, F.VALID_MD_DF_FILEPATH)
    logging.info(valid_quests_df.shape)
    #valid_quests_df = valid_quests_df.drop_duplicates(subset=["asin", "unixTime"], keep='first', inplace=False)
    valid_file = open(QA_VALIDATION_DFPATH, "a")
    valid_quests_df.to_csv(valid_file, sep="_", header=bool(int(os.path.getsize(QA_VALIDATION_DFPATH)) <= 0))
    valid_file.close()

    test_quests_df = get_questions_fordataset(df_fname, F.TEST_MD_DF_FILEPATH)
    logging.info(test_quests_df.shape)
    #test_quests_df = test_quests_df.drop_duplicates(subset=["asin", "unixTime"], keep='first', inplace=False)
    test_file = open(QA_TEST_DFPATH, "a")
    test_quests_df.to_csv(test_file, sep="_", header=bool(int(os.path.getsize(QA_TEST_DFPATH)) <= 0))
    test_file.close()

    logging.info("Reading from file: %s", df_fname)
    all_questions = pd.read_csv(df_fname, sep="_", dtype='str')
    all_questions.replace(to_replace=nan, value='', inplace=True, limit=None, regex=False, method='pad')
    all_questions_ls = all_questions.values
    all_questions_lts = list(map(lambda q_ls : tuple(q_ls), all_questions_ls))
    all_questions_set = set(all_questions_lts)

    matching_qs_datasets = [train_quests_df, valid_quests_df, test_quests_df]

    for m_qs_df in matching_qs_datasets:

        m_qs_df_ls = m_qs_df.values
        m_qs_df_lts = list(map(lambda q_ls: tuple(q_ls), m_qs_df_ls))
        m_qs_df_set = set(m_qs_df_lts)

        logging.debug("***")

        logging.info("Size of the all_questions_set: %s", len(all_questions_set))
        logging.info("Size of the matching subset: %s", len(m_qs_df_set))
        logging.info("Size of intersection: %s", len(m_qs_df_set.intersection(all_questions_set)))
        logging.info("Size of difference All-Subset: %s", len(all_questions_set.difference(m_qs_df_set)))
        all_questions_set = all_questions_set.difference(m_qs_df_set)

    other_questions_df = pd.DataFrame(list(all_questions_set), columns=["asin", "answer", "answerTime", "answerType", "question", "questionType", "unixTime"])
    #logging.info(other_questions_df[0:5])

    distribute_quests_withoutps(other_questions_df)






def organize_qa_all():
    MyUtils.init_logging("ReadQuestions.log", loglevel=logging.INFO)
    clean_old_files(all=False)
    filenames = get_filenames()
    core_filenames = list(map(lambda s: utilities.MyUtils_strings.remove_string_end(s, ".json.gz"), filenames))
    #logging.info(str(core_filenames))
    for cf in core_filenames:
        organize_qa_subfile(cf)
    clean_empty_files()


########## Load already existing question datasets

#The values for keyword can be: "train", "trainsubset", "validation", "test"
def load_qa(keyword):
    path = ""
    if keyword == "train":
        path = QA_TRAIN_DFPATH
    elif keyword == "trainsubset":
        path = QA_TRAINSUBSET_DFPATH
    elif keyword == "validation":
        path = QA_VALIDATION_DFPATH
    elif keyword == "test":
        path = QA_TEST_DFPATH

    startLoad = time()
    qa_df = pd.read_csv(path, sep="_", engine='c')
    endLoad = time()
    logging.info("Time spent loading the file: %s" ,str(round(endLoad - startLoad, 3)))
    return qa_df