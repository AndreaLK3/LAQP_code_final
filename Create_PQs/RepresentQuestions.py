#import ast
import logging
import utilities.Filenames as F
from time import time
import pandas as pd
import Create_PQs.MyRAKE as MyRAKE
import Create_PQs.ReadQuestions as RQ
import utilities.MyUtils as MyUtils
import Create_PQs.Represent_Common as RC
import csv
import os
import Create_PQs.PreprocessDescriptions as PPD
import re
from gc import collect

#### Helper function: reads one last time the already sorted .csv file of questions,
#### and separates the duplicate id strings appending "1" QUESTIONS_TEMP_FILENAME
import utilities.MyUtils_flags


def separate_duplicate_ids(dataset_fpath):

    MyUtils.init_logging("SeparateDuplicateIds.log")
    f = open(F.QUESTIONS_TEMP, "w"); f.close()#clean the pivot outfile between runs
    segment_size = 10**4
    start = time()
    counter = 0
    prodquest_counter = 0 #internal counter, to help differentiate the ids of the questions asked for the same product
    with open(dataset_fpath, "r", newline='') as questions_file_withduplicateids:
        with open(F.QUESTIONS_TEMP, "a", newline='') as questions_outfile:
            reader_old = csv.reader(questions_file_withduplicateids, delimiter='_', quotechar='"')
            writer_new = csv.writer(questions_outfile, delimiter='_', quotechar='"')
            previous_prod_id = "start"
            while True:
                try:
                    quest = reader_old.__next__()
                    #logging.info(quest)
                    quest_prod_id = quest[0][0:10] #before it used quest[1][0:10]. In the categories for the OL tool, it uses 0. (To check)
                    if previous_prod_id == quest_prod_id:
                        updated_id = quest[0] + str(prodquest_counter)
                        quest[0] = updated_id
                        logging.debug("Updated id: %s" , quest[0])
                        prodquest_counter = prodquest_counter + 1
                    else:
                        prodquest_counter = 0
                    previous_prod_id = quest_prod_id
                    writer_new.writerow(quest)
                    counter = counter +1
                    if counter % segment_size == 0:
                        logging.info("Question %s processed...", counter)
                except StopIteration:
                    end = time()
                    logging.info("Duplicate IDs separated; time elapsed: %s", round(end - start, 3))
                    break

    os.replace(dataset_fpath, dataset_fpath + "_withduplicateids")#temporarily renaming the old file
    os.rename(F.QUESTIONS_TEMP, dataset_fpath)#rename the postprocessed file to 'final'



def represent_questions(questions_df_filepath, d2v_model, kwsvecs_filepath, outfilepath):
    MyUtils.init_logging("EncodeQuestions.log")

    # get stopwords & punctuation pattern; when we are operating on the test or validation sets,
    # we need to preprocess the product description in order to infer_vector in the Doc2Vec model
    stopws_pattern = PPD.getStopwordsPattern(includePunctuation=False)
    punct_pattern = re.compile(r'([!"#$%&()*+,./:;<=>?@\[\\\]^_`{|}-~\'])|([--])')

    # The file with questions data contains all the questions. The order is the same of the kws'vecs file.
    # Although the kws'vecs file has some missing questions...
    chunk_size = 2* 10 ** 4
    chunk_id = 1
    with open(questions_df_filepath, "r") as metainfo_file:
        with open(kwsvecs_filepath, "r") as kwsvecs_file:
            metainfo_reader_iterator = pd.read_csv(metainfo_file, chunksize=1, sep="_", dtype=object)
            #metainfo_reader_iterator.read(1)  # take out the header?
            for kwvecs_chunk in pd.read_csv(kwsvecs_file, chunksize=chunk_size):
                chunk_start = time()
                qs_out_chunk = []
                for kwsvecs_tuple in kwvecs_chunk.itertuples():
                    kws_id = str(kwsvecs_tuple.id)
                    match = False
                    while (not match):
                        q_info_elemchunk = metainfo_reader_iterator.read(1)
                        for q_info in q_info_elemchunk.itertuples():  # there is only one
                            q_id = str(q_info.asin) + "@" + str(q_info.unixTime)
                            if kws_id == q_id:
                                logging.debug("Match : %s",str(kws_id))
                                match = True
                                # Write down the kws_id, the product that also has keywords' vectors
                                try:
                                    q_vec = d2v_model.docvecs[kws_id]
                                except KeyError:
                                    # In this case, I should infer the vector. To be used when percent_touse in D2V is < 1
                                    desc_words = (PPD.createDocForRow(q_info, stopws_pattern, punct_pattern)).words
                                    q_vec = d2v_model.infer_vector(desc_words)
                                #q_vec_rep = re.sub('\n','',q_vec)
                                #q_kwsVectors_rep = re.sub('\n','',kwsvecs_tuple.kwsVectors)
                                q_representation = (q_id, q_info.questionType, q_vec, kwsvecs_tuple.kwsVectors)
                                #logging.info(q_vec)

                            else:
                                logging.debug("No match between  : %s and %s" , str(kws_id), str(q_id))
                                # Write down the meta_id, the product without the keywords' vectors
                                q_representation = (q_id, q_info.questionType, "NOQUESTVEC", "NOKWSVECTORS"),

                            qs_out_chunk.append(q_representation)
                #pd.DataFrame(qs_out_chunk).to_csv(questions_final_outfile, mode="a", header="False", sep="_")
                RC.dump_chunk(qs_out_chunk, chunk_id, "questions", outfilepath)
                collect()
                chunk_end = time()
                logging.info("Creating Representation of questions. Chunk %s completed in time : %s seconds"
                             ,chunk_id,str(round(chunk_end - chunk_start, 3)))
                chunk_id = chunk_id + 1
    # if MyUtils.FLAG_TRAIN in outfilepath:
    #     dataset_typeflag = "train"
    # elif MyUtils.FLAG_VALID in outfilepath:
    #     dataset_typeflag = "valid"
    # elif MyUtils.FLAG_TEST in outfilepath:
    #     dataset_typeflag = "test"
    # else:
    #     dataset_typeflag = ''

    parts_dir_path = os.path.dirname(outfilepath)
    RC.reuniteandsort("questions", parts_dir_path, outfilepath)
    separate_duplicate_ids(outfilepath)
    logging.info("Completed: encoding questions.")




########### Main functions for the execution:

def exe_train_full(models_already_made):
    files_id = utilities.MyUtils_flags.FLAG_QUESTS + "_" + utilities.MyUtils_flags.FLAG_TRAIN
    #ENC.prepare_datasets() #generally invoked elsewhere, so that the dataset for questions and products remains identical
    if models_already_made:
        (d2v_model, phrases_model) = RC.load_the_models()
    else:
        (d2v_model, phrases_model) = RC.create_the_models()

    MyRAKE.my_rake_exe(in_df_filepath=RQ.QA_TRAIN_DFPATH, elementTextAttribute="question",
                       threshold_fraction=2/3, out_kwsdf_filepath=F.QUESTIONS_KEYWORDS_RAW + files_id)

    MyRAKE.vectorize_keywords(in_kwsdf_filepath=F.QUESTIONS_KEYWORDS_RAW + files_id, phrases_model=phrases_model,
                              d2v_model=d2v_model, out_kwvecs_filepath=F.QUEST_KWSVECS + files_id)

    represent_questions(RQ.QA_TRAIN_DFPATH, d2v_model, F.QUEST_KWSVECS + files_id, F.QUESTIONS_FINAL_TRAIN)


# Shorcut: To use ONLY when the previous execution was exe_trainsubset_full
def exe_train_premade():
    files_id = utilities.MyUtils_flags.FLAG_QUESTS + "_" + utilities.MyUtils_flags.FLAG_TRAIN
    # description documents have already been created, and prepared with phrases as well
    (d2v_model, phrases_model) = RC.load_the_models()
    # Keyword vectors already made. Just need to put it together
    represent_questions(RQ.QA_TRAIN_DFPATH, d2v_model, F.QUEST_KWSVECS + files_id,  F.QUESTIONS_FINAL_TRAIN)



def exe_valid():
    files_id = utilities.MyUtils_flags.FLAG_QUESTS + "_" + utilities.MyUtils_flags.FLAG_VALID
    (d2v_model, phrases_model) = RC.load_the_models()

    MyRAKE.my_rake_exe(in_df_filepath=RQ.QA_VALIDATION_DFPATH, elementTextAttribute="question",
                       threshold_fraction=2/3, out_kwsdf_filepath=F.QUESTIONS_KEYWORDS_RAW + files_id)

    MyRAKE.vectorize_keywords(in_kwsdf_filepath=F.QUESTIONS_KEYWORDS_RAW + files_id, phrases_model=phrases_model,
                              d2v_model=d2v_model, out_kwvecs_filepath=F.QUEST_KWSVECS + files_id)

    represent_questions(RQ.QA_VALIDATION_DFPATH, d2v_model, F.QUEST_KWSVECS + files_id, F.QUESTIONS_FINAL_VALID)


def exe_test():
    files_id = utilities.MyUtils_flags.FLAG_QUESTS + "_" + utilities.MyUtils_flags.FLAG_TEST
    (d2v_model, phrases_model) = RC.load_the_models()

    MyRAKE.my_rake_exe(in_df_filepath=RQ.QA_TEST_DFPATH, elementTextAttribute="question",
                       threshold_fraction=2/3, out_kwsdf_filepath=F.QUESTIONS_KEYWORDS_RAW + files_id)

    MyRAKE.vectorize_keywords(in_kwsdf_filepath=F.QUESTIONS_KEYWORDS_RAW + files_id, phrases_model=phrases_model,
                              d2v_model=d2v_model, out_kwvecs_filepath=F.QUEST_KWSVECS + files_id)

    represent_questions(RQ.QA_TEST_DFPATH, d2v_model, F.QUEST_KWSVECS + files_id, F.QUESTIONS_FINAL_TEST)