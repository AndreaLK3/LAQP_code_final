import Create_PQs.ExtractMetadataInfo as EMI
import Create_PQs.Represent_Common as RC
from time import time
import Create_PQs.MyRAKE as MyRAKE
import utilities.Filenames as F
import Create_PQs.PreprocessDescriptions as PPD
import pandas as pd
import logging
import utilities.MyUtils as MyUtils
from gc import collect
import re
import os.path

##### Functions that finalize the encoding for products and questions #####
import utilities.MyUtils_flags


def represent_products(d2v_model, metainfo_df_filepath, kwsvecs_filepath, outfilepath):
    MyUtils.init_logging("EncodeProducts.log")

    f =  open(outfilepath, "w"); f.close() #clean outfile between runs

    #get stopwords & punctuation pattern; when we are operating on the test or validation sets,
    #we need to preprocess the product description in order to infer_vector in the Doc2Vec model
    stopws_pattern = PPD.getStopwordsPattern(includePunctuation=False)
    punct_pattern = re.compile(r'([!"#$%&()*+,./:;<=>?@\[\\\]^_`{|}-~\'])|([--])')

    #The file with product metadata info contains all the products. The order is the same of the kws'vecs file.
    #Although the kws'vecs file has some missing products...

    chunk_size = 10**4
    with open(metainfo_df_filepath, "r") as metainfo_file:
        with open(kwsvecs_filepath, "r") as kwsvecs_file:
            metainfo_reader_iterator = pd.read_csv(metainfo_file, chunksize=1, sep="_", dtype=object)
            #metainfo_reader_iterator.read(1) #take out the header?
            chunk_id = 1
            for kwvecs_chunk in pd.read_csv(kwsvecs_file, chunksize=chunk_size):
                chunk_start = time()
                products_out_chunk = []
                for kwsvecs_tuple in kwvecs_chunk.itertuples():
                    kws_id = str(kwsvecs_tuple.id)
                    match = False
                    while (not match):
                        metainfo_elemchunk = metainfo_reader_iterator.read(1)
                        for metainfo in metainfo_elemchunk.itertuples():#there is only one
                            meta_id = str(metainfo.id)
                            if kws_id == meta_id:
                                #logging.info("Match : " + str(kws_id))
                                match = True
                                # Write down the kws_id, the product that also has keywords' vectors
                                try:
                                    desc_vec = d2v_model.docvecs[kws_id]
                                except KeyError:
                                    #In this case, I should infer the vector. To be used when percent_touse in D2V is < 1, and for validation/test
                                    desc_words = (PPD.createDocForRow(metainfo, stopws_pattern, punct_pattern)).words
                                    desc_vec = d2v_model.infer_vector(desc_words)
                                    #logging.warning("Vector of %s not found in Doc2Vec model", str(kws_id))
                                product_representation = (metainfo.id, metainfo.price, metainfo.titlevec,
                                                    desc_vec, metainfo.mdcategories, kwsvecs_tuple.kwsVectors)
                                products_out_chunk.append(product_representation)
                            else:
                                #logging.info("No match between  : " + str(kws_id) + " and " + str(meta_id))
                                #Write down the meta_id, the product without the keywords' vectors
                                product_representation = (metainfo.id, metainfo.price, metainfo.titlevec,
                                                    "NODESCVEC", metainfo.mdcategories, "NOKWSVECTORS")
                                products_out_chunk.append(product_representation)
                RC.dump_chunk(products_out_chunk, chunk_id, "products", outfilepath)
                chunk_id = chunk_id + 1
                chunk_end = time()
                logging.info("Creating Representation of products. Chunk completed in time : %s seconds",
                                    str(round(chunk_end - chunk_start, 3)))
                collect()

    # if "train" in outfilepath:
    #     dataset_type = MyUtils.FLAG_TRAIN
    # elif "valid" in outfilepath:
    #     dataset_type = MyUtils.FLAG_VALID
    # elif "test" in outfilepath:
    #     dataset_type = MyUtils.FLAG_TEST

    parts_dir_path = os.path.dirname(outfilepath)
    RC.reuniteandsort("products", parts_dir_path, outfilepath)
    logging.info("Completed: encoding products.")




##### Main functions, for the execution #####

#To use ONLY when the previous execution was exe_trainsubset_full,
# otherwise the intermediate results will be pertaining to other datasets (eg. test, validation)
def exe_train_premade():
    files_id = utilities.MyUtils_flags.FLAG_PRODS + "_" + utilities.MyUtils_flags.FLAG_TRAIN
    #description documents have already been created, and prepared with phrases as well
    (d2v_model, phrases_model)= RC.load_the_models()
    # Keyword vectors already made. Just need to put it together
    represent_products(d2v_model, F.PRODUCTS_METADATA + files_id, F.DESC_KWSVECS + files_id, F.PRODUCTS_FINAL_TRAIN)


def exe_train_full():
    #ENC.prepare_datasets() #generally invoked elsewhere, so that the dataset for questions and products remains identical
    files_id = utilities.MyUtils_flags.FLAG_PRODS + "_" + utilities.MyUtils_flags.FLAG_TRAIN

    #(d2v_model, phrases_model) = RC.create_the_models()
    (d2v_model, phrases_model) = RC.load_the_models()
    EMI.process_all_mdinfo(F.TRAIN_MD_DF_FILEPATH, F.PRODUCTS_METADATA + files_id, phrases_model, d2v_model)


    # Keywords from the description text
    MyRAKE.my_rake_exe(in_df_filepath=F.TRAIN_MD_DF_FILEPATH, elementTextAttribute="description",
                       threshold_fraction=1/3, out_kwsdf_filepath=F.DESCS_KEYWORDS_RAW +files_id)

    MyRAKE.vectorize_keywords(F.DESCS_KEYWORDS_RAW + files_id, phrases_model, d2v_model, F.DESC_KWSVECS + files_id)

    represent_products(d2v_model, F.PRODUCTS_METADATA + files_id, F.DESC_KWSVECS + files_id, F.PRODUCTS_FINAL_TRAIN)



def exe_valid():
    files_id = utilities.MyUtils_flags.FLAG_PRODS + "_" + utilities.MyUtils_flags.FLAG_VALID
    (d2v_model, phrases_model) = RC.load_the_models()

    EMI.process_all_mdinfo(F.VALID_MD_DF_FILEPATH, F.PRODUCTS_METADATA + files_id, phrases_model, d2v_model)

    # Keywords from the description text
    MyRAKE.my_rake_exe(in_df_filepath=F.VALID_MD_DF_FILEPATH, elementTextAttribute="description",
                       threshold_fraction=1 / 3, out_kwsdf_filepath=F.DESCS_KEYWORDS_RAW + files_id)

    MyRAKE.vectorize_keywords(F.DESCS_KEYWORDS_RAW + files_id, phrases_model, d2v_model, F.DESC_KWSVECS + files_id)

    represent_products(d2v_model, F.PRODUCTS_METADATA + files_id, F.DESC_KWSVECS + files_id, F.PRODUCTS_FINAL_VALID)

def exe_test():
    files_id = utilities.MyUtils_flags.FLAG_PRODS + "_" + utilities.MyUtils_flags.FLAG_TEST
    (d2v_model, phrases_model) = RC.load_the_models()

    EMI.process_all_mdinfo(F.TEST_MD_DF_FILEPATH, F.PRODUCTS_METADATA + files_id, phrases_model, d2v_model)

    # Keywords from the description text
    MyRAKE.my_rake_exe(in_df_filepath=F.TEST_MD_DF_FILEPATH, elementTextAttribute="description",
                       threshold_fraction=1 / 3, out_kwsdf_filepath=F.DESCS_KEYWORDS_RAW + files_id)

    MyRAKE.vectorize_keywords(F.DESCS_KEYWORDS_RAW + files_id, phrases_model, d2v_model, F.DESC_KWSVECS + files_id)

    represent_products(d2v_model, F.PRODUCTS_METADATA + files_id, F.DESC_KWSVECS + files_id, F.PRODUCTS_FINAL_TEST)

