#import Create_PQs.PreprocessDescriptions as PPD
#import Create_PQs.PreprocessQuestions as PPQ
import gensim.models.doc2vec as D2V
import utilities.MyUtils as MyUtils
from time import time
from multiprocessing import cpu_count
import utilities.Filenames as F
import pandas as pd
import numpy
import pympler.asizeof as mem
import ast
import logging
from gc import collect
import os.path

#Time elapsed: 380.05s
def create_docvectors_model():
    MyUtils.init_logging("VectorizeDescriptions.log")
    start = time()

    docs_percent_touse = 1  # on the full training set, 0.3 is probably advisable.
    chunk_size = 10 ** 5

    doc_filenames = [F.DESCDOCS, F.QADOCS]
    doc_files = [open(doc_filename, "r") for doc_filename in doc_filenames]
    trainingset_ls = []
    for doc_file in doc_files:
        for descdocs_chunk in pd.read_csv(doc_file, chunksize=chunk_size):
            len_c = len(descdocs_chunk)
            indices = list(sorted(numpy.random.choice(len_c, int(docs_percent_touse * len_c), replace=False)))
            selected_rows = descdocs_chunk.iloc[indices]
            docs = []
            for tupl in selected_rows.itertuples():
                docs.append(D2V.TaggedDocument(words=ast.literal_eval(tupl.words), tags=ast.literal_eval(tupl.tags)))
            trainingset_ls.extend(docs)
            logging.info("Reading in the documents' words. Chunk processed...")
        logging.info("Completed: reading in a set of documents.")
        doc_file.close()
    del doc_files; del doc_filenames; collect()


    print(trainingset_ls[0:1])
    logging.info("Total number of documents in the corpus: %s", len(trainingset_ls))
    logging.info("Starting to build vocabulary and Doc2Vec model.")

    model = D2V.Doc2Vec(min_count=4, size=200, dm=1, workers=cpu_count(), # Ignore singletons; create vectors of 200 dims; use PV-DM
                        docvecs_mapfile = os.path.join("gensim_models", "doc2vec_memorymapped_vectors"))
    # create the overall vocabulary, from the descriptions and the questions:
    model.build_vocab(documents=trainingset_ls, update=False, progress_per=10000, keep_raw_vocab=False)

    logging.info("D2V Vocabulary created")

    model.train(documents=trainingset_ls, total_examples=len(trainingset_ls), epochs=10, start_alpha=0.025,
                end_alpha=0.001, word_count=0, queue_factor=2, report_delay=1.0)

    model.save(F.D2V_MODEL)

    end = time()
    logging.info("Doc2Vec model saved. Time elapsed = %s", str(round(end - start , 3)))
    logging.info("Memory size in MBs = %s", str(mem.asizeof(model) // 2 ** 20))

    return model



def load_vocabulary_withfrequencies():
    model_v = D2V.Doc2Vec.load(F.D2V_VOCABULARY)
    return model_v.raw_vocab



#loads a pre-made model (that also includes a pre-made vocabulary)
def load_model():
    model = D2V.Doc2Vec.load(F.D2V_MODEL)
    #example_asin = "0001048775"
    #MyUtils.printAndLog(str(model.docvecs[example_asin]))
    #to get vector of document that are not present in corpus
    #docvec = d2v_model.docvecs.infer_vector(‘war.txt’)
    return model




def test():
    MyUtils.init_logging("VectorizeDescriptions.log")
    docs_percent_touse = 1  # on the full training set, 0.3 is probably advisable.
    chunk_size = 10 ** 5

    doc_filenames = [F.DESCDOCS] #, F.QADOCS_FILEPATH
    trainingset_ls = []
    for doc_filename in doc_filenames:
        for descdocs_chunk in pd.read_csv(doc_filename, chunksize=chunk_size):
            len_c = len(descdocs_chunk)
            indices = list(sorted(numpy.random.choice(len_c, int(docs_percent_touse * len_c), replace=False)))
            selected_rows = descdocs_chunk.iloc[indices]
            docs = []
            for tupl in selected_rows.itertuples():
                docs.append(D2V.TaggedDocument(words=ast.literal_eval(tupl.words), tags=ast.literal_eval(tupl.tags)))
            trainingset_ls.extend(docs)
            logging.info("Reading in the documents' words. Chunk processed...")
        logging.info("Completed: reading in a set of documents.")

    d2v_model = load_model()

    subset = trainingset_ls[0:5]
    logging.debug("%s", str(subset))
    for doc in  subset:
        tag = doc.tags
        logging.debug("*** : %s" , str(tag))
        logging.debug("XXX : %s" , str(tag[0]))
        logging.debug("%s",str(d2v_model.docvecs[tag[0]]))
