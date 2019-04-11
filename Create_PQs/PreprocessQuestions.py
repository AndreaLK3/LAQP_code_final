import Create_PQs.PreprocessDescriptions as PPD
import Create_PQs.ReadQuestions as RQ
import pandas as pd
import utilities.MyUtils as MyUtils
import time
import re
from sys import getsizeof
import nltk
import gensim.models.doc2vec as D2V
import utilities.Filenames as F
import logging

########## Main function that processes a single element
def createDocForRow(row, stopwords_pattern, punct_pattern):
    #print(row)
    row_asin = row.asin             #row["asin"]
    row_unixtime = row.unixTime
    row_id = row_asin + "@" + str(row_unixtime)
    row_desc = str(row.question) #str(row["description"])

    if row_desc != 'nan' and len(row_desc) > 0:

        #Method2: preprocess and then use word_tokenize, that implicitly calls the sentence tokenizer
        row_desc_0 = row_desc.lower()
        row_desc_1 = PPD.expandContractions(row_desc_0)
        row_desc_2 = PPD.separateAttachedWords(row_desc_1)

        row_desc_3 = (re.compile(r'(\s-\s)')).sub(repl=" ; ", string=row_desc_2)  # isolated hyphens transformation step
        docText = stopwords_pattern.sub(repl=" ", string=row_desc_3) #stopwords removal step

        docWords_1 = nltk.tokenize.word_tokenize(docText)
        docWords_2 = list(map(PPD.removeStartingApostrophe, docWords_1))
        docWords_3 = PPD.joinSeparatedHyphens(docWords_2)

        # all punctuation signs that were not filtered
        docWords_4 = list(filter(lambda w : True if (bool(punct_pattern.match(w)) == False) else False , docWords_3))
        #if len(docWords_3) < len(docWords_2):
        #    print("Something has sbeen eliminated, " + str(len(docWords_3) - len(docWords_2)))

        row_doc = D2V.TaggedDocument(words=docWords_4, tags=[row_id])

        return row_doc






#Time spent, working on the subset: 287.4s, 309.1s, 251.5s
def createQuestionDocuments():
    MyUtils.init_logging("PreprocessQuestions.log")
    ds = open(F.QADOCS_RAW, 'w') #cleaning the file between runs
    ds.close()
    start_creatingInput = time.time()

    # Method: itertuples + pickle. Objective: preprocess text and create TaggedDocuments
    sw_pattern = PPD.getStopwordsPattern(includePunctuation=False)
    punct_pattern = re.compile(r'([!"#$%&()*+,./:;<=>?@\[\\\]^_`{|}-~\'])|([--])')
    chunk_length = 0.5 * (10 ** 5)
    with open(F.QADOCS_RAW, "a") as qadocs_file:
        qadocs_file.write(",words,tags\n")
        for input_segment in pd.read_csv(RQ.QA_TRAIN_DFPATH, chunksize=chunk_length, sep="_"):
            chunk_0 = map(lambda tupl : createDocForRow(tupl, sw_pattern,punct_pattern), input_segment.itertuples())
            chunk_1 = list(filter(lambda x: x is not None, chunk_0))
            print(getsizeof(chunk_1) // (2 ** 10))  # debugging : get size of the chunk in Kilobytes. It also works as a progress update
            pd.DataFrame(chunk_1).to_csv(path_or_buf=qadocs_file, mode="a", header=False)
            logging.info("Chunk of documents created...")

    end_creatingInput = time.time()
    logging.info("Time spent creating the Documents: %s",
                        str(round(end_creatingInput - start_creatingInput, 3)) )

