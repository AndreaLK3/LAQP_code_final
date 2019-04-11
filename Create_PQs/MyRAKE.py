from time import time
import nltk
import Create_PQs.ReadMetadata as RM
import Create_PQs.PreprocessDescriptions as PD
import numpy
import re
import utilities.MyUtils as MyUtils
import ast
import multiprocessing
import pathos.pools
import pandas as pd
from math import ceil
import logging
from gc import  collect

###### I : label the delimiters and the content words ######

#tuple code : 0 = content word; 1 = delimiter
def markAsDelimiter(word):
    if "STOPWORD" in word:
        return (1,word)
    else:
        return (0,word)


def markContent(text, stopwords_pattern):
    desc_0 = text.lower()
    desc_1 = PD.expandContractions(desc_0)
    desc_2 = PD.separateAttachedWords(desc_1)

    desc_3 = (re.compile(r'(\s-\s)')).sub(repl=" ; ", string=desc_2)  # isolated hyphens transformation step
    docText = stopwords_pattern.sub(repl=" STOPWORD ", string=desc_3)
    desc_ls = nltk.tokenize.word_tokenize(docText)

    desc_ls_1 = list(map(PD.removeStartingApostrophe, desc_ls))
    desc_ls_2 = PD.joinSeparatedHyphens(desc_ls_1)
    coded_ls = list(map(lambda word : markAsDelimiter(word), desc_ls_2))

    return coded_ls
###########

###### II : extract the candidate keywords (i.e. sequences of contiguous content words)
def getCandidateKeywords(coded_ls):
    candidates = []
    candidate = []
    for tuple in coded_ls:
        label = tuple[0]
        word = tuple[1]
        if label == 0:
            candidate.append(word)
        if label == 1:
            if len(candidate) > 0:
                candidates.append(candidate)
            candidate = []
    if len(candidate) > 0:
        candidates.append(candidate)
    return candidates
###########


########### III : create the word scores #########
def create_wordScores(candidates):

    words_set = set()
    for c in candidates:
        for w in c:
            words_set.add(w)

    wordData_dict = {} #each word 's value is a dictionary, with frequency and degree
    for word in words_set:
        wordData_dict[word] = {"frequency": 0, "degree":0}
        for c in candidates:
            if word in c:
                wordData_dict[word]["frequency"] = wordData_dict[word]["frequency"] + 1
                wordData_dict[word]["degree"] = wordData_dict[word]["degree"] + len(c)

    wordScores_dict = {}
    for w,dict in wordData_dict.items():
        wordScores_dict[w] = computeWordScore(dict["frequency"], dict["degree"])

    return wordScores_dict


def computeWordScore(freq, deg):
    return deg / freq
##########

########### IV: get the scores for the candidate keywords, and select the topmost T #########
def create_candidateScores(candidates, wordScores_dict):

    candidates = eliminate_duplicates_lls(candidates)
    candidateScores_tls = [] #list of tuples: candidate string representation, score
    for c in candidates:
        c_representation = ""
        score = 0
        for w in c:
            c_representation = c_representation + " " + w
            score = score + wordScores_dict[w]
        candidateScores_tls.append((c_representation, score))

    return candidateScores_tls


def checkForDuplicates(cand1, cand2):
    identical = True
    if len(cand1) != len(cand2):
        identical = False
    for i in range(len(cand1)):
        w1 = cand1[i]
        w2 = cand2[i]
        if w1 != w2:
            identical = False
    return identical


def eliminate_duplicates_lls(candidates):
    candidates_no_duplicates = []

    if len(candidates) < 2:
        return candidates
    else:
        for i in range(0,len(candidates)):
            c1 = candidates[i]
            identical = False
            for j in range(i+1,len(candidates)):
                c2 = candidates[j]
                if len(c1)==len(c2):
                    identical = max(int(identical), int(checkForDuplicates(c1,c2)))
            if bool(identical) == True:
                #print("Found duplicate entry for:" + str(c1))
                pass#skip c1. Only c2 will be added
            else:
                candidates_no_duplicates.append(c1)


    return candidates_no_duplicates


def select_keywords(candidates, candidateScores_tls, threshold_fraction):
    all_contentwords = numpy.array(candidates).flatten()
    N = len(all_contentwords)
    T = ceil( N * threshold_fraction)

    candidateScores_tls.sort(key = lambda tuple : tuple[1], reverse = True)
    return candidateScores_tls[0:T]



########## Main functions ##########


def apply_my_rake(text, stopwords_pattern, threshold_frac):
    coded_ls = markContent(text, stopwords_pattern)
    candidates = eliminate_duplicates_lls(getCandidateKeywords(coded_ls))
    dict_wordScores = create_wordScores(candidates)
    candScores = create_candidateScores(candidates, dict_wordScores)
    keywords = select_keywords(candidates, candScores, threshold_frac)
    return keywords



def map_applymyrake(args_tuple):
    element = args_tuple[0]; elem_txtattribute = args_tuple[1]
    stopwords_pattern = args_tuple[2]; indf_columns = args_tuple[3]
    threshold_fraction = args_tuple[4]
    keywords = None
    if str(getattr(element, elem_txtattribute)) != "nan" and len(str(getattr(element, elem_txtattribute))) > 0:
        keywords = apply_my_rake(str(getattr(element, elem_txtattribute)), stopwords_pattern, threshold_fraction)
    if keywords is not None:
        if 'unixTime' in indf_columns:  # we are processing questions. Use the extended ID
            return (str(element.asin) + "@" + str(element.unixTime), keywords)
        else:
            return (element.asin, keywords)
    else:
        return None

def apply_rake_to_element(element, elem_txtattribute,  stopwords_pattern , indf_columns, threshold_fraction):
    keywords = None
    if str(getattr(element, elem_txtattribute)) != "nan" and len(str(getattr(element, elem_txtattribute))) > 0:
        keywords = apply_my_rake(str(getattr(element, elem_txtattribute)), stopwords_pattern, threshold_fraction)
    if keywords is not None:
        if 'unixTime' in indf_columns:  # we are processing questions. Use the extended ID
            return (str(element.asin) + "@" + str(element.unixTime), keywords)
        else:return (element.asin, keywords)
    else:
        return None




def my_rake_exe(in_df_filepath, elementTextAttribute, threshold_fraction, out_kwsdf_filepath):
    #logging.basicConfig(filename="MyRAKE.log", level=logging.DEBUG,
    #                    format='%(asctime)s -%(levelname)s:%(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
    MyUtils.init_logging(logfilename="MyRAKE.log")
    logging.info("Keyword extraction started.")
    sw_pattern = PD.getStopwordsPattern(includePunctuation=True)
    numbers_pattern = re.compile(r'([0-9])+') #numbers are not keywords
    allsw_expression = "|".join([sw_pattern.pattern, numbers_pattern.pattern])
    allsw_pattern = re.compile(allsw_expression)
    f = open(out_kwsdf_filepath, "w"); f.close() #clean between runs

    segment_nrows = int(1.0 * 10**4)
    current_segment = 1
    logging.info("Number of elements in a segment: " + str(segment_nrows))

    with open(out_kwsdf_filepath, "a") as outfile:
        outfile.write(",id,keywords\n")
        with open(in_df_filepath, "r") as in_df_file:
            #logging.info("current subfile being processed: %s", traindf_filepath)
            for input_segment in pd.read_csv(in_df_file, chunksize=segment_nrows, sep="_", engine='c', error_bad_lines=False):
                executor = pathos.pools.ProcessPool(max(1,multiprocessing.cpu_count()-1))
                if len(input_segment) < segment_nrows:
                    logging.warning("Segment with length %s < %s ;\n"+"either lines with unreadable characters were dropped,"
                                    + "or this is the last chunk", len(input_segment), segment_nrows)
                seg_start = time()
                #seg_lts = []
                args = [(MyUtils.refine_tuple(element), elementTextAttribute, allsw_pattern,
                        input_segment.columns, threshold_fraction) for element in input_segment.itertuples()]
                #logging.info("The arguments for the current segment have been created. Length: %s", len(args))
                seg_lts_map = executor.map(map_applymyrake, args) #executor.map
                #logging.info("Mapping operation completed: keywords created; proceeding to filter intermediate list...")
                seg_lts = list( filter(lambda x : x is not None, seg_lts_map))
                # for prod_tuple in input_segment.itertuples():
                #     logging.info(prod_tuple.asin)
                #     id_kws_tuple = apply_rake_to_element(prod_tuple, elementTextAttribute, allsw_pattern,
                #                                            input_segment.columns,threshold_fraction)
                #     if id_kws_tuple is not None:
                #         seg_lts.append(id_kws_tuple)
                #logging.info("List filtered; proceeding to save keywords to file...")
                pd.DataFrame(seg_lts).to_csv(outfile, mode="a", header=False)
                seg_end = time()
                logging.info("* Keyword extraction ; the segment n. %s of the input dataframe has been processed in %s seconds",
                             current_segment, str(round(seg_end - seg_start, 3)) )
                executor.terminate()
                executor.restart()
                collect()
                current_segment = current_segment+1
    logging.info("Keyword extraction : finished.")

###############
######### Functions to use the Doc2Vec model to transform the keywords into vectors

def vectorize_kw_ls(argstuple):
    #t00 = time()
    prod_kws=argstuple[0]; phrases_model=argstuple[1]; d2v_model=argstuple[2]
    max_length = argstuple[3]
    id = prod_kws.id
    kw_ls= ast.literal_eval(prod_kws.keywords)
    n = min(len(kw_ls), max_length) # Due to space and speed reasons, I pick only the first 10 keywords of a prod desc
    kw_ls = kw_ls[0:n]
    #print(kw_ls)
    kwvecs_ls = []
    for kw_tuple in kw_ls:
        kw = kw_tuple[0]
        kw_phrased = phrases_model[kw.split()]
        kw_vec = d2v_model.infer_vector(kw_phrased)
        kwvecs_ls.append(kw_vec)
    #t11 = time()
    return (id,kwvecs_ls)


def dict_key_generator(a_dict):
    for key in a_dict.keys():
        yield key

def vectorize_keywords(in_kwsdf_filepath, phrases_model, d2v_model, out_kwvecs_filepath):
    MyUtils.init_logging(logfilename="MyRAKE_vectorizekws.log")
    logging.info("Started to vectorize the keywords")
    f = open(out_kwvecs_filepath, "w");  f.close()  # clean between runs
    segment_nrows  =  10**4 #10**4
    logging.info("Number of elements in a segment: %s",str(segment_nrows))

    current_segment = 1
    max_len = 10 #n of keywords to extract
    with open(out_kwvecs_filepath, "a") as outfile:
        outfile.write(",id,kwsVectors\n")
        for input_segment in pd.read_csv(in_kwsdf_filepath, chunksize=segment_nrows):
            executor =  pathos.pools.ThreadPool(multiprocessing.cpu_count())
            t00 = time()
            args = ((elem_kws, phrases_model, d2v_model, max_len) for elem_kws in input_segment.itertuples())
            kws_vecs= list( executor.map( vectorize_kw_ls, args))
            pd.DataFrame(kws_vecs).to_csv(outfile, mode='a', header=False)
            logging.info("Keyword vectorization ; segment n.%s of the input dataframe has been processed...", current_segment)
            current_segment = current_segment +1
            t11 = time()
            logging.info("Time elapsed for a segment : %s", str(round(t11 - t00,3)))
            executor.terminate()
            executor.restart()
    logging.info("Keyword vectorization : finished.")



##### Test functions
def test_my_rake():
    the_stopwords_pattern = PD.getStopwordsPattern(includePunctuation=True)
    md_df = RM.load_md(RM.READKEYWORD_TRAINSUBSET)
    elem = MyUtils.pickRandomElement(md_df)
    while elem.description == "nan" or len(elem.description)==0: #a null value may be nan (for prods) or '' (for quests)
        elem = MyUtils.pickRandomElement(md_df)
    apply_my_rake(elem.description, the_stopwords_pattern)

