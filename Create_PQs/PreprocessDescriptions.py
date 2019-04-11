from nltk.corpus import stopwords
from sys import getsizeof
import gensim.models.doc2vec as D2V
import nltk
import utilities.MyUtils as MyUtils
import time
import re
import _pickle as pickle #importing cPickle here for better speed
import pandas as pd
import numpy
import utilities.Filenames as F
import logging

PUNCTUATION_PATTERNSTRING = r'([!"#$%&()*+,./:;<=>?@\[\\\]^_`{|}~\'])|([-]{2})' #r'([!"#$%&()*+,./:;<=>?@\[\\\]^_`{|}-~])'



########## I : Expand English contractions
def expandContractions(desc_string):
    #list of English contractions adapted from http://www.sjsu.edu/writingcenter/docs/Contractions.pdf,
    #(José State University Writing Center)

    contractions_dict = {
        "aren't":"are not",        "can't": "cannot",
        "couldn't": "could not",   "didn't": "did not",
        "doesn't": "does not",     "don't": "do not",
        "hadn't": "had not",       "hasn't": "has not",
        "haven't": "have not",     "he'll": "he will",
        "I'll": "I will",          "I'm": "I am",
        "I've": "I have",          "isn't": "is not",
        "mustn't": "must not",     "shan't": "shall not",
        "she'll": "she will",      "shouldn't": "should not",
        "they'll": "they will",    "they're": "they are",
        "they've": "they have",    "we're": "we are",
        "we've": "we have ",       "weren't": "were not",
        "what'll": "what will",    "what're": "what are",
        "won't": "will not",       "wouldn't": "would not",
        "you'll": "you will",      "you're": "you are",
        "you've": "you have"
    }

    for k,v in contractions_dict.items():
        desc_string = desc_string.replace(k, v)

    return desc_string

##########


########## II: separate words that were inappropriately attached by punctuation signs ##########

def replaceFunction(matchObj):
    #MyUtils.printAndLog("|" + str(matchObj.group(0)) + "|")  # wordA.wordB
    punctPatternS = r'([!"#$%&()*+,./:;<=>?@\[\\\]^_`{|}~])'
    frag_ls = re.compile(punctPatternS).split(matchObj.group(0))
    if len(frag_ls) < 3:
        print("WARNING: " + str(frag_ls))
        return " " + str(frag_ls[0]) + " "
    else:
        #print("Correct")
        return str(" " + frag_ls[0] + " " + frag_ls[1] + " " + frag_ls[2] + " " )


def separateAttachedWords(desc):
    word = "[a-zA-Z0-9]+"
    punctPatternS = r'([!"#$%&()*+,./:;<=>?@\[\\\]^_`{|}~])' #do not separate on hyphen
    pattern_inString = "\s" + word + punctPatternS + word + "\s"
    pattern_startOfString = "^" + word + punctPatternS  + word + "\s"
    pattern_endOfString = "\s" + word + punctPatternS  + word + "$"
    patternObject = re.compile(pattern_inString + "|" + pattern_startOfString + "|" + pattern_endOfString)

    desc = re.sub(pattern=patternObject, repl=replaceFunction, string=desc)
    # print(_desc)
    return desc
###########

########## III operating on a list of words: remove starting single quotes,
########## and join hyphenated words that the tokenizer had separated  ##########
def removeStartingApostrophe(word):
    if word[0] == "'":
        return word[1:]
    else:
        return word

def joinSeparatedHyphens(wordlist):
    times = len([i for i, x in enumerate(wordlist) if x == "-"])
    for time in range(times):
        try:
            hyphen_ind = wordlist.index("-")
        except ValueError:
            break #do nothing. No hyphens here
        if hyphen_ind != 0 and hyphen_ind < len(wordlist) - 1: #hyphen at the borders -> wrong, and replicates wordlist
            try:
                before = wordlist[hyphen_ind-1]
                after = wordlist[hyphen_ind + 1]
                wordlist_before = wordlist[0:hyphen_ind-1]
                wordlist_after = wordlist[hyphen_ind+2:]
                composite_word = before + "-" + after
                wordlist = wordlist_before + [composite_word] + wordlist_after
            except IndexError:
                pass#list index out of range == hyphen at the start or end of sentence. Do nothing
    return wordlist

##########


########## IV: Stopwords ##########
def getStopwordlist():
    stopwords_set = set(stopwords.words("english")) #start with the NLTK list
    source_paths = [F.DOCFREQ_STOPWORDS_FILEPATH, F.WORDFREQ_STOPWORDS_FILEPATH];

    for path in source_paths:

        the_df = pd.read_csv(path, header=0, usecols=["word"])
        the_set = set( numpy.array(the_df.values.tolist()).flatten() )
        stopwords_set = stopwords_set.union(the_set)

    stopwords_ls = list( map(re.escape , list(stopwords_set)))

    #write to file
    with open(F.STOPWORDS_OUTFILEPATH, "wb") as sw_outfile:
        pickle.dump(list(stopwords_ls), file=sw_outfile, protocol=4)

    return stopwords_ls

def getStopwordsPattern(includePunctuation):
    sw_ls = getStopwordlist()
    patternString = "|".join(list(map(lambda sw: r'\b' + sw + r'(?![\w-])', sw_ls)))  # look ahead for hyphen
    if includePunctuation:
        pattern = re.compile(
            patternString + "|" + PUNCTUATION_PATTERNSTRING)  # add punctuation signs
    else:
        pattern = re.compile(patternString)
    return pattern


def loadStopwordlist():
    with open(F.STOPWORDS_OUTFILEPATH, "rb") as sw_outfile:
        sw_ls = pickle.load(sw_outfile)
    return sw_ls

##########

########## Main function that processes a single element
def createDocForRow(row, stopwords_pattern, punct_pattern):
    try:
        row_asin = row.asin
    except AttributeError:
        row_asin = row.id #(when applied on a question, or a product tuple taken from the metadata_info intermediate results)

    try:
        row_desc = str(row.description)
    except AttributeError:  #(when applied on a question, when we seek to obtain the document to use doc2vec.infer_vector)
        row_desc = str(row.question)

    if row_desc != 'nan' and len(row_desc) > 0:

        #Method2: preprocess and then use word_tokenize, that implicitly calls the sentence tokenizer
        row_desc_0 = row_desc.lower()
        row_desc_1 = expandContractions(row_desc_0)
        row_desc_2 = separateAttachedWords(row_desc_1)

        row_desc_3 = (re.compile(r'(\s-\s)')).sub(repl=" ; ", string=row_desc_2)  # isolated hyphens transformation step
        docText = stopwords_pattern.sub(repl=" ", string=row_desc_3) #stopwords removal step

        docWords_1 = nltk.tokenize.word_tokenize(docText)
        docWords_2 = list(map(removeStartingApostrophe, docWords_1))
        docWords_3 = joinSeparatedHyphens(docWords_2)

        # all punctuation signs that were not filtered
        docWords_4 = list(filter(lambda w : True if (bool(punct_pattern.match(w)) == False) else False , docWords_3))
        #if len(docWords_3) < len(docWords_2):
        #    print("Something has been eliminated, " + str(len(docWords_3) - len(docWords_2)))

        row_doc = D2V.TaggedDocument(words=docWords_4, tags=[row_asin])

        return row_doc
    else:
        return None


#Main functions
def createDescriptionDocuments():
    MyUtils.init_logging("PreprocessDescriptions.log")
    ds = open(F.DESCDOCS_RAW, 'w') #cleaning the file between runs
    ds.close()
    start_creatingInput = time.time()

    sw_pattern = getStopwordsPattern(includePunctuation=False)
    punct_pattern = re.compile(r'([!"#$%&()*+,./:;<=>?@\[\\\]^_`{|}-~\'])|([--])')
    chunk_length = 0.5 * (10 ** 5)

    with open(F.DESCDOCS_RAW, "a") as descdocs_file:
        descdocs_file.write(",words,tags\n")
        for input_segment in pd.read_csv(F.TRAIN_MD_DF_FILEPATH, chunksize=chunk_length, sep="_"):
            chunk_0 = map(lambda tupl : createDocForRow(tupl, sw_pattern,punct_pattern), input_segment.itertuples())
            chunk_1 = list(filter(lambda x: x is not None, chunk_0))
            print(getsizeof(chunk_1) // (2 ** 10))  # debugging : get size of the chunk in Kilobytes. It also works as a progress update
            pd.DataFrame(chunk_1).to_csv(path_or_buf=descdocs_file, mode="a", header=False)
            logging.info("Chunk of documents created...")

    end_creatingInput = time.time()
    logging.info("Time spent creating the Documents: %s",
                        str(round(end_creatingInput - start_creatingInput, 3)) )
