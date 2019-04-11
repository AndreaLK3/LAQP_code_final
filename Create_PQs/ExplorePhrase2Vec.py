import utilities.MyUtils as MyUtils
import Create_PQs.ReadMetadata as RM
import gensim.models.phrases as phrases
import Create_PQs.PreprocessDescriptions as PPD
import utilities.Filenames as F
import pandas as pd
import logging
import ast
import json
from gc import collect

#reads the log and stores in another file all the tokens that were joined
def writelog_toresults():
    phrases_set = set()
    with open(F.LOGFILENAME,"r") as logfile:
        with open(F.PHRASES_FILENAME,"w") as outfile:
            lines = logfile.readlines()
            for line in lines:
                wordslist = line.split() #split over whitespace
                for word in wordslist :
                    word.replace("[", "")
                    word.replace("]", "")
                    word.replace(",", "")
                    if "_" in word:
                        phrases_set.add(word)
            outfile.write(str(phrases_set))




def explore_phrase2vec(min_freq, phrases_threshold):
    MyUtils.init_logging("Explore_Phrase2Vec.log")
    words_lls = []
    doc_filenames = [F.DESCDOCS_RAW, F.QADOCS_RAW]
    doc_files = [open(doc_filename, "r") for doc_filename in doc_filenames]
    all_docwords = []
    chunk_size = 10**5
    for doc_file in doc_filenames:
        for docs_chunk in pd.read_csv(doc_file, chunksize=chunk_size):
            len_c = len(docs_chunk)
            words_chunk = []
            #indices = list(sorted(numpy.random.choice(len_c, int(docs_percent_touse * len_c), replace=False)))
            #selected_rows = docs_chunk.iloc[indices]
            for tupl in docs_chunk.itertuples():
                #words = tupl.words.replace("'",'"')
                #logging.info(words)
                #word_ls = json.loads(words)#ast.literal_eval(tupl.words)
                word_ls = eval(tupl.words, {'__builtins__': {}})
                words_chunk.append(word_ls)
            all_docwords.extend(words_chunk)
            logging.info("Added chunk from file %s to documents list...", doc_file)


    logging.info("Number of documents: %s", len(all_docwords))
    phrases_model = phrases.Phrases(sentences=all_docwords, min_count=min_freq, threshold=phrases_threshold, delimiter=b'_')
    #logging.info("***The Phrases model's frequency vocabulary: %s", str(phrases_model.vocab))
    phrases_vocab = phrases_model.vocab
    del phrases_model
    collect()
    sorted_vocabulary = sorted ( list(phrases_vocab.items()), key=lambda tpl: tpl[1], reverse=True)
    phrases_sorted_vocabulary = list( filter (lambda tpl: '_' in str(tpl[1]) , sorted_vocabulary) )
    individual_words_sorted_vocabulary = list( filter (lambda tpl: not('_' in str(tpl[1])) , sorted_vocabulary ))
    logging.info("***The vocabulary of phrases, ordered by frequency : %s ", phrases_sorted_vocabulary)
    logging.info("***The vocabulary of words, ordered by frequency : %s ", individual_words_sorted_vocabulary)
    #phrases_model.save("Exploration_phrasesModel_mincount"+ str(min_freq) + "_T"+str(phrases_threshold) + ".model")

    for i in range(len(words_lls)//4):
        print(str(phrases_model[words_lls[i]]))

    #writelog_toresults()

    #Let us test a new string ( e.g. an example title)
    #new_title = ["pulitzer", "prize", "new", "york", "united","states"]
    #print(str(phrases_model[new_title]))
    #new_title_2 = "polar bears advisory board".split()
    #print(str(phrases_model[new_title_2]))
    #new_title_3 = "arctic circle".split()
    #print(str(phrases_model[new_title_3]))
