import pandas
import utilities.Filenames as F
import gensim.models.doc2vec as D2V
import utilities.MyUtils as MyUtils
import logging

def getSingletons(dictTuple):
    if dictTuple[1] == 1:
        return True
    else:
        return False

def getURLs(dictTuple):
    if dictTuple[0].find("www") != -1 or dictTuple[0].find("http") != -1 or dictTuple[0].find(".com") != -1:
        return True
    else:
        return False


def explore():
    MyUtils.init_logging("ExploreStopwordsMethods.log")
    descDocs_ls = list(pandas.read_csv(F.DESCDOCS_RAW))[0:10]

    print( (descDocs_ls[0:5]) ) #exploration & debug
    model_forVocabulary = D2V.Doc2Vec()
    model_forVocabulary.build_vocab(sentences=descDocs_ls, update=False, progress_per=1000, keep_raw_vocab=True, trim_rule=None)


    #convert the vocabulary dictionary into a list for ordering
    vocab_list = [(k,v) for k,v in model_forVocabulary.raw_vocab.items()]
    vocab_list.sort(key= lambda tuple : tuple[1], reverse=True) #sorted in place, descendingly, depending on the value
    logging.info("Length of the whole vocabulary : " + str(len(vocab_list)))
    logging.info(str(vocab_list[0:400]))
    pandas.DataFrame(vocab_list[0:400]).to_csv("stopwords/wordFrequencies_ls.csv")

    singletons_vocab_list = list( filter(lambda tupl : getSingletons(tupl), vocab_list) )
    logging.info("Number of singletons : " + str(len(singletons_vocab_list)))
    logging.info(str(singletons_vocab_list[0:1000]))

    urls_vocab_list = list(filter(lambda tupl: getURLs(tupl), vocab_list))
    logging.info("Number of URLs : " + str(len(urls_vocab_list)))
    logging.info(str(urls_vocab_list[0:1000]))


    wordDocFrequency_dict = dict.fromkeys(model_forVocabulary.raw_vocab.keys(),0)

    for taggedDocument in descDocs_ls:
        already_encountered = []
        words_ls = taggedDocument.words
        for word in words_ls:
            if word not in already_encountered:
                wordDocFrequency_dict[word] = wordDocFrequency_dict[word]+1
                already_encountered.append(word)

    # It would be log (N / df(w)). For ordering purposes, (N / df(w)) suffices, or even (1 / df(w))
    # Therefore, to pick the words with lowest IDF we must pick those with a higher df(w)
    docFreq_list = [(k, v) for k, v in wordDocFrequency_dict.items()]
    docFreq_list.sort(key=lambda tuple: tuple[1], reverse=True)  # sorted in place, descendingly, depending on the value
    logging.info("The Doc-frequencies of the words have been determined.")
    logging.info(str(docFreq_list[0:400]))
    pandas.DataFrame(docFreq_list[0:400]).to_csv("stopwords/docFrequencies_ls.csv")


