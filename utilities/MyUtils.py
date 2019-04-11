import collections

import numpy
import itertools
from collections import namedtuple
import logging
import sys


###########log file
def init_logging(logfilename, loglevel=logging.INFO):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(level=loglevel, filename=logfilename, filemode="w",
                        format='%(asctime)s -%(levelname)s : %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
    # print(logging.getLogger())
    if len(logging.getLogger().handlers) < 2:
        outlog_h = logging.StreamHandler(sys.stdout)
        outlog_h.setLevel(loglevel)
        logging.getLogger().addHandler(outlog_h)



########### other utilities ###########

def pickRandomElement(metadata_df, max_index=1000):
    chosen_index = numpy.random.randint(0, max_index)
    elem = metadata_df.loc[chosen_index]
    return elem

def flatten_lls(lls):
    ls = [val for sublist in lls for val in sublist]
    return ls


#Returns a list of iterators over the slices
def split_dict_in_chunks(whole_dict, num_chunks):
    chunk_size = len(whole_dict) // num_chunks
    #print("Elems in a chunk :" + str(chunk_size)) ;  print("Elems in the dict : " + str(len(whole_dict)) )
    pointer = 0
    iterators_ls = []
    while pointer < len(whole_dict):
        #print(pointer)
        chunk_iter = itertools.islice(whole_dict, pointer, pointer + chunk_size)
        iterators_ls.append(chunk_iter)
        pointer = pointer + chunk_size
    return iterators_ls


def refine_tuple(element_tuple):
    attributes = []
    values = []
    for name,value in element_tuple._asdict().items():
        if "_" not in name:
            attributes.append(name)
            values.append(value)
    elem_newformat = namedtuple('elem_newformat', attributes)
    elem_newtuple = elem_newformat._make(values)
    return elem_newtuple

########## Functions to transform a p/q tuple into a namedtuple
def quest_lstonamedtuple(quest_ls, offset=1):
    Question = collections.namedtuple('Question', 'id questionType questionVec kwsVectors')
    q_t = Question( id=quest_ls[offset], questionType=quest_ls[offset+1], questionVec=quest_ls[offset+2], kwsVectors=quest_ls[offset+3])
    return q_t


def prodls_tonamedtuple(prod_ls, offset=1):
    #logging.info(prod_ls) #'', 'id', 'price', 'titlevec', 'descvec', 'mdcategories', 'kwsVectors'
    Product = namedtuple('Product', 'id price titlevec descvec mdcategories kwsVectors')
    p_t = Product(id=prod_ls[offset], price=prod_ls[offset+1], titlevec=prod_ls[offset+2],
                  descvec=prod_ls[offset+3], mdcategories=prod_ls[offset+4], kwsVectors=prod_ls[offset+5])
    return p_t

######### Exponential smoothing of an array.
######### Works in the same way as the Smoothing in Tensorboard (alpha \in (0,1); alpha -> 1 ==> more smoothing )
def smooth_array_exp(array, alpha=0.5):
    smoothed_array = []
    smoothed_value = 0
    for i in range(len(array)):
        if i == 0:
            smoothed_value = array[i]
        else:
            smoothed_value = alpha * smoothed_value + (1-alpha)*array[i]
        smoothed_array.append(smoothed_value)
    return smoothed_array


#### Given a matrix with cosine similarity between pairs of elements,
#### pick the best matches over the rows (or columns, if fewer)
def pick_maxmatches_matrix(matrix):
    (rows,cols) = matrix.shape
    if rows <= cols:
        return numpy.amax(matrix, axis=1)
    else:
        return numpy.amax(matrix, axis=0)


def check_series_ids_sorted(series, series_length):
    ordered= True
    for i in range(1,series_length-1):
        if str((series.iloc[i])[0:10]) <= str((series.iloc[i + 1])[0:10]):
            pass
        else:
            logging.info("Not ordered: %s and %s", series.iloc[i], series.iloc[i + 1])
            ordered=False
    return ordered