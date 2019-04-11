import re


#### String adjustments
from nltk import tokenize


def remove_string_end(s, end):
    if s.endswith(end):
        s1 = s[:-(len(end))]
        return s1
    else:
        return s

def remove_string_start(s, start):
    if s.startswith(start):
        s1 = s[len(start):]
        return s1
    else:
        return s

def polish_strings(text_ls):
    return (list(map(lambda text: polish_string(text), text_ls)))


def polish_string(text):
    tokens = text.split()
    words = ' '.join(tokens)
    return words
#####


########## Functions to read the string representations of numpy arrays

def fromstring_toarray(vector_string, whitespaces_to_commas=True):
    s1 = (vector_string).replace("[ ", "[")
    whitespaces_pt = re.compile(r'(\s)+')
    if whitespaces_to_commas:
        s2 = re.sub(pattern=whitespaces_pt, repl=", ", string=s1)
    else:
        s2 = s1
    arr = eval(s2, {'__builtins__': {}})#arr = ast.literal_eval(s2)
    return arr


def fromlls_toarrays(vectors_lls_string):
    vectors = []
    matches_iter = re.finditer('\[([^()])+\]', vectors_lls_string)
    for match in matches_iter:
        vector_str = match.group(0)
        vector = eval(vector_str, {'__builtins__': {}})#vector = ast.literal_eval(vector_str)
        vectors.append(vector)
    #logging.debug("fromlls_toarrays vectors: %s", vectors)
    return vectors


#### Read a string representation of a lls of categories; transform it into a list of Doc2Vec vectors
def categories_to_vecs_lls(categories_lls, d2v_model):
    punct_pattern = re.compile(r'([!"#$%&()*+,./:;<=>?@\[\\\]^_`{|}-~\'])|([--])')
    categories_vecs_lls = []
    categories_vecs_ls = []
    for categ_ls in categories_lls:
        #logging.debug("Categories for a product: %s",categ_ls)
        if type(categ_ls) is list:
            for categ in categ_ls:
                #logging.info("\tExamining: %s", categ)
                categ_1 = categ.lower()
                categ_2 = tokenize.word_tokenize(categ_1)
                categ_3 = list(filter(lambda w: True if (bool(punct_pattern.match(w)) == False) else False, categ_2))
                #logging.info("\tTokenized category words: %s", categ_3)
                categ_vec = d2v_model.infer_vector(categ_3)
                categories_vecs_ls.append(categ_vec)
        categories_vecs_lls.append(categories_vecs_ls)
        #logging.info(np.array(categories_vecs_lls).shape)
    return categories_vecs_lls