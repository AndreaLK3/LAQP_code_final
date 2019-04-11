import utilities.MyUtils as MyUtils
import Create_PQs.PreprocessDescriptions as PPD
import nltk
from gensim.models.doc2vec import TaggedDocument
import utilities.Filenames as F
import pandas as pd
from time import time
import logging

def load_info(postprocessed):
    if postprocessed == False:
        filename = F.PRODUCTS_METADATA_RAW
    else:
        filename =  F.PRODUCTS_METADATA
    prods = pd.read_csv(filename)
    #MyUtils.printAndLog(str(prod_dict_ls[0:5]))
    return prods



def createDocForTitle(title_text, stopwords_pattern):
    # Method2: preprocess and then use word_tokenize, that implicitly calls the sentence tokenizer
    title_0 = title_text.lower()
    title_1 = PPD.expandContractions(title_0)

    title_2 = stopwords_pattern.sub(repl=" ", string=title_1)  # stopwords removal step

    titleWords_1 = nltk.tokenize.word_tokenize(title_2)

    row_doc = TaggedDocument(words=titleWords_1, tags=[0]) #(id not used)

    return row_doc



def process_prodinfo(product, phrases_model, d2v_model, sw_pattern):
    #MyUtils.printAndLog(str(product))
    asin = product.asin
    price = float(product.price)
    description = str(product.description)
    # "brand" currently unused
    title = product.title
    if str(title) != "nan":
        title_doc = createDocForTitle(title_text=str(title), stopwords_pattern=sw_pattern)
        phrased_words=phrases_model[title_doc.words]
        titlevec = d2v_model.infer_vector(phrased_words)
    else:
        titlevec = "NOTITLEVEC"
    if str(product.categories) != "nan":
        categories_lls = eval(product.categories, {'__builtins__': {}})
        md_categories = MyUtils.flatten_lls(categories_lls)
    else:
        md_categories = []
        #MyUtils.printAndLog("Product processed")
    return (asin, description, price, titlevec, md_categories)



#n: Loads the necessary data. Saves the result to file, and also returns it
def process_all_mdinfo(prods_in_df, outfilepath, phrases_model, d2v_model):
    MyUtils.init_logging("ExtractMetadataInfo.log")

    f = open(outfilepath, "w") ; f.close() #clean between runs
    sw_pattern = PPD.getStopwordsPattern(includePunctuation=True)
    logging.info("Started postprocessing other metadata info")

    segment_nrows = 5 * 10**4
    logging.info("Number of elements in a segment: %s", str(segment_nrows))
    with open(outfilepath, "a") as out_file :
        out_file.write("_id_description_price_titlevec_mdcategories\n")
        for input_segment in pd.read_csv(prods_in_df, chunksize=segment_nrows, sep="_"):
            chunk_start = time()
            mdinfo_lts = []
            for prod_tupl in input_segment.itertuples():
                prodinfo_tuple = process_prodinfo(prod_tupl, phrases_model, d2v_model, sw_pattern)
                mdinfo_lts.append(prodinfo_tuple)
            pd.DataFrame(mdinfo_lts).to_csv(out_file, mode="a",header=False, sep="_")
            chunk_end  = time()
            logging.info("Processing: other metadata info. Segment completed in time : %s seconds", str(round(chunk_end - chunk_start,3)))
    logging.info("Completed: processing product metadata.")
