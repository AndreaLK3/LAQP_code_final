import pandas as pd
import gzip
import sys
import os
sys.path.append(os.path.abspath('..'))
import utilities.MyUtils as MyUtils
import utilities.Filenames as F
import time
import os
import numpy
import logging

READKEYWORD_TRAIN = "train"
READKEYWORD_TRAINSUBSET = "trainsubset"
READKEYWORD_VALIDATION = "validation"
READKEYWORD_TEST = "test"


#Input: a list of strings (one for each line in the metadata)
#Output: a list of dictionaries
def create_dict_from_data(chunk):
    prodDictionaries_ls = []
    for l in range(len(chunk)):
        obj = eval(chunk[l], {'__builtins__': {}})
        prodDictionaries_ls.append(obj)
    #MyUtils.writeToLog(str(objs[0]))
    return prodDictionaries_ls


#Reads the json.gz file and transforms it into a .csv containing product dictionaries
#Time spent: 328.16s
def readfirst_md():
    MyUtils.init_logging("ReadMetadata.log")
    startRead = time.time()

    chunks_dfs = []

    with gzip.open(F.METADATA_FILEPATH, 'rb') as metadata_file: #use gzip.open and rb if the file is compressed

        tot_filesize = os.path.getsize(F.METADATA_FILEPATH)
        #MyUtils.printAndLog(str(tot_filesize))
        chunk_bytesize = 8 * (2 ** 20)

        num_of_chunks = tot_filesize // chunk_bytesize +1
        logging.info("Number of chunks = %s", str(num_of_chunks))
        for c in range(num_of_chunks):
            chunk = metadata_file.readlines(chunk_bytesize) #returns a list of strings, one for each line
            small_df = pd.DataFrame(create_dict_from_data(chunk))
            small_df = small_df.set_index(keys="asin")  # This sets the 'asin' as the index (n: but also drops the column)
            chunks_dfs.append( small_df )
            if c % (num_of_chunks // 10) == 0 :
                logging.info("Reading the metadata dataset... %s completed",str( round( (c / num_of_chunks )*100 , 2)) )

    metadata_df = pd.concat(chunks_dfs)
    #logging.info("%s", str(metadata_df.iloc[[0, 1, 2, 3, 4], :]))

    logging.info("Saving the metadata Dataframe")
    md_dataframe_file = open(F.MD_DATAFRAME_FILEPATH, "w")
    metadata_df.to_csv(md_dataframe_file, sep="_")
    md_dataframe_file.close()

    endRead = time.time()
    theTime = endRead - startRead
    logging.info("Total time spent = %s", str(theTime))

    return metadata_df



# Separate and save the following random subsets of the metadata:
# 1)Training  2)Training subset  3)Validation  4)Test
# Time spent:652.85s (due to overflowing the 12GB RAM and having to use around 2GB of swap)
def organize_md(metadata_df):
    MyUtils.init_logging("ReadMetadata.log")

    start = time.time()

    logging.info("%s", str(metadata_df.shape))
    tot_rows = metadata_df.shape[0]

    training_fraction = 0.8

    training_indices_set = set(numpy.random.choice(tot_rows, int(training_fraction * tot_rows) , replace=False))
    training_indices = sorted(list(training_indices_set))
    training_df = metadata_df.iloc[training_indices]
    train_file = open(F.TRAIN_MD_DF_FILEPATH, "w")
    training_df.to_csv(train_file, sep="_")
    train_file.close()
    del training_df
    logging.info("Metadata : training set organized.")

    training_subset_indices = sorted(numpy.random.choice(training_indices, int(0.2 * len(training_indices)), replace=False))
    training_subset_df = metadata_df.iloc[training_subset_indices]
    traning_subset_file = open(F.TRAINSUBSET_MD_DF_FILEPATH, "w")
    training_subset_df.to_csv(traning_subset_file, sep="_")
    traning_subset_file.close()
    del training_subset_df
    logging.info("Metadata : training subset organized.")

    remaining_indices_set = set(list(range(0,tot_rows))) - training_indices_set
    remaining_indices_ls = list(remaining_indices_set)
    validation_indices_set = set(numpy.random.choice(remaining_indices_ls, int(0.5 * len(remaining_indices_ls)) , replace=False))
    validation_indices = sorted(list(validation_indices_set))
    validation_df = metadata_df.iloc[validation_indices]
    valid_file = open(F.VALID_MD_DF_FILEPATH, "w")
    validation_df.to_csv(valid_file, sep="_")
    valid_file.close()
    del validation_df
    logging.info("Metadata : validation set organized.")


    test_indices_set = remaining_indices_set - validation_indices_set
    test_indices = sorted(list(test_indices_set))
    test_df = metadata_df.iloc[test_indices]
    test_file = open(F.TEST_MD_DF_FILEPATH, "w")
    test_df.to_csv(test_file, sep="_")
    test_file.close()
    del test_df
    logging.info("Metadata : test set organized.")

    end = time.time()
    logging.info("Time spent organizing the metadata training, validation and test subsets: %s", str(round(end - start, 3)))


#The values for keyword can be: "train", "trainsubset", "validation", "test"
def load_md(keyword):
    path = ""
    if keyword == "train":
        path = F.TRAIN_MD_DF_FILEPATH
    elif keyword == "trainsubset":
        path = F.TRAINSUBSET_MD_DF_FILEPATH
    elif keyword == "validation":
        path = F.VALID_MD_DF_FILEPATH
    elif keyword == "test":
        path = F.TEST_MD_DF_FILEPATH

    startLoad = time.time()
    metadata_df = pd.read_csv(path, sep="_", engine='c')
    endLoad = time.time()
    logging.info("Time spent loading the file: %s", str(round(endLoad - startLoad, 3)))
    return metadata_df



if __name__ == "__main__":
    readfirst_md()
    organize_md()