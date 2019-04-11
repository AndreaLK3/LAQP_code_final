import utilities.Filenames as F
import sqlite3
import utilities.MyUtils as MyUtils
import pandas as pd
import logging
from time import time
from gc import collect
import os

def clean_representations_dbs(path_to_directory):
    if os.path.isdir(path_to_directory):
        for filename in os.listdir("" + path_to_directory):
            if "db" in filename:
                filepath = "" + path_to_directory + "/" + filename
                if os.path.isfile(filepath):
                    logging.info("Going to clean the file: %s", str(filepath))
                    os.unlink(filepath)

##### This may be needed only for the training dataset.
##### Objective: allowing to compute p-to-p and q-to-q similarity, using random access
def create_representations_db(represented_elements_filepath, outdb_filepath, id_column_name="id"):
    MyUtils.init_logging("CreatePs&Qs_RepresentationDatabases.log")

    elements_file = open(represented_elements_filepath, "r")
    f = open(outdb_filepath, "w")
    f.close() #clean outdb between runs

    segment_size = 2* 10**4
    segment_id = 1
    db_conn = sqlite3.connect(outdb_filepath)
    c = db_conn.cursor()

    for in_segment in pd.read_csv(elements_file, sep="_", chunksize=segment_size, dtype="str", quotechar='"'):
        #for tpl in in_segment.itertuples():
        #    logging.info(tpl)
        #    raise Exception
        start = time()

        tablename = "elements"+str(segment_id)

        in_segment.to_sql(tablename, db_conn, chunksize=10**4, if_exists='append', dtype={id_column_name:"varchar(63)"})
        collect()

        c.execute("CREATE INDEX indexid_"+str(segment_id)+" ON "+ tablename+" ("+id_column_name+");")
        logging.info("The segment n.%s, with %s represented elements, has been copied from final_file to database, and indexed; "+
                     "time elapsed: %s seconds",
                     segment_id, segment_size, round(time() - start, 3))
        segment_id = segment_id + 1
        db_conn.commit()
    db_conn.close()
