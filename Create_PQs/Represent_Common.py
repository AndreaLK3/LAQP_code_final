import Create_PQs.ReadQuestions as RQ
import Create_PQs.ReadMetadata as RM
import logging
import gensim.models.phrases as phrases
import pandas as pd
import utilities.Filenames as F
from time import time
import numpy
import ast
import gensim.models.doc2vec as D2V
import pympler.asizeof as mem
import utilities.MyUtils
import Create_PQs.VectorizeDescriptions as VD
import utilities.MyUtils as MyUtils
import os
import csv
import Create_PQs.PreprocessDescriptions as PPD
import Create_PQs.PreprocessQuestions as PPQ

from gc import collect

########## Functions to sort the output:
import utilities.MyUtils_filesystem
import utilities.MyUtils_flags


def wrapper_iteratornext(reader, num_lines_toreadover=1):
    for i in range(num_lines_toreadover-1):
        reader.__next__()
    try :
        return reader.__next__()
    except StopIteration:
        return "zz" # a "+inf" value for string comparison: 'z' is > than all uppercase letters and numbers
    except Exception as e:
        logging.warning("Warning: Unexpected exception while csv-reading: %s ", e)
        logging.warning("On reader: %s", reader)
        next_elem = reader.__next__()
        logging.warning("Moving on to next element: %s", next_elem)
        return next_elem



def dump_chunk(qs_out_ls, chunkfile_id, kind_stringflag, outfilepath):
    filename = outfilepath + "_part_" + str(chunkfile_id)
    chunk_outfile = open(filename, "w")
    if "questions" in kind_stringflag.lower():
        chunk_outfile.write("_id_questionType_questionVec_kwsVectors\n")
    if "products" in kind_stringflag.lower():
        chunk_outfile.write("_id_price_titlevec_descvec_mdcategories_kwsVectors\n")

    qs_out_ls_1 = list(filter(lambda elem: True if len(elem)>2 and len(elem[0])>=0 else False, qs_out_ls))
    if len(qs_out_ls_1) > len(qs_out_ls):
        logging.warning("Warning: malformed question dropped from the chunk to dump")
    ordered_qs_out_ls = sorted(qs_out_ls_1, key=lambda elem_t: elem_t[0])
    pd.DataFrame(ordered_qs_out_ls).to_csv(chunk_outfile, header=False, sep="_")
    chunk_outfile.close()

#The products-or-questions stringflag can be: "products", "questions"
#The dataset stringflag can be : "train", "valid", "test"
def reuniteandsort(ps_or_qs_stringflag, parts_dir_path, outfilepath):
    MyUtils.init_logging("Reunite.log")
    logging.info("Started reuniting and sorting the representations...")

    f = open(outfilepath, "w")
    f.close()  # clean outfile between runs
    united_outfile = open(outfilepath, "a")
    if "questions" in ps_or_qs_stringflag.lower():
        united_outfile.write("_id_questionType_questionVec_kwsVectors\n")
    if "products" in ps_or_qs_stringflag.lower():
        united_outfile.write("_id_price_titlevec_descvec_mdcategories_kwsVectors\n")
    out_buffer = []
    max_outbuffer_size = 10**4

    chunk_filenames_ls = []
    filehandlers_ls = []
    readers_ls = []
    for filename in os.listdir(parts_dir_path):
        ###counting on the fact that I clean all partial files with MyUtils_2.clean_partial_representations(parts_dir_path)
        ### after each execution
        # if "questions" in ps_or_qs_stringflag.lower():
        #     if "part" in filename:# and "question" in filename :#and dataset_stringflag.lower() in filename:
        #         chunk_filenames_ls.append(filename)
        # if "products" in ps_or_qs_stringflag.lower(): #and dataset_stringflag.lower() in filename:
        if "part" in filename:# and "product" in filename:
            chunk_filenames_ls.append(filename)
    logging.info("Chunk filenames: %s", chunk_filenames_ls)
    for chunk_filename in chunk_filenames_ls:
        filehandlers_ls.append(open(os.path.join(parts_dir_path,chunk_filename), "r", newline=''))
    for filehandler in filehandlers_ls:
        readers_ls.append(csv.reader(filehandler, delimiter='_', quotechar='"'))

    logging.info(list(zip(chunk_filenames_ls,readers_ls)))
    elements = [wrapper_iteratornext(reader,2) for reader in readers_ls]

    while(elements != ["zz" for i in range(len(readers_ls))]):
        #logging.info(elements)
        q_ids = [element[1] for element in elements]
        #logging.info(q_ids)
        ordered_indices = numpy.argsort(q_ids)
        first_index =ordered_indices[0]
        element_to_write = elements[first_index]
        new_elem = wrapper_iteratornext(readers_ls[first_index])
        elements[first_index] = new_elem
        #logging.info("The element to write: %s", element_to_write)
        if len(element_to_write[1]) >= 10:  # id check ; skip headers with id = 0 or 'id'
            out_buffer.append(element_to_write[1:])
        if len(out_buffer) >= max_outbuffer_size:
            logging.info("Ids: %s", q_ids)
            out_buffer = list(filter( lambda elem: len(elem[1])>0, out_buffer))
            pd.DataFrame(out_buffer).to_csv(united_outfile, mode="a", header=False, sep="_")
            out_buffer = []
    #last buffer chunk:
    pd.DataFrame(out_buffer).to_csv(united_outfile, mode="a", header=False, sep="_")

    utilities.MyUtils_filesystem.clean_partial_representations(parts_dir_path)
    clean_0_ids(outfilepath, ps_or_qs_stringflag)




def clean_0_ids(outfile_tofilter_path, ps_or_qs_stringflag):

    MyUtils.init_logging("temp.log")
    logging.info(outfile_tofilter_path)
    new_outfile_path = F.ELEMENTS_TEMP
    f = open(new_outfile_path, 'w'); f.close()
    new_outfile = open(new_outfile_path,'a')

    if utilities.MyUtils_flags.FLAG_QUESTS in ps_or_qs_stringflag or utilities.MyUtils_flags.FLAG_QUESTIONS in ps_or_qs_stringflag:
    #     new_outfile.write("_id_questionType_questionVec_kwsVectors\n")
    #     logging.info("Writing: _id_questionType_questionVec_kwsVectors", )
        the_header = ["id","questionType","questionVec","kwsVectors"] #or the_header?
    if utilities.MyUtils_flags.FLAG_PRODS in ps_or_qs_stringflag or utilities.MyUtils_flags.FLAG_PRODUCTS in ps_or_qs_stringflag:
    #     new_outfile.write("_id_price_titlevec_descvec_mdcategories_kwsVectors\n")
    #     logging.info("Writing: _id_price_titlevec_descvec_mdcategories_kwsVectors", )
        the_header = ["id", "price", "titlevec", "descvec", "mdcategories", "kwsVectors"]

    segment_size = 5 * 10**4
    segment_counter = 1
    for segment in pd.read_csv(outfile_tofilter_path, chunksize=segment_size, sep='_'):
        segment_buffer = []
        for elem_tuple in segment.itertuples():
            if len(elem_tuple.id)>=9:#.id
                segment_buffer.append(elem_tuple)
            else:
                logging.info(elem_tuple.id)
        #temp_columns = ['_4','_5','_6','NODESCVEC','_8','NOKWSVECTORS']
        #segment_header = True if segment_counter==1 else False
        segment_df = pd.DataFrame(segment_buffer)[the_header]
        segment_df.columns = the_header
        segment_df.to_csv(new_outfile_path,mode="a", header=bool(segment_counter==1), sep="_", index=False)
        logging.info("Filtered '0' ids from segment n. %s...", segment_counter)
        segment_counter = segment_counter+1
        collect()

    os.rename(src=new_outfile_path, dst=outfile_tofilter_path)


def order_test(finalfile_path):
    MyUtils.init_logging("temp.log")
    segment_size = 10 ** 4
    segment_counter = 1
    for segment in pd.read_csv(finalfile_path, chunksize=segment_size, sep='_'):
        #for tpl in segment.itertuples():
        #    logging.info(tpl)
        asins = segment.id
        ordered = utilities.MyUtils.check_series_ids_sorted(asins, len(asins))
        logging.info("Is the chunks of elements n.%s ordered?: %s",
                     segment_counter,ordered)
        #if not ordered:
        #    logging.info(asins)
        segment_counter = segment_counter+1
        collect()


##########


########## Functions to implement some of the steps of the encoding process
def prepare_datasets():
    #tracker_1 = SummaryTracker()
    md_df = RM.readfirst_md()
    RM.organize_md(md_df)
    del md_df
    RQ.readfirst_qa()
    RQ.organize_qa_all()
    collect() #manual garbage collection
    #logging.info(tracker_1.print_diff())


def create_the_models():
    MyUtils.init_logging("Encode_Common.log")
    PPD.createDescriptionDocuments()  # creates and saves version 1.0 of the docs, before phrases
    PPQ.createQuestionDocuments()
    collect()  # manual garbage collection

    create_phrases_model()
    prepare_dq_documents()  # v.1.1 of the docs, after phrases

    collect()  # manual garbage collection
    VD.create_docvectors_model()  # the Doc2Vec model, with the vectors of the training subset
    collect()

    d2v_model = VD.load_model()
    logging.info("d2v_model, memory size in MBs = %s", str(mem.asizeof(d2v_model) // 2 ** 20))
    phrases_model = phrases.Phrases.load(F.PHRASES_MODEL)
    logging.info("phrases_model, memory size in MBs = %s" , str(mem.asizeof(phrases_model) // 2 ** 20))

    logging.info("Doc2Vec and Phrases models loaded.")
    return (d2v_model, phrases_model)


def load_the_models():
    d2v_model = VD.load_model()
    phrases_model = phrases.Phrases.load(F.PHRASES_MODEL)
    return (d2v_model, phrases_model)


##### Functions to create the Phrases model from both descriptions and questions, #####
##### the Phrases model is then used to update the documents
def create_phrases_model():
    MyUtils.init_logging("Encode_Common.log")
    logging.info("Starting preparation of phrases...")
    docs_percent_touse = 1 #0.5.
    chunk_size = 10**5

    doc_filenames = [F.DESCDOCS_RAW, F.QADOCS_RAW]
    doc_files = [open(doc_filename,"r") for doc_filename in doc_filenames]
    all_docwords = []
    for doc_file in doc_filenames:
        for docs_chunk in pd.read_csv(doc_file, chunksize=chunk_size):
            len_c = len(docs_chunk)
            words_chunk = []
            indices = list(sorted(numpy.random.choice(len_c, int(docs_percent_touse * len_c) , replace=False)) )
            selected_rows = docs_chunk.iloc[indices]
            for tupl in selected_rows.itertuples():
                word_ls = ast.literal_eval(tupl.words)
                words_chunk.append(word_ls)
            all_docwords.extend(words_chunk)
            logging.info("Reading in the documents' words. Chunk processed...")
        logging.info("Completed: reading in a set of documents' words")  # @ time = " + str(round(time1 - start, 3)))

    logging.info("Number of documents to use in the Phrases model: %s", str(len(all_docwords)))
    del doc_filenames;del doc_files; collect()

    phrases_model = phrases.Phrases(sentences=all_docwords, min_count=20, threshold=300, delimiter=b'_', max_vocab_size=30*10**6)
    #phraser_model = phrases.Phraser(phrases_model)
    #time2 = time();
    logging.info("Phrases model created") #@ time = " + str(round(time2 - start, 3)))
    logging.info("Memory size in MBs = %s", str(mem.asizeof(phrases_model) // 2 ** 20))

    phrases_model.save(F.PHRASES_MODEL)

    return phrases_model



def prepare_dq_documents():
    MyUtils.init_logging("Encode_Common.log")
    start = time()
    phrases_model = phrases.Phrases.load(F.PHRASES_MODEL)
    logging.info("Started updating the TaggedDocuments according to the Phrases model...")
    doc_filenames = [(F.DESCDOCS_RAW, F.DESCDOCS), (F.QADOCS_RAW, F.QADOCS)]

    for tupl_fn in doc_filenames:
        input_filename = tupl_fn[0]; output_filename = tupl_fn[1]
        f = open(output_filename, "w"); f.close()  # clean output file between runs

        with open(output_filename, "a") as newdocs_file:
            newdocs_file.write(",words,tags\n")

            with open(input_filename, "r") as rawdocs_file:
                chunk_n_elems = 10**5
                for segment in pd.read_csv(rawdocs_file, chunksize=chunk_n_elems):
                    new_docs_chunk = []
                    for tupl in segment.itertuples():
                        try :
                            old_words = ast.literal_eval(tupl.words) #evaluates the string back into a list
                            new_words = phrases_model[old_words]
                            new_docs_chunk.append(D2V.TaggedDocument(words=new_words, tags=tupl.tags))
                        except ValueError:
                            logging.warning("Info: literal evaluation did not apply to element: %s" , str(tupl.tags))
                    pd.DataFrame(new_docs_chunk).to_csv(newdocs_file, mode="a", header=False)
                    logging.info("Documents updated with phrases: a chunk has been processed")
        logging.info("Completed: a set of documents has been updated with Phrases")

    time3 = time() ;  logging.info("New documents created, in time = %s", str(round(time3 - start, 3)))