import logging
import os
import shutil

import pandas as pd

######### Filesystem functions
from utilities import Filenames as F


def clean_directory(path_to_directory):
    if os.path.isdir(path_to_directory):
        for filename in os.listdir("" + path_to_directory):
            filepath = "" + path_to_directory + "/" + filename
            if os.path.isfile(filepath):
                logging.info("Going to clean the file: %s", str(filepath))
                os.unlink(filepath)
            elif os.path.isdir(filepath):
                logging.info("Going to remove the directory: %s", str(filepath))
                shutil.rmtree(filepath)

# Clean 'part' files from the appropriate directory :
def clean_partial_representations(parts_dir_path):
    paths_toremove = []
    for filename in os.listdir(parts_dir_path):
            if "part" in filename:
                paths_toremove.append(os.path.join(parts_dir_path,filename))
    for ptr in paths_toremove:
        logging.info("Going to clean the file: %s", str(ptr))
        os.remove(ptr)


def split_csv_file(tosplit_filepath, destination_folderpath, subfile_num_lines, newfiles_corename):

    input_file = open(tosplit_filepath, "r")
    if not os.path.exists(destination_folderpath):
        os.makedirs(destination_folderpath)
    for filep in os.listdir(destination_folderpath):
        os.remove(os.path.join(destination_folderpath,filep))

    subfile_paths = []
    current_subfile = 1
    for input_segment in pd.read_csv(input_file, chunksize=subfile_num_lines, sep="_"):
        subfile_path = os.path.join(destination_folderpath, newfiles_corename + str(current_subfile) + ".csv")
        subfile_paths.append(subfile_path)
        subfile = open(subfile_path,"w")
        pd.DataFrame(input_segment).to_csv(subfile, sep="_")
        subfile.close()
        current_subfile = current_subfile+1

    input_file.close()
    return subfile_paths


### get the length (number of lines) of a csv file
def get_csv_length(filepath):
    segment_size = 10**4 #use chunks to handle large files
    csv_num_lines = 0
    for segment in pd.read_csv(filepath, chunksize=segment_size, sep="_"):
        csv_num_lines = csv_num_lines + segment.shape[0]
    return csv_num_lines



##### Get the category directory paths when operating on the category datasets in OnlineLearning
def get_category_dirpaths():
    dirpaths = []
    dir_walk_generator = os.walk(F.ONLINE_QUESTIONS_CSVS_DIR)
    basedir_subdirs_files_t = dir_walk_generator.__next__()
    subdirs_ls = basedir_subdirs_files_t[1]
    for subdir_name in subdirs_ls:
        dirpaths.append(os.path.join(F.ONLINE_QUESTIONS_CSVS_DIR, subdir_name))
    return dirpaths