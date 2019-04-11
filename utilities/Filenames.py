import os.path

##### First part: create the representations

### ReadMetadata
METADATA_FILEPATH = os.path.normpath(os.path.join('datasets','metadata','metadata.json.gz'))  #'metadata.json.gz.001'
MD_DATAFRAME_FILEPATH = os.path.normpath(os.path.join('datasets','metadata','md_df.csv'))     #'md_df_001.csv'
TRAIN_MD_DF_FILEPATH = os.path.normpath(os.path.join('datasets','metadata','train_md_df.csv'))
VALID_MD_DF_FILEPATH = os.path.normpath(os.path.join('datasets','metadata','valid_md_df.csv'))
TEST_MD_DF_FILEPATH = os.path.normpath(os.path.join('datasets','metadata','test_md_df.csv'))
TRAINSUBSET_MD_DF_FILEPATH = os.path.normpath(os.path.join('datasets','metadata','trainsubset_md_df.csv'))

TRAIN_MD_DF_DB = os.path.normpath(os.path.join('Ranking','Test_elements_databases','train_products.db'))
TEST_MD_DF_DB = os.path.normpath(os.path.join('Ranking','Test_elements_databases','test_products.db'))

 
#ReadQuestions
QA_DIR_PATH = os.path.normpath(os.path.join('datasets','questionsAndAnswers'))
QA_OUTFILE_CORENAME = 'QA_DATASET'
QA_DBS_DIRNAME = "qa_databases"

QA_TRAIN_DB =  os.path.normpath(os.path.join('Ranking','Test_elements_databases','train_questions.db'))
QA_TEST_DB =  os.path.normpath(os.path.join('Ranking','Test_elements_databases','test_questions.db'))


#PreprocessDescriptions
DESCDOCS_RAW = os.path.normpath(os.path.join('Create_PQs','descriptions_documents_raw.csv'))
DESCDOCS = os.path.normpath(os.path.join('Create_PQs','descriptions_documents.csv'))
 
WORDFREQ_STOPWORDS_FILEPATH = os.path.normpath(os.path.join('Create_PQs', 'stopwords','wordFrequencies_adjusted_ls.csv'))
DOCFREQ_STOPWORDS_FILEPATH = os.path.normpath(os.path.join('Create_PQs','stopwords','docFrequencies_adjusted_ls.csv'))
STOPWORDS_OUTFILEPATH = os.path.normpath(os.path.join('Create_PQs','stopwords','myStopwordsList.pickle'))
STOPWORDS_CSV_OUTFILEPATH = os.path.normpath(os.path.join('Create_PQs','stopwords','myStopwordsList.csv'))
 
#PreprocessQuestions
QADOCS_RAW = os.path.normpath(os.path.join('Create_PQs','questions_documents_raw.csv'))
QADOCS = os.path.normpath(os.path.join('Create_PQs','questions_documents.csv'))

#ExtractMetadataInfo
PRODUCTS_METADATA_RAW = os.path.normpath(os.path.join('Create_PQs','product_dictionaries_raw.csv'))
PRODUCTS_METADATA = os.path.normpath(os.path.join('Create_PQs','product_dictionaries.csv'))

#ExplorePhrase2vec
LOGFILENAME = os.path.normpath(os.path.join('PF.log'))
PHRASES_FILENAME = os.path.normpath(os.path.join('Create_PQs','ExplorePhrases.txt'))

#ExtractHTMLInfo
HTML_OUTFILE = os.path.normpath(os.path.join('Create_PQs','products_htmlinfo.pickle'))

#VectorizeDescriptions
D2V_MODEL = os.path.normpath(os.path.join('gensim_models','doc2VecTrainedmodel.model'))
D2V_VOCABULARY = os.path.normpath(os.path.join('gensim_models','doc2Vec_modelForVocabulary.model'))


#RepresentProducts
PHRASES_MODEL = os.path.normpath(os.path.join('gensim_models','phrases.model'))
PRODUCTS_FINAL_TRAIN = os.path.normpath(os.path.join('Create_PQs', 'Results_entities', 'encoded_products_train.csv'))
PRODUCTS_FINAL_VALID = os.path.normpath(os.path.join('Create_PQs', 'Results_entities', 'encoded_products_validation.csv'))
PRODUCTS_FINAL_TEST = os.path.normpath(os.path.join('Create_PQs', 'Results_entities', 'encoded_products_test.csv'))
ELEMENTS_TEMP =  os.path.normpath(os.path.join('Create_PQs', 'Results_entities', 'temp_elements.csv'))

#RepresentQuestions
QUESTIONS_FINAL_TRAIN = os.path.normpath(os.path.join('Create_PQs', 'Results_entities', 'encoded_questions_train.csv'))
QUESTIONS_FINAL_VALID = os.path.normpath(os.path.join('Create_PQs', 'Results_entities', 'encoded_questions_validation.csv'))
QUESTIONS_FINAL_TEST = os.path.normpath(os.path.join('Create_PQs', 'Results_entities', 'encoded_questions_test.csv'))
QUESTIONS_TEMP = os.path.normpath(os.path.join('Create_PQs','Results_entities','encoded_questions_noduplicates.csv'))

#MyRAKE
DESCS_KEYWORDS_RAW = os.path.normpath(os.path.join('Create_PQs','keywords','desc_kws_raw.csv'))
QUESTIONS_KEYWORDS_RAW = os.path.normpath(os.path.join('Create_PQs','keywords','quest_kws_raw.csv'))
DESC_KWSVECS = os.path.normpath(os.path.join('Create_PQs','keywords','desc_kws_vectorized.csv'))
QUEST_KWSVECS = os.path.normpath(os.path.join('Create_PQs','keywords','quest_kws_vectorized.csv'))


#RepresentationsDatabases
PRODUCTS_FINAL_TRAIN_DBNAME = 'encoded_products_train.db'
PRODUCTS_FINAL_TRAIN_DB = os.path.normpath(os.path.join('Create_PQs', 'Results_entities', 'Databases_representations', PRODUCTS_FINAL_TRAIN_DBNAME))
PRODUCTS_FINAL_VALID_DB = os.path.normpath(os.path.join('Create_PQs', 'Results_entities', 'Databases_representations', 'encoded_products_valid.db'))
PRODUCTS_FINAL_TEST_DB = os.path.normpath(os.path.join('Create_PQs', 'Results_entities', 'Databases_representations', 'encoded_products_test.db'))

QUESTIONS_FINAL_TRAIN_DBNAME = 'encoded_questions_train.db'
QUESTIONS_FINAL_TRAIN_DB = os.path.normpath(os.path.join('Create_PQs', 'Results_entities', 'Databases_representations', QUESTIONS_FINAL_TRAIN_DBNAME))
QUESTIONS_FINAL_VALID_DB = os.path.normpath(os.path.join('Create_PQs', 'Results_entities', 'Databases_representations', 'encoded_questions_valid.db'))
QUESTIONS_FINAL_TEST_DB = os.path.normpath(os.path.join('Create_PQs', 'Results_entities', 'Databases_representations', 'encoded_questions_test.db'))

#### Second part : Defining the Candidates

# Product Similarity
SIMILARITY_PRODUCTS_DB = os.path.normpath(os.path.join('AssociationNN','Similarity', 'ProductSimilarity.db'))

#NN_Datasets_Instances
CATEGORIES_DICTIONARY = os.path.normpath(os.path.join('AssociationNN','encodedcategories_dict.pickle'))

PRODSWITHQUESTS_IDS = os.path.normpath(os.path.join('AssociationNN', 'prods&askedqs_ids.csv'))
PRODSWITHQUESTS_IDS_TEMP = os.path.normpath(os.path.join('AssociationNN','prods&askedqs_ids_temp.csv'))
PRODSWITHQUESTS_IDS_ALL = os.path.normpath(os.path.join('AssociationNN','prods&askedqs_ids_all_'))# + dataset_typeflag

PRODS_NEGATIVEINDICES = os.path.normpath(os.path.join('AssociationNN','prods_negativeindices.pickle'))
POSITIVEQUESTS_IDS = os.path.normpath(os.path.join('AssociationNN','qs_positiveexamples_ids.csv'))
NEGATIVEQUESTS_IDS = os.path.normpath(os.path.join('AssociationNN','qs_negativeexamples_ids.csv'))
PRODS_WITH_NOTASKEDQUESTS_IDS = os.path.normpath(os.path.join('AssociationNN','prods&notaskedqs_ids.csv'))

#Databases for the NN
CANDIDATE_NEGQS_DB = os.path.normpath(os.path.join('AssociationNN','Databases','CandidateNegativeQs.db'))

#Databases for the NN : Numerical encoding of products and questions
PRODS_NUMENCODING_DB_TRAIN = os.path.normpath(os.path.join('AssociationNN', 'Databases', 'training', 'Ps_NumericalEncoding.db'))
PRODS_NUMENCODING_DB_VALID = os.path.normpath(os.path.join('AssociationNN', 'Databases', 'validation', 'Ps_NumericalEncoding.db'))
PRODS_NUMENCODING_DB_TEST = os.path.normpath(os.path.join('AssociationNN', 'Databases', 'test', 'Ps_NumericalEncoding.db'))
QUESTS_NUMENCODING_DB_TRAIN = os.path.normpath(os.path.join('AssociationNN', 'Databases', 'training', 'Qs_NumericalEncoding.db'))
QUESTS_NUMENCODING_DB_VALID = os.path.normpath(os.path.join('AssociationNN', 'Databases', 'validation', 'Qs_NumericalEncoding.db'))
QUESTS_NUMENCODING_DB_TEST = os.path.normpath(os.path.join('AssociationNN', 'Databases', 'test', 'Qs_NumericalEncoding.db'))

#Databases for the NN : NN input datasets
NN_TEMP_INSTANCES_DB = os.path.normpath(os.path.join('AssociationNN', 'Databases', 'training', 'AssociationNN_temp_instances.db'))
NN_TRAIN_INSTANCES_DB = os.path.normpath(os.path.join('AssociationNN', 'Databases', 'training', 'AssociationNN_traininstances.db'))
NN_VALID_INSTANCES_DB = os.path.normpath(os.path.join('AssociationNN', 'Databases', 'validation', 'AssociationNN_validationinstances.db'))
NN_TEST_INSTANCES_DB = os.path.normpath(os.path.join('AssociationNN', 'Databases', 'test', 'AssociationNN_testinstances.db'))

#NN_Network Results
TENSORBOARD_ANN_DIR = os.path.normpath(os.path.join('AssociationNN','summaries_dir_ANN'))


#OnlineLearning

ONLINE_PQMATCHES = os.path.normpath(os.path.join('OnlineLearning', 'PQs_matches_ids_beforefilter.csv'))
ONLINE_PQMATCHES_FILTERED = os.path.normpath(os.path.join('OnlineLearning', 'PQs_matches_ids.csv'))
ONLINE_NEGINDICES_LTS = os.path.normpath(os.path.join('OnlineLearning', 'negative_qs_lts_coordinates.csv'))

ONLINE_INSTANCEIDS_GLOBAL_DB = os.path.normpath(os.path.join('OnlineLearning', 'BalancedDataset','Global','global_training_instances.db'))
ONLINE_TEMP_DB = os.path.normpath(os.path.join('OnlineLearning', 'temp.db'))

COSINE_SIM_THRESHOLDS_DB = os.path.normpath(os.path.join('OnlineLearning', 'CosineSimilarity', 'cosinesim_thresholds.db'))

#OnlineLearning - unbalanced category-based instances
ONLINE_QUESTIONS_CSVS_DIR = os.path.normpath(os.path.join('OnlineLearning', 'CategoryDatasets'))


##### Third part: Ranking

#Operating on the test dataset to obtain the Candidates
SAVED_NN = os.path.normpath(os.path.join('Ranking','NN', 'saved_neural_network'))
NN_TEST_OUTPUT_DIR = os.path.normpath(os.path.join('Ranking','NN', 'Results_testset'))
CANDIDATES_NN_DB = os.path.normpath(os.path.join('Ranking', 'candidates_ids_nn.db'))
CANDIDATES_NN_DB_RANKED = os.path.normpath(os.path.join('Ranking', 'candidates_distances_nn.db'))
CANDIDATES_NN_DB_COMPLETE = os.path.normpath(os.path.join('Ranking', 'Ranked_candidates_andtext_nn.db'))

RANKING_TEMP_DB = os.path.normpath(os.path.join('Ranking','temp_candidates.db'))


CANDIDATES_ONLINE_BALANCED_DB = os.path.normpath(os.path.join('Ranking', 'candidates_ids_online_balanced.db'))
CANDIDATES_ONLINE_UNBALANCED_DB = os.path.normpath(os.path.join('Ranking', 'candidates_ids_online_unbalanced.db'))
CANDIDATES_ONLINE_DB_RANKED = os.path.normpath(os.path.join('Ranking', 'candidates_distances_ol.db'))
CANDIDATES_ONLINE_BALANCED_COMPLETE = os.path.normpath(os.path.join('Ranking', 'Ranked_candidates_andtext_OL_balanced.db'))
CANDIDATES_ONLINE_UNBALANCED_COMPLETE = os.path.normpath(os.path.join('Ranking', 'Ranked_candidates_andtext_OL_unbalanced.db'))