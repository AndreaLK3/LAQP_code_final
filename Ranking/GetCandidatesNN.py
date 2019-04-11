import tensorflow as tf
import utilities.Filenames as F
import logging
import utilities.MyUtils as MyUtils
import utilities.MyUtils_filesystem as MyUtils_filesystem
import pandas as pd
import os.path
from AssociationNN.NN_Train_Common import compute_output_dataset_accuracy
import sqlite3

def apply_NN_on_testset():
    with tf.Session() as session:
        MyUtils.init_logging("GetCandidatesNN.log")

        test_db = sqlite3.connect(F.NN_TEST_INSTANCES_DB)
        test_db_c = test_db.cursor()

        saver = tf.train.import_meta_graph(F.SAVED_NN+'.meta')
        saver.restore(session, tf.train.latest_checkpoint(os.path.dirname(F.SAVED_NN)))

        MyUtils_filesystem.clean_directory(F.NN_TEST_OUTPUT_DIR)
        test_filewriter = tf.summary.FileWriter(os.path.join(F.NN_TEST_OUTPUT_DIR), session.graph)


    #with tf.variable_scope("reuse_fortest_scope", reuse=tf.AUTO_REUSE):
        graph = tf.get_default_graph()
        input_placeholder = graph.get_tensor_by_name("input_pl:0")
        labels_placeholder = graph.get_tensor_by_name("labels_pl:0")
        placeholders = (input_placeholder, labels_placeholder)

        test_accuracy_summary = graph.get_tensor_by_name("Accuracy:0")
        predictions = graph.get_tensor_by_name("predictions:0")
        tf_metric, tf_metric_update = tf.metrics.accuracy(labels=labels_placeholder, predictions=predictions,
                                                          name="Test_accuracy")

        tasks = [tf_metric_update, predictions]
        writing_tasks = [test_accuracy_summary]

        accuracy_variables = list(filter(lambda var: "accuracy" in var.name, tf.local_variables() ) )
        logging.debug("Accuracy variables: %s",  accuracy_variables)
        session.run(tf.variables_initializer(accuracy_variables))

        (test_accuracy_value, foundcandidates_mapls)= \
            compute_output_dataset_accuracy(1, placeholders, test_db_c,tasks, writing_tasks, [test_filewriter], session)

        foundcandidates_ls = sorted(foundcandidates_mapls, key=lambda elem: elem[0])

        logging.info("Test accuracy value: %s", test_accuracy_value)
        logging.info("Candidates found: %s", foundcandidates_ls)

        candidates_outdb = sqlite3.connect(F.CANDIDATES_NN_DB)
        #outdb_c = candidates_outdb.cursor()

        candidates_df = pd.DataFrame(foundcandidates_ls, columns=["p_id", "q_id"], dtype=str)
        candidates_df.to_sql(name="candidates", con=candidates_outdb, if_exists="replace", dtype={"p_id":"varchar(63)",
                                                                                                  "q_id":"varchar(63)"})


