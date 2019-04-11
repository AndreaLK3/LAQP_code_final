import utilities.Filenames as F
import utilities.MyUtils_flags as MyUtils_flags
import utilities.MyUtils_dbs as MyUtils_dbs
import logging
import tensorflow as tf
import os.path
import ast
import AssociationNN.NN_Network as NN


def get_num_training_iterations(batch_size):
    trainset_length = MyUtils_dbs.get_nn_dataset_length(MyUtils_flags.FLAG_TRAIN)
    max_iter = trainset_length // batch_size  # in 1 epoch, you can not have more iterations than batches
    logging.info("Number of iterations per epoch: %s", max_iter)
    return max_iter


def get_filewriters(w_train_tasks, w_valid_tasks, tensorboard_dir_path, session):
    # creating the filewriters for the loss&accuracy documentation tasks
    t_summarytensor_names = list(
        map(lambda filename: filename[0:-2], [summarytensor.name for summarytensor in w_train_tasks]))
    t_filewriters = [
        tf.summary.FileWriter(os.path.join(tensorboard_dir_path, MyUtils_flags.FLAG_TRAIN, summaryname), session.graph)
        for summaryname in t_summarytensor_names]
    v_summarytensor_names = list(
        map(lambda filename: filename[0:-2], [summarytensor.name for summarytensor in w_valid_tasks]))
    v_filewriters = [
        tf.summary.FileWriter(os.path.join(tensorboard_dir_path, MyUtils_flags.FLAG_VALID, summaryname), session.graph)
        for summaryname in v_summarytensor_names]

    return t_filewriters, v_filewriters


#Receives the filewriters as parameter, so that as a side effect it writes on TensorBoard
def compute_output_dataset_accuracy(current_step, placeholders, dataset_db_cursor, tasks,
                                    writing_tasks, v_filewriters, session):

    (input_placeholder, labels_placeholder) = placeholders
    dataset_db_cursor.execute(" SELECT p_id, q_id, x, y from instances ")
    valid_dataset_str = dataset_db_cursor.fetchall()
    valid_dataset_asbatch = list(map(lambda elem: (elem[0], elem[1], ast.literal_eval(elem[2]), int(elem[3])), valid_dataset_str))
    v_feed_dict = NN.fill_feed_dict(valid_dataset_asbatch, input_placeholder, labels_placeholder)
    v_wr_results = session.run(tasks + writing_tasks, feed_dict=v_feed_dict)

    validation_accuracy_value = v_wr_results[0] #the first validation task, i.e. tf_metric_update

    for k in range(0, len(writing_tasks)):
        summary = v_wr_results[len(tasks) + k]
        v_filewriters[k].add_summary(summary, current_step)
        v_filewriters[k].flush()

    outputs = v_wr_results[1] #the second validation task, i.e. predictions
    valid_dataset_ids = list(map(lambda elem: (elem[0], elem[1]), valid_dataset_str))
    ids_outputs_mapls = list(zip(valid_dataset_ids, outputs))
    foundcandidates_mapls_0 = list(filter( lambda ids_outputs_tpl : ids_outputs_tpl[1]>=1 , ids_outputs_mapls))
    foundcandidates_mapls = list(map (lambda elem_tpl : (elem_tpl[0][0], elem_tpl[0][1]) , foundcandidates_mapls_0)) #remove the '1' label

    return (validation_accuracy_value, foundcandidates_mapls)


def reinitialize_running_accuracy_vars(session):
    running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="accuracy")
    running_vars_initializer = tf.variables_initializer(var_list=running_vars)
    session.run(running_vars_initializer)