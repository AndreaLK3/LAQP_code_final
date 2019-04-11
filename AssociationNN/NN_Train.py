import AssociationNN.NN_Network as NN
import AssociationNN.NN_Train_CLR as CLR
import tensorflow as tf
import numpy as np
import logging
import utilities.MyUtils as MyUtils
import utilities.Filenames as F
from time import time
import os.path
import AssociationNN.NN_Train_Common as NTC
import sqlite3
import utilities.MyUtils_dbs as MyUtils_dbs
import utilities.MyUtils_filesystem as MyUtils_filesystem
import utilities.MyUtils_flags as MyUtils_flags

#### Main function ####

def train_NN(learning_rate = 0.01, max_epochs=1000, batch_size=32, dropout_rate=0, hiddenlayers_ls=None):

    MyUtils.init_logging("train_NN.log")
    hiddenlayers_ls_str = [str(num_elems) for num_elems in hiddenlayers_ls]
    tensorboard_dir_path = os.path.join(F.TENSORBOARD_ANN_DIR, 'trainingset_' + str(
        MyUtils_dbs.get_nn_dataset_length("train")),
                            "bs_" + str(batch_size),
                            "hls_" + "-".join(hiddenlayers_ls_str),
                            "lr_" + str(learning_rate),
                            "drop_" + str(dropout_rate) + "eps_" + str(max_epochs))
    if not os.path.exists(tensorboard_dir_path):
        os.makedirs(tensorboard_dir_path)
    MyUtils_filesystem.clean_directory(tensorboard_dir_path)

    tf.reset_default_graph()
    session = tf.Session()

    logging.info("Creating the placeholders for input and labels...")
    (input_placeholder, labels_placeholder) = NN.get_model_placeholders(batch_size)
    placeholders = (input_placeholder, labels_placeholder)

    logging.info("Connecting the loss computation and forward structure...")
    train_loss = NN.nn_loss_computation(logits=NN.nn_inference(input=input_placeholder, layers_hidden_units_ls=hiddenlayers_ls,
                                                               dropout_rate=dropout_rate),
                               labels=labels_placeholder)

    lrate_tensor = tf.placeholder(shape=[], dtype=tf.float32, name="lrate_tensor")


    ####### Defining the optimizer
    if str(learning_rate).lower() == MyUtils_flags.FLAG_ADAM:
        starting_lrate = MyUtils_flags.FLAG_ADAM
        optimizer = tf.train.AdamOptimizer()
    else:
        if str(learning_rate).lower() == MyUtils_flags.FLAG_RMSPROP:
            starting_lrate = MyUtils_flags.FLAG_RMSPROP
            optimizer = tf.train.RMSPropOptimizer(0.001)
        else:
            starting_lrate = learning_rate
            optimizer = tf.train.GradientDescentOptimizer(lrate_tensor)
            if str(learning_rate).lower() == MyUtils_flags.FLAG_CLR:
                _best_lr, min_lr, max_lr = CLR.find_cyclical_lrate_loop(placeholders, batch_size, hiddenlayers_ls,
                                                                        dropout_rate)

    # Summaries, and gathering information:
    train_loss_summary = tf.summary.scalar('Cross-entropy', train_loss)
    predictions = tf.argmax(tf.nn.softmax(logits=NN.nn_inference(input_placeholder, hiddenlayers_ls, dropout_rate)),
                            axis=1, name="predictions")

    tf_metric, tf_metric_update = tf.metrics.accuracy(labels=labels_placeholder, predictions=predictions,
                                                      name="accuracy")

    accuracy_summary = tf.summary.scalar('Accuracy', tf_metric_update)

    logging.info("Defining the optimizer's minimization task on the loss function...")
    minimizer_task = optimizer.minimize(train_loss)

    #Global variables are initialized after the graph structure
    tf.global_variables_initializer().run(session=session)

    #defining the tasks that will be run inside the training loop
    training_tasks = [minimizer_task, train_loss, predictions, tf_metric_update]
    validation_tasks = [tf_metric_update, predictions]
    validation_writing_tasks = [accuracy_summary]
    train_writing_tasks = [train_loss_summary, accuracy_summary]

    tasks_dictionary = {MyUtils_flags.FLAG_TRAIN_TASKS: training_tasks,
                        MyUtils_flags.FLAG_WRITING_TRAIN_TASKS: train_writing_tasks,
                        MyUtils_flags.FLAG_VALIDATION_TASKS: validation_tasks,
                        MyUtils_flags.FLAG_WRITING_VALIDATION_TASKS: validation_writing_tasks}

    #connection to the validation dataset
    valid_db_conn = sqlite3.connect(F.NN_VALID_INSTANCES_DB)
    valid_db_cursor = valid_db_conn.cursor()

    if str(learning_rate).lower() == MyUtils_flags.FLAG_CLR:
        CLR.training_loop_clr(tasks_dictionary, placeholders, batch_size,
                                  max_epochs, min_lr, max_lr, valid_db_cursor, tensorboard_dir_path)
    else:
        training_loop(tasks_dictionary, placeholders, starting_lrate,
                      batch_size, max_epochs, valid_db_cursor, tensorboard_dir_path, session)


######## Loop: training with a fixed learning rate ########

def training_loop(tasks_dict, placeholders, start_lrate, batch_size, max_epochs, valid_db_cursor, tensorboard_dir_path, session):
    (input_placeholder, labels_placeholder) = placeholders
    max_iter = NTC.get_num_training_iterations(batch_size)

    train_tasks = tasks_dict[MyUtils_flags.FLAG_TRAIN_TASKS]
    w_train_tasks = tasks_dict[MyUtils_flags.FLAG_WRITING_TRAIN_TASKS]
    valid_tasks = tasks_dict[MyUtils_flags.FLAG_VALIDATION_TASKS]
    w_valid_tasks = tasks_dict[MyUtils_flags.FLAG_WRITING_VALIDATION_TASKS]

    t_filewriters, v_filewriters = NTC.get_filewriters(w_train_tasks, w_valid_tasks, tensorboard_dir_path, session)
    running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="accuracy")
    running_vars_initializer = tf.variables_initializer(var_list=running_vars)

    #linear decay for the learning rate, across the entirety of the process: it goes from start_lr to (10^-1)*start_lr.
    if type(start_lrate) is float:
        lr_decrease = ( start_lrate - (start_lrate / 10) ) / max_epochs

    maximum_observed_valid_acc = 0  # to determine if we save the current state of the network or not
    model_saver = tf.train.Saver()

    for i in range(1, max_epochs + 1):
        logging.info("Training epoch n. %s started...", str(i))
        start_epoch_time = time()
        t_batch_generator = NN.generator_of_batches(batch_size, MyUtils_flags.FLAG_TRAIN)

        if type(start_lrate) is float:
            lr = start_lrate - i*lr_decrease
            logging.info("Learning rate used in the epoch : %s", lr)

        # Train, in the current epoch
        for j in range(0, max_iter):

            if j % (max_iter // 20) == 0:
                logging.info("Iteration: %s on %s .", j, max_iter)

            session.run(running_vars_initializer)#new batch: re-initializing the accuracy computation
            t_batch = t_batch_generator.__next__()
            t_feed_dict = NN.fill_feed_dict(t_batch, input_placeholder, labels_placeholder)

            if type(start_lrate) is float: #if we are running a gradientDescentOptimizer, not an AdamOpt.:
                t_feed_dict.update({"lrate_tensor:0": lr})

            logging.debug("Tasks: %s", train_tasks + w_train_tasks)
            logging.debug("feed_dict: %s", t_feed_dict)
            t_wr_results = session.run(train_tasks + w_train_tasks, feed_dict=t_feed_dict)

            start_t_wtasks_index = len(train_tasks)
            # batch training accuracy: 1to10 infopoints in each epoch.
            if j % (max_iter // 10 + 1) == 0:
                logging.info("Infopoint: writing batch training accuracy")
                for k in range(len(w_train_tasks)):
                    summary = t_wr_results[start_t_wtasks_index + k]
                    t_filewriters[k].add_summary(summary, i * max_iter + j)
                    t_filewriters[k].flush()

            # validation accuracy: on the whole validation set, in order not to be influenced by "batch difficulty"
            if j == max_iter - 1:
                logging.info("Infopoint: computing validation dataset accuracy")
                current_step = i * max_iter
                (valid_acc_value, _foundcandidates) = NTC.compute_output_dataset_accuracy(current_step, placeholders, valid_db_cursor, valid_tasks,
                                                                      w_valid_tasks, v_filewriters, session)

                if valid_acc_value > maximum_observed_valid_acc:
                    logging.info("(Validation accuracy = %s) > (Previous max validation accuracy = %s). Saving NN",
                                 round(valid_acc_value, 3), round(maximum_observed_valid_acc, 3))
                    maximum_observed_valid_acc = valid_acc_value
                    model_saver.save(sess=session, save_path=F.SAVED_NN)



        end_epoch_time = time()
        logging.info("Time spent on training epoch: %s seconds", round(end_epoch_time-start_epoch_time,3))

