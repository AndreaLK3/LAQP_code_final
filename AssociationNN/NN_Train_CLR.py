import AssociationNN.NN_Network as NN
import tensorflow as tf
import numpy as np
import logging
import utilities.Filenames as F
import utilities.MyUtils as MyUtils
from time import time
import os.path
import AssociationNN.NN_Train_Common as NTC
import utilities.MyUtils_dbs as MyUtils_dbs
import utilities.MyUtils_flags as MyUtils_flags

######## Cyclical Learning Rate ########

#Auxiliary function: given a matrix of loss values during the training that had increasing learning rate,
#pick the best learning rate (i.e. the one that caused the steepest decrease)

def pick_lr_boundaries(loss_matrix, lr_start, lr_increase):
    rows_best_lrs = []
    for row in loss_matrix:
        smoothed_row = MyUtils.smooth_array_exp(row, alpha=0.3)
        rowmin_index = np.argmin(np.diff(smoothed_row))+1
        logging.info("Best index for the row: %s, with value: %s", rowmin_index, round(smoothed_row[rowmin_index],6))
        row_best_lr = lr_start + lr_increase * rowmin_index
        logging.info("Best learning rate found for the epoch: %s", round(row_best_lr,7))
        rows_best_lrs.append(row_best_lr)

    best_lr = np.average(rows_best_lrs)
    logging.info("Best estimated value for the cyclical learning rate : %s", round(best_lr,7))
    max_lr = best_lr #rule of thumb: max = best, min = 1/3 max
    min_lr = 1/3 * max_lr
    return best_lr, min_lr, max_lr

#Auxiliary function: manual compuation of accuracy for the given batch:
def get_batch_accuracy(b_predictions, b_labels):
    #logging.info(b_labels)
    #logging.info(b_predictions)
    num_correct_predictions = sum(np.array(b_predictions) == np.array(b_labels))
    b_acc = num_correct_predictions / len(b_predictions)
    #logging.info(b_acc)
    return b_acc


def find_cyclical_lrate_loop(placeholders, batch_size, hiddenlayers_ls, drop_rate, lrate_start=10 ** (-7), lrate_end = 0.2):
    trainset_length = MyUtils_dbs.get_nn_dataset_length(MyUtils_flags.FLAG_TRAIN)
    max_iter = trainset_length // batch_size  # in 1 epoch, you can not have more iterations than batches

    hiddenlayers_ls_str = [str(num_elems) for num_elems in hiddenlayers_ls]
    tensorboard_dir_path = os.path.join(F.TENSORBOARD_ANN_DIR, 'trainingset_' + str(
        MyUtils_dbs.get_nn_dataset_length("train")),
                                        "bs_" + str(batch_size),
                                        "hls_" + "-".join(hiddenlayers_ls_str),
                                        "lr_clr",
                                        "drop_" + str(drop_rate) + "_explore")
    if not os.path.exists(tensorboard_dir_path):
        os.makedirs(tensorboard_dir_path)

    session = tf.Session()  # separate session for trying to find the optimal l.r. for the Cyclical Learning Rate
    logging.info("*** Session: Cyclical Learning Rate")
    (input_placeholder, labels_placeholder) = placeholders

    train_loss = NN.nn_loss_computation(
        logits=NN.nn_inference(input=input_placeholder, layers_hidden_units_ls=hiddenlayers_ls, dropout_rate=drop_rate),
        labels=labels_placeholder)
    #train_loss_summary = tf.summary.scalar('Cross-entropy', train_loss)

    predictions = tf.argmax(tf.nn.softmax(logits=NN.nn_inference(input_placeholder, hiddenlayers_ls, drop_rate)), axis=1)
    #tf_metric, tf_metric_update = tf.metrics.accuracy(labels=labels_placeholder, predictions=predictions,
    #                                                  name="CLR_train_accuracy")
    running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="CLR_train_accuracy")
    #train_accuracy_summary = tf.summary.scalar('Accuracy', tf_metric_update)

    lrate_tensor = tf.placeholder(shape=[], dtype=tf.float32, name="lrate_tensor")
    optimizer = tf.train.GradientDescentOptimizer(lrate_tensor)
    minimizer_task = optimizer.minimize(train_loss)


    # Global variables are initialized after the graph structure
    tf.global_variables_initializer().run(session=session)

    running_vars_initializer = tf.variables_initializer(var_list=running_vars)
    session.run(running_vars_initializer)

    logging.info("Number of iterations per epoch (and linear steps in the search for the learning rate): %s", max_iter)
    lrate_increase = (lrate_end - lrate_start) / max_iter
    logging.info("Step increase of the learning rate in the exploration epochs: %s", round(lrate_increase,7))
    trial_epochs = 5
    loss_matrix = np.zeros((trial_epochs, max_iter))
    accuracy_matrix = np.zeros((trial_epochs, max_iter))
    for i in range(1,trial_epochs + 1):
        start_epoch_time = time()
        logging.info("Search for the base learning rate; Starting training epoch n. %s", i)
        batch_generator = NN.generator_of_batches(batch_size, MyUtils_flags.FLAG_TRAIN)

        # Train, in the current epoch
        for j in range(0,max_iter):
            session.run(running_vars_initializer) #new batch: re-initializing the accuracy computation
            batch = \
                batch_generator.__next__()
            current_iteration_feed_dict = NN.fill_feed_dict(batch, input_placeholder, labels_placeholder)
            learning_rate = lrate_start + lrate_increase * j
            current_iteration_feed_dict.update({lrate_tensor: learning_rate})


            if j % (max_iter // 20) == 0:
                logging.info("Iteration: %s on %s .", j, max_iter)
            _, current_loss, b_predictions, b_labels = session.run([minimizer_task, train_loss, predictions, labels_placeholder],
                                                                              feed_dict=current_iteration_feed_dict)
            loss_matrix[i-1][j] = current_loss
            accuracy_matrix[i-1][j] = get_batch_accuracy(b_predictions, b_labels)

        end_epoch_time = time()
        logging.info("Searching for the base values for the cyclical learning rate. "+
                     "Training on epoch %s executed. Time elapsed: %s", i, round(end_epoch_time-start_epoch_time,3))

    best_lr, min_lr, max_lr = pick_lr_boundaries(loss_matrix, lrate_start, lrate_increase)
    session.close()

    #write the lr to a logfile
    lrfile = open(os.path.join(tensorboard_dir_path, "found_lr.log"), "w")
    lrfile.write("Cyclical Learning rate: applying the LR test on "+ str(trial_epochs) + "epochs ;\n "+
                 "the average learning rate granting the steepest descent of the loss function is: " + str(best_lr))
    lrfile.close()

    return best_lr, min_lr, max_lr



######## Loop: training with a cyclical learning rate ########

def training_loop_clr(tasks_dict, placeholders, batch_size,
                      max_epochs, base_lrate, max_lrate, valid_db_cursor, tensorboard_dir_path):
    (input_placeholder, labels_placeholder) = placeholders
    max_iter = NTC.get_num_training_iterations(batch_size)

    session = tf.Session()  # separate session
    tf.global_variables_initializer().run(session=session)

    training_tasks = tasks_dict[MyUtils_flags.FLAG_TRAIN_TASKS]
    validation_tasks = tasks_dict[MyUtils_flags.FLAG_VALIDATION_TASKS]
    validation_writing_tasks = tasks_dict[MyUtils_flags.FLAG_WRITING_VALIDATION_TASKS]
    train_writing_tasks = tasks_dict[MyUtils_flags.FLAG_WRITING_TRAIN_TASKS]

    running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="accuracy")
    running_vars_initializer = tf.variables_initializer(var_list=running_vars)
    t_filewriters, v_filewriters = NTC.get_filewriters(train_writing_tasks, validation_writing_tasks, tensorboard_dir_path, session)


    stepsize = 4 * max_iter#size (n. of iterations) of a lr half-cycle. Can go from 2 to 8 times the max_iter
    max_lr_decrease = ((4/5) * max_lrate) / max_epochs #### as the training progresses, we linearly decrease the
    min_lr_decrease = ((4/5) * base_lrate) / max_epochs # extremes of the learning rate interval
    logging.debug("Max_lr= %s, base_lr=%s, max_lr_decrease=%s, min_lr_decrease=%s, n.of epochs=%s",
                 round(max_lrate,6), round(base_lrate,6), round(max_lr_decrease,6), round(min_lr_decrease,6), max_epochs)

    maximum_observed_valid_acc = 0 #to determine if we save the current state of the network or not
    model_saver= tf.train.Saver()

    for i in range(1, max_epochs + 1):
        logging.info("Training epoch n. %s started...", str(i))
        start_epoch_time = time()
        t_batch_generator = NN.generator_of_batches(batch_size, MyUtils_flags.FLAG_TRAIN)

        max_lrate = max_lrate - max_lr_decrease
        base_lrate = base_lrate - min_lr_decrease
        logging.info("Extremes of the learning rate interval : %s , %s", round(max_lrate,6), round(base_lrate,6))

        # Train, in the current epoch
        for j in range(0, max_iter):

            if j % (max_iter // 20) == 0:
                logging.info("Iteration: %s on %s .", j, max_iter)

            #cyclical learning rate:
            cycle = np.floor(1 + j / (2 * stepsize))
            x = np.abs(j / stepsize - 2 * cycle + 1)
            lr = base_lrate + (max_lrate - base_lrate) * np.maximum(0, (1 - x))

            session.run(running_vars_initializer)#new batch: re-initializing the accuracy computation

            #train on batch
            batch = t_batch_generator.__next__()
            current_iteration_feed_dict = NN.fill_feed_dict(batch, input_placeholder, labels_placeholder)
            current_iteration_feed_dict.update({"lrate_tensor:0" : lr})
            alltasks = training_tasks + train_writing_tasks
            all_results = session.run(alltasks,feed_dict=current_iteration_feed_dict)
            start_wtasks_index = len(training_tasks)

            # batch training accuracy: 1to10 infopoints in each epoch.
            if j % (max_iter // 10 + 1) == 0:
                logging.info("Infopoint: writing batch training accuracy")
                for k in range(len(train_writing_tasks)):
                    summary = all_results[start_wtasks_index+k]
                    #logging.info(summary)
                    t_filewriters[k].add_summary(summary, i * max_iter + j)
                    t_filewriters[k].flush()

            # validation accuracy: on the whole validation set, in order not to be influenced by "batch difficulty"
            # 20to29 infopoints, in the whole training process (eg. 100 epochs -> once every 5 epochs)
            if j==max_iter-1:
                logging.info("Infopoint: computing validation dataset accuracy")
                current_step = i * max_iter

                (valid_acc_value, _foundcandidates) = NTC.compute_output_dataset_accuracy(current_step, placeholders,
                                                                                          valid_db_cursor, validation_tasks,
                                                                                          validation_writing_tasks, v_filewriters,
                                                                                          session)

                if valid_acc_value > maximum_observed_valid_acc:
                    logging.info("(Validation accuracy = %s) > (Previous max validation accuracy = %s). Saving NN",
                                 round(valid_acc_value, 3), round(maximum_observed_valid_acc, 3))
                    maximum_observed_valid_acc = valid_acc_value
                    model_saver.save(sess=session, save_path=F.SAVED_NN)


        end_epoch_time = time()
        logging.info("Time spent on training epoch: %s seconds", round(end_epoch_time-start_epoch_time,3))




