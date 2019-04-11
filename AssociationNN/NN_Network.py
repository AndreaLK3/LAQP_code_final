import tensorflow as tf
import numpy as np
import logging
import utilities.MyUtils as MyUtils
import utilities.Filenames as F
import sqlite3
import ast
from time import time
import json

import utilities.MyUtils_dbs as MyUtils_dbs
import utilities.MyUtils_flags as MyUtils_flags

RANDOM_SEED = 42
tf.set_random_seed(RANDOM_SEED)


#source for part of the code:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/mnist.py#L8
# and the example code, as well

# We have 2 classes (1 and 0: question was/wasn't asked for the product)
NUM_CLASSES = 2


########## I : For training, pick balanced random minibatches, and get them with a generator

### N: In this version, the training set has already been shuffled. No randomization of indices is necessary
### (Picking adjacent rowid-s speeds up the SELECT operation...)
def generator_of_batches(batch_size, dataset_type):
    if dataset_type == MyUtils_flags.FLAG_TRAIN:
        dataset_length = MyUtils_dbs.get_nn_dataset_length(MyUtils_flags.FLAG_TRAIN)
        db_conn = sqlite3.connect(F.NN_TRAIN_INSTANCES_DB)
    if dataset_type == MyUtils_flags.FLAG_VALID:
        dataset_length = MyUtils_dbs.get_nn_dataset_length(MyUtils_flags.FLAG_VALID)
        db_conn = sqlite3.connect(F.NN_VALID_INSTANCES_DB)
    if dataset_type == MyUtils_flags.FLAG_TEST:
        dataset_length = MyUtils_dbs.get_nn_dataset_length(MyUtils_flags.FLAG_TEST)
        db_conn = sqlite3.connect(F.NN_TEST_INSTANCES_DB)

    c = db_conn.cursor()
    num_of_batches = dataset_length // batch_size + 1
    half_mark_offset = dataset_length // 2

    for i in range(0, num_of_batches):
        start_index_pos = i * (batch_size//2)
        end_index_pos = min( (i+1)*(batch_size//2), half_mark_offset)
        start_index_neg = half_mark_offset + i * (batch_size//2)
        end_index_neg =  half_mark_offset + min( (i+1)*(batch_size//2), dataset_length)

        c.execute("SELECT p_id, q_id, x,y FROM instances WHERE rowid IN " + str(tuple(range(start_index_pos, end_index_pos))))
        rows = c.fetchall()
        c.execute("SELECT p_id, q_id, x,y FROM instances WHERE rowid IN " + str(tuple(range(start_index_neg, end_index_neg))))
        rows_neg = c.fetchall()

        rows.extend(rows_neg)
        batch = list(map(lambda elem: (str(elem[0]), str(elem[1]), json.loads(elem[2]), int(elem[3])), rows))
        yield batch



def fill_feed_dict(batch_raw, input_pl, labels_pl):


    #p_ids =  [instance[0] for instance in batch_raw]
    #q_ids = [instance[1] for instance in batch_raw]
    xs = [instance[2] for instance in batch_raw]
    ys = [instance[3] for instance in batch_raw]

    feed_dict = {
        input_pl: xs,
        labels_pl: ys,
    }
    logging.debug(feed_dict)
    return feed_dict


########## II: The structure of the Neural Network

def nn_inference(input, layers_hidden_units_ls, dropout_rate):
    #Build the model up to where it may be used for inference.

    current_layer_id = 1
    prev_layer_units = input.shape[1]
    hidden_1 = input
    with tf.variable_scope("nn_varscope", reuse=tf.AUTO_REUSE):
        for num_of_hidden_units in layers_hidden_units_ls:
            # Add hidden layer to the model
            logging.info("prev_layer_units: %s, ", prev_layer_units)
            logging.info("num_of_hidden_units: %s, ", num_of_hidden_units)
            with tf.name_scope('hidden'+str(current_layer_id)):
                weights = tf.get_variable(name="weights_in_h"+str(current_layer_id), shape=(prev_layer_units, num_of_hidden_units),
                                          initializer=tf.contrib.layers.xavier_initializer())
                biases = tf.get_variable(name='biases_h'+str(current_layer_id), shape=num_of_hidden_units,
                                         initializer=tf.contrib.layers.xavier_initializer())  # tf.zeros([hidden1_units])
                hidden_2 = tf.nn.tanh(tf.matmul(hidden_1, weights) + biases)  # arctangent used as activation function
                hidden_2_withdropout = tf.layers.dropout(hidden_2, rate=dropout_rate,training=True,name="hiddenlayeroutput_withdropout")

                # (could also use relu (Rectified Linear Unit) and see how it works.)
                #tensorboard logging:
                tf.summary.histogram(name="weights_in_h"+str(current_layer_id), values=weights)
                tf.summary.histogram(name="biases_" + str(current_layer_id), values=biases)
                #preparing for the next layer:
                prev_layer_units = num_of_hidden_units
                hidden_1 = hidden_2_withdropout
                current_layer_id = current_layer_id +1

        # To the output:
        with tf.name_scope('output'):
            weights = tf.get_variable(name="weights_hlast_out", shape= [prev_layer_units, NUM_CLASSES],
                                    initializer=tf.contrib.layers.xavier_initializer())
            biases = tf.get_variable(initializer=tf.zeros((NUM_CLASSES)),name='biases_out')
            logits = tf.matmul(hidden_2, weights) + biases ##The output activation function and error are handled later
            tf.summary.histogram(name="logits", values=logits)
            #n: output dimensions: if ==1, outfunc=sigmoid ; if>=2, outfunc=softmax
            #logging.info("Logits tensor: %s", logits)

    return logits


def nn_loss_computation(logits, labels):
    #labels = tf.to_int64(labels)
    logging.info("Labels: %s", labels)

    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=logits))



def get_model_placeholders(batch_size):
    # temporary connection to database... I wish to determine the number of input dimensions
    batch_generator = generator_of_batches(4, MyUtils_flags.FLAG_TRAIN)
    batch = batch_generator.__next__()
    #logging.info(batch)
    elem_1 = batch[0]
    x_1 = elem_1[2]
    input_placeholder = tf.placeholder(shape=(None, len(x_1)), dtype=tf.float32, name="input_pl") #batch_size or None can be used
    logging.info("Input placeholder: %s", input_placeholder)

    labels_placeholder = tf.placeholder(shape=(None,), dtype=tf.int32, name="labels_pl")

    return (input_placeholder, labels_placeholder)


def get_permuted_indices(trainset_length, batch_size):
    num_of_batches = int(trainset_length / batch_size)  # eg. 19345 // 32 == 604

    # note: if we wish to create balanced batches, with 1/2 positive and 1/2 negative examples,
    # we need to permute separately the first and the second half of the indices, and then access the 2 halves separately
    half_mark = trainset_length // 2
    # logging.info("Indices halfmark: %s", half_mark)

    ids_pos = np.random.choice(range(0, half_mark), half_mark,
                               replace=False)  # "sample without replacement so we get every sample once"
    # the concrete effect of the previous line is to permute the indices from 0 to half_mark
    # logging.info(ids_pos)
    ids_neg = np.random.choice(range(half_mark, trainset_length), half_mark, replace=False)
    # logging.info(ids_neg)

    return (ids_pos, ids_neg, num_of_batches)