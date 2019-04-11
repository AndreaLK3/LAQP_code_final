import numpy as np
import logging

import utilities.MyUtils as MyUtils
import utilities.MyUtils_dbs as MyUtils_dbs
import utilities.MyUtils_filesystem as MyUtils_filesystem
import utilities.MyUtils_strings as MyUtils_strings
import utilities.MyUtils_flags as MyUtils_flags
import OnlineLearning.OrganizeCategoryElementsDBs as CCE
import OnlineLearning.ActionsInizialization as AI
import sqlite3
import utilities.Filenames as F
import Create_PQs.VectorizeDescriptions as VD
from collections import namedtuple
import tensorflow as tf
import os.path
from gc import collect
from time import time

import OnlineLearning.ActionsForCategoryDatasets as ACD
import OnlineLearning.ActionsForGlobalDataset as AGD

##### Extracting the ids of product and question from the database that determines the elements of a training instance,
##### and then creating the training instance, using the Representations of the product and the question


def get_instance_encoded_dictionary(prod_id, question_id, ps_db_c, qs_db_c, d2v_model):

    product_row = MyUtils_dbs.search_in_alltables_db(ps_db_c, "SELECT * FROM ", "WHERE id = '" + prod_id + "'")
    question_row = MyUtils_dbs.search_in_alltables_db(qs_db_c, "SELECT * FROM ", "WHERE id = '" + str(question_id) + "'")
    prod_tuple = MyUtils.prodls_tonamedtuple(product_row[0])
    q_tuple = MyUtils.quest_lstonamedtuple(question_row[0])

    instance_x = {}
    instance_x["p_descvec"] = MyUtils_strings.fromstring_toarray(prod_tuple.descvec)
    instance_x["p_titlevec"] = MyUtils_strings.fromstring_toarray(prod_tuple.titlevec)
    instance_x["p_kwsVectors"] = MyUtils_strings.fromlls_toarrays(prod_tuple.kwsVectors)
    #logging.debug("instance_x['p_kwsVectors'].shape : %s", np.array(instance_x["p_kwsVectors"]).shape)
    instance_x["p_mdcategories"] =  MyUtils_strings.categories_to_vecs_lls(
        MyUtils_strings.fromlls_toarrays(prod_tuple.mdcategories), d2v_model)
    if len(np.array(instance_x["p_mdcategories"]).shape) >=3:
        logging.debug("instance_x['p_mdcategories'].shape : %s", np.array(instance_x["p_mdcategories"]).shape)
        instance_x["p_mdcategories"] = instance_x["p_mdcategories"][0]


    instance_x["q_questionVec"] = MyUtils_strings.fromstring_toarray(q_tuple.questionVec)
    instance_x["q_questionType"] = q_tuple.questionType
    instance_x["q_kwsVectors"] = MyUtils_strings.fromlls_toarrays(q_tuple.kwsVectors)

    instance_y = 1 if q_tuple.id[0:10] in prod_id else 0
    instance = namedtuple('instance', 'x y')
    inst = instance(x=instance_x, y=instance_y)

    return inst


def extract_instance(ps_db_c, qs_db_c, instances_c, t, balanced_instances, d2v_model):
    if balanced_instances:
        row_num = t // 2 +1
        # choose alternatively, a positive and a negative instance.
        if t % 2 != 0:
            (p_id, q_id) = instances_c.execute("SELECT * FROM positiveinstances WHERE rowid = ?", (row_num,)).fetchone()
        else:
            (p_id, q_id) = instances_c.execute("SELECT * FROM negativeinstances WHERE rowid = ?", (row_num,)).fetchone()
    else: #not balanced. using categories
        result = instances_c.execute("SELECT * FROM instances WHERE rowid = ?", (t,)).fetchone()
        if result is None:
            logging.warning("Did not found instance with rowid = %s in database", t)
            return None
        else:
            (p_id, q_id, _y) = result
    instance_t = get_instance_encoded_dictionary(p_id, q_id, ps_db_c, qs_db_c, d2v_model)

    return ((instance_t.x, instance_t.y), p_id, q_id)
#####

##### Defining the tensorflow tensors and filewriters, to write in a graph
##### the accumulated loss of the various actions
def get_tensors_summaries_filewriters(actions_ls, results_dirpath, session):
    a_tensors = []
    a_losses_summary = tf.summary.scalar("Global_relative_loss", tf.placeholder(shape=[], dtype=tf.float32, name="global_relative_loss_tensor"))
    a_relativelosses_summary = tf.summary.scalar("Window_relative_loss",
                                            tf.placeholder(shape=[], dtype=tf.float32, name="window_relative_loss_tensor"))
    a_choiceprob_summary = tf.summary.scalar("Choice_probability",
                                                 tf.placeholder(shape=[], dtype=tf.float32,
                                                                name="choice_probability_tensor"))
    a_summaries= [a_losses_summary, a_relativelosses_summary, a_choiceprob_summary]
    a_filewriters = []
    for action in actions_ls:
        action_loss_tensor = tf.placeholder(shape=[], dtype=tf.float32, name=action.name)
        a_tensors.append(action_loss_tensor)
        action_filewriter = tf.summary.FileWriter(os.path.join(results_dirpath, action.name), session.graph)
        a_filewriters.append(action_filewriter)
    return (a_tensors, a_summaries, a_filewriters)


##### Keeping the queue with 'window relative loss': for each action,
##### we mantain and measure the 0-1 loss for the last 100 rounds
def update_window_loss(loss_listofqueues, new_values, queue_length):
    for i in range(len(new_values)): #note : we require than len(loss_listofqueues) == len(new_values)
        action_lossqueue = loss_listofqueues[i]
        action_newvalue = new_values[i]
        action_lossqueue.insert(0,action_newvalue)
        if len(action_lossqueue) >queue_length:
            action_lossqueue.pop()
    return loss_listofqueues
##############################


##############################
##### The Hedge Algorithm (Exponential Weights), for a problem configured as i.i.d. Prediction with Expert Advice
def run_hedge(actions_ls=None, eta=None, max_T=None, balanced_instances=True, restart_candidates=True):
    MyUtils.init_logging("Hedge-run_hedge.log")

    ### initialization: either we use the single global balanced dataset, or the imbalanced category datasets
    if balanced_instances:
        dbs_paths = [(F.ONLINE_INSTANCEIDS_GLOBAL_DB, F.PRODUCTS_FINAL_TRAIN_DB, F.QUESTIONS_FINAL_TRAIN_DB)]
    else:
        category_dirpaths = MyUtils_filesystem.get_category_dirpaths()
        dbs_paths  = [] #list of tuples, with 3 elements: instancedb, products_db, qs_db
        for c_dir_path in category_dirpaths:
            for fname in os.listdir(c_dir_path):
                if "db" in fname:
                    if MyUtils_flags.FLAG_INSTANCEIDS in fname:
                        categ_instances_dbpath = os.path.join(c_dir_path, fname)
                    elif MyUtils_flags.FLAG_PRODUCTS in fname:
                        categ_prods_dbpath =  os.path.join(c_dir_path, fname)
                    else:
                        categ_qs_dbpath = os.path.join(c_dir_path, fname)
            dbs_paths.append((categ_instances_dbpath, categ_prods_dbpath, categ_qs_dbpath))

    ### connecting with the database containing the candidates
    if balanced_instances:
        output_candidates_dbpath = F.CANDIDATES_ONLINE_BALANCED_DB
    else:
        output_candidates_dbpath = F.CANDIDATES_ONLINE_UNBALANCED_DB

    if restart_candidates:
        f = open(output_candidates_dbpath, "w"); f.close()
    output_candidates_db = sqlite3.connect(output_candidates_dbpath)
    output_candidates_c = output_candidates_db.cursor()
    if restart_candidates:
        output_candidates_c.execute("""CREATE TABLE candidates (
                                        p_id varchar(63),
                                        q_id varchar(63)   )""");

    #For each dataset: connect to databases of instances, Ps, and Qs
    for (instances_dbpath, prods_dbpath, quests_dbpath) in dbs_paths:
        instances_db = sqlite3.connect(instances_dbpath)
        instances_ids_c = instances_db.cursor()
        prods_db = sqlite3.connect(prods_dbpath)
        ps_c = prods_db.cursor()
        quests_db = sqlite3.connect(quests_dbpath)
        qs_c = quests_db.cursor()

        chosen_dataset_name = os.path.basename(os.path.dirname(instances_dbpath))
        logging.info("Online Learning: operating on dataset: %s", chosen_dataset_name)

        #### define the number of rounds
        if max_T is None:
            max_T = MyUtils_dbs.get_tot_num_rows_db(instances_ids_c)
            logging.info("Total number of rounds (i.e. instances in the training set): %s", max_T)

        #### define the actions
        if actions_ls is None:
            if balanced_instances == False:
                actions_ls = ACD.get_actionsforcategories()
            else:
                actions_ls = AGD.get_actionsforbalanced()

        #### define the "learning rate"
        if eta is None:
            eta = np.sqrt((2 * np.log(len(actions_ls))) / max_T)

        #### output directory for Tensorboard logging
        results_dirpath = os.path.join('OnlineLearning', 'Experiments_results',
                                       str(chosen_dataset_name),
                                       'numactions_' + str(len(actions_ls)),
                                       'instances_' + str(max_T)
                                       )#datetime.datetime.today().strftime('%Y-%m-%d')
        if not os.path.exists(results_dirpath):
            os.makedirs(results_dirpath)
            MyUtils_filesystem.clean_directory(results_dirpath)
            #### the actual core of the algorithm
            hedge_loop(eta, max_T, actions_ls, instances_ids_c, ps_c, qs_c, output_candidates_db,
                       balanced_instances, results_dirpath)
        else:
            logging.info("Online Learning results already computed for : %s", results_dirpath)



#################
##### Initializes logging, and executes the rounds' loop of the Hedge algorithm.
def hedge_loop(eta, max_T, actions_ls, instances_ids_c, ps_c, qs_c, outcandidates_db, balanced_instances, results_dirpath):
    MyUtils.init_logging(os.path.join(results_dirpath, "Hedge-run.log"))
    outcandidates_c = outcandidates_db.cursor()

    ### Statistics
    num_predicted_0s = 0
    num_predicted_1s = 0
    accumulated_actions_opinions=np.zeros(len(actions_ls))

    #### initializing the Tensorflow graph to register the action losses
    tf.reset_default_graph()
    session = tf.Session()
    (a_loss_tensors, a_summaries, a_filewriters) = get_tensors_summaries_filewriters(actions_ls,results_dirpath, session)
    algorithm_regret_summary = tf.summary.scalar("Hedge_regret",
                                            tf.placeholder(shape=[], dtype=tf.float32, name="hedge_regret_tensor"))
    algorithm_filewriter = tf.summary.FileWriter(os.path.join(results_dirpath, 'Hedge_algorithm'), session.graph)
    tf.global_variables_initializer().run(session=session)

    ### initializing elements used in Hedge: learning rate, lists of losses, &co.
    logging.info("Learning rate eta = %s", eta)
    eta_decrease = ((eta - (eta)) / max_T)  # step for linear decrease. Currently 0
    all_accumulated_losses = np.zeros(len(actions_ls))
    window_lossqueues_ls = [[] for i in range(len(actions_ls))]
    algorithm_lossqueue_ls = [[]]
    alg_total_loss = 0
    doc2vec_model = VD.load_model()  # to transform the mdcategories into Doc2Vec vectors for each instance

    previous_start_time = 0
    log_update_rounds = max_T // 200
    logging.info("Rounds between log writes= %s", log_update_rounds)
    num_pos_instances = 0
    num_neg_instances = 0

    for t in range(1, max_T):

        t0 = time()
        instance_result = extract_instance(ps_c, qs_c, instances_ids_c, t, balanced_instances, doc2vec_model)
        if instance_result is not None:
            ((x, y), p_id, q_id) = instance_result
            if y == 1:
                num_pos_instances = num_pos_instances + 1
            else:
                num_neg_instances = num_neg_instances + 1

            t1 = time()
            choice_probabilities_ls = np.exp(-eta * all_accumulated_losses) / np.sum(np.exp(-eta * all_accumulated_losses),
                                                                                     axis=0)

            chosen_action_index = np.random.choice(np.arange(len(actions_ls)), replace=False, p=choice_probabilities_ls)
            logging.debug("Chosen action:%s", chosen_action_index)

            t2 = time()
            all_round_losses = list(map(lambda action: abs(action.execute_action(x) - y), actions_ls))  # 0-1 loss here
            logging.debug("Column of round losses: %s", all_round_losses)
            window_lossqueues_ls = update_window_loss(window_lossqueues_ls, all_round_losses, log_update_rounds)
            windowlosses_ls = [sum(action_lossqueue) / len(action_lossqueue) for action_lossqueue in window_lossqueues_ls]

            t3 = time()
            loss_from_chosen_action = all_round_losses[chosen_action_index]
            # logging.info("Incurred loss: %s", loss_from_chosen_action)
            alg_total_loss = alg_total_loss + loss_from_chosen_action
            algorithm_lossqueue_ls = update_window_loss(algorithm_lossqueue_ls, [loss_from_chosen_action],
                                                        log_update_rounds)
            algorithm_windowloss = sum(algorithm_lossqueue_ls[0]) / len(algorithm_lossqueue_ls[0])
            all_accumulated_losses = np.sum([all_accumulated_losses, all_round_losses], axis=0)
            eta = eta - eta_decrease
            t4 = time()

            actions_opinions = list(map(lambda action: action.execute_action(x), actions_ls))
            accumulated_actions_opinions = [sum(x) for x in zip(accumulated_actions_opinions, actions_opinions)]
            chosen_action_opinion = actions_opinions[chosen_action_index]
            if chosen_action_opinion == 0:
                num_predicted_0s = num_predicted_0s+1
            else: #if chosen_action_opinion == 1:
                num_predicted_1s = num_predicted_1s+1
                outcandidates_c.execute("INSERT INTO candidates VALUES (?,?)", (p_id, q_id))


            if t % (log_update_rounds) == 0:
                # Text logging
                logging.info("\n***Current round: %s / %s", t, max_T)
                end_segment = time()
                logging.info("Time needed to go through 1/200th of all the rounds: %s s",
                             round(end_segment - previous_start_time, 6))
                previous_start_time = time()
                #actions_opinions = list(map( lambda action : action.execute_action(x), actions_ls))
                #logging.info("Actions opinions:%s", actions_opinions)
                logging.info("***\n Choice probabilities: %s", choice_probabilities_ls)
                logging.info("Loss of the algorithm: %s", alg_total_loss)
                logging.info("Window relative loss of the algorithm: %s %%", round(algorithm_windowloss, 3)*100)
                logging.debug("All accumulated losses: %s", all_accumulated_losses)

                logging.info("Number of positive instances encountered so far: %s", num_pos_instances)
                logging.info("Number of negative instances encountered so far: %s", num_neg_instances)
                logging.info("Number of 1s predicted so far: %s", num_predicted_1s)
                logging.info("Number of 0s predicted so far: %s", num_predicted_0s)
                accumulated_actions_opinions_with_names = [ (actions_ls[i].name, accumulated_actions_opinions[i])
                                                            for i in range(len(accumulated_actions_opinions)) ]
                logging.info("Number of 1s predicted by each action: %s", accumulated_actions_opinions_with_names)


                regret = alg_total_loss - min(all_accumulated_losses)
                logging.info("Regret: %s", regret)

                outcandidates_db.commit()

                # Tensorboard logging
                session.run(a_loss_tensors, feed_dict=dict(zip(a_loss_tensors, all_round_losses)))

                summaryresults = [
                    session.run(a_summaries, feed_dict={'global_relative_loss_tensor:0': all_accumulated_losses[i] / t,
                                                        'window_relative_loss_tensor:0': windowlosses_ls[i],
                                                        'choice_probability_tensor:0': choice_probabilities_ls[i]})
                    for i in range(len(all_accumulated_losses))]
                for k in range(len(actions_ls)):
                    summary_ls = summaryresults[k]
                    for summary in summary_ls:
                        a_filewriters[k].add_summary(summary, t)
                        a_filewriters[k].flush()

                t5 = time()
                alg_summary_results = [session.run(algorithm_regret_summary, feed_dict={'hedge_regret_tensor:0': regret})]
                alg_summary_results.extend(
                    session.run(a_summaries, feed_dict={'global_relative_loss_tensor:0': alg_total_loss / t,
                                                        'window_relative_loss_tensor:0': algorithm_windowloss,
                                                         'choice_probability_tensor:0': 0}) )
                for alg_summary_result in alg_summary_results:
                    algorithm_filewriter.add_summary(alg_summary_result, t)
                algorithm_filewriter.flush()
                t6 = time()
                times = list(map(lambda phasetime : round(phasetime,6) , [t1-t0, t2-t1, t3-t2, t4-t3, t5-t4, t6-t5]))
                logging.debug("Round time analysis: %s", times)
                collect()




