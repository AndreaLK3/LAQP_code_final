import numpy as np
import scipy.spatial.distance as distance
import logging
import utilities.MyUtils
import OnlineLearning.CosineSim as CS

class Action():

    def execute_action(self, x):
        pass


class RandomSimulator(Action):

    def __init__(self, upper_number):
        self.upper_number = upper_number
        self.name = "RandomSimulator"+str(upper_number)

    def execute_action(self, x):
        return int( (np.random.randint(low=0, high=self.upper_number+1)) >= 1)


###### Comparison between one Doc2Vec vector(P.titlevec or P.descvec) and another (Q.questionVec), using cosine similarity
class Compare2SingleVectors(Action):
    def __init__(self, p_featurename, cosine_sim_fraction, name):
        self.p_featurename = p_featurename #one of p_titlevec or p_descvec
        self.cosine_sim_threshold = CS.get_similarity_breakpoint(p_featurename[2:], "questionVec", cosine_sim_fraction)
        self.name = name

    def execute_action(self, x):
        cosine_sim = 1 - distance.cosine(x[self.p_featurename], x["q_questionVec"])
        if cosine_sim >= self.cosine_sim_threshold:
            return 1
        else:
            return 0


###### Comparison between a list of Doc2Vec vectors(P.mdcategories or P.kwsVectors) and a single vector (Q.questionVec)
class CompareListToVector(Action):
    def __init__(self, p_featurename, cosine_sim_fraction, name):
        self.p_featurename = p_featurename #one of p_kwsVectors or p_mdcategories
        self.cosine_sim_threshold = CS.get_similarity_breakpoint(p_featurename[2:], "questionVec", cosine_sim_fraction)
        self.name = name

    def execute_action(self, x):
        p_vectors_ls = x[self.p_featurename]
        N = len(p_vectors_ls)
        M = ( N * (N+1) ) // 2

        q_text = x["q_questionVec"]

        # for p_vector in p_vectors_ls:
        #     logging.debug("x.%s, shape: %s", self.p_featurename, np.array(p_vector).shape)
        # logging.debug("x.q.questionVec, shape: %s", np.array(q_text).shape)
        cosine_sims = sorted( list(map(lambda p_vector : 1 - distance.cosine(q_text, p_vector), p_vectors_ls))) #(ascending)
        weights = [w / M for w in range(1,N+1)]
        avg_cosine_sim = sum ( np.multiply(cosine_sims, weights) )

        if avg_cosine_sim >= self.cosine_sim_threshold:
            return 1
        else:
            return 0


###### Comparison between 2 lists of Doc2Vec vectors(P.mdcategories or P.kwsVectors, and Q.kwsVectors)
class Compare2Lists(Action):
    def __init__(self, p_featurename, cosine_sim_fraction, name):
        self.p_featurename = p_featurename #one of p_kwsVectors or p_mdcategories
        self.cosine_sim_threshold = CS.get_similarity_breakpoint(p_featurename[2:], "kwsVectors", cosine_sim_fraction)
        self.name = name

    def execute_action(self, x):
        vectors_ls_1 = x[self.p_featurename]
        vectors_ls_2 = x["q_kwsVectors"]
        m = len(vectors_ls_1)
        n = len(vectors_ls_2)
        sim_matrix = np.ones(shape=(m, n)) * -1
        for i in range(m):
            kw_vec_1 = vectors_ls_1[i]
            # logging.debug("Vector 1 : %s, kw_vec_1)
            for j in range(n):
                kw_vec_2 = vectors_ls_2[j]
                # logging.debug(kw_vec_2)
                sim_matrix[i][j] = 1 - distance.cosine(u=kw_vec_1, v=kw_vec_2)
        #logging.debug("\nThe sim.matrix : %s", sim_matrix)
        aggregated_sim = np.average(utilities.MyUtils.pick_maxmatches_matrix(sim_matrix))
        #logging.debug("Aggregated' similarity: %s", kws_aggregated_sim)
        if aggregated_sim >= self.cosine_sim_threshold:
            return 1
        else:
            return 0