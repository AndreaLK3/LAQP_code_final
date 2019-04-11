import OnlineLearning.BasicActions as A
import utilities.Filenames

####### Creating the Actions that will be evaluated in the Online Learning setting.
####### The types of Actions are defined in the the module OnlineLearning.Actions
def initialize_actions_all_featurecomparisons():

    all_actions_initialized = initialize_actions_compareVec2Vec() + initialize_actions_compareLs2Vec() \
                              + initialize_actions_compareLs2Ls()

    return all_actions_initialized



####### Subcategories of actions:
def initialize_actions_compareVec2Vec():
    return init_actions_compare_PdescQtext() + init_actions_compare_PtitleQtext()


def init_actions_compare_PdescQtext():
    cosine_sim_thresholds = [0.5, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    actions = []
    for sim_threshold in cosine_sim_thresholds:
        actions.append(A.Compare2SingleVectors(p_featurename="p_descvec", cosine_sim_fraction=sim_threshold,
                                 name="P_descvec_Q_questionVec_" + str(sim_threshold)))
    return actions

def init_actions_compare_PtitleQtext():
    cosine_sim_thresholds = [0.5, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    actions = []
    for sim_threshold in cosine_sim_thresholds:
        actions.append(A.Compare2SingleVectors(p_featurename="p_titlevec", cosine_sim_fraction=sim_threshold,
                                               name="P_titlevec_Q_questionVec_" + str(sim_threshold)))
    return actions


####################


def initialize_actions_compareLs2Vec():
    return init_actions_compare_PkeywordsQtext() + init_actions_compare_PcategoriesQtext()


def init_actions_compare_PkeywordsQtext():
    cosine_sim_thresholds = [0.5, 0.60, 0.70, 0.80, 0.90 ]
    actions = []
    for sim_threshold in cosine_sim_thresholds:
        actions.append(A.CompareListToVector(p_featurename="p_kwsVectors", cosine_sim_fraction=sim_threshold,
                                               name="P_kwsVectors_Q_questionVec_" + str(sim_threshold)))
    return actions

def init_actions_compare_PcategoriesQtext():
    cosine_sim_thresholds = [0.5, 0.60, 0.70, 0.80, 0.90]
    actions = []
    for sim_threshold in cosine_sim_thresholds:
        actions.append(A.CompareListToVector(p_featurename="p_mdcategories", cosine_sim_fraction=sim_threshold,
                                               name="P_mdcategories_Q_questionVec_" + str(sim_threshold)))
    return actions

####################


def initialize_actions_compareLs2Ls():
    return init_actions_compare_PkeywordsQkeywords() + init_actions_compare_PcategoriesQkeywords()

def init_actions_compare_PkeywordsQkeywords():
    cosine_sim_thresholds = [0.5, 0.60, 0.70, 0.80, 0.90]
    actions = []
    for sim_threshold in cosine_sim_thresholds:
        actions.append(A.Compare2Lists(p_featurename="p_kwsVectors", cosine_sim_fraction=sim_threshold,
                                               name="P_kwsVectors_Q_kwsVectors_" + str(sim_threshold)))
    return actions

def init_actions_compare_PcategoriesQkeywords():
    cosine_sim_thresholds = [0.5, 0.60, 0.70, 0.80, 0.90]
    actions = []
    for sim_threshold in cosine_sim_thresholds:
        actions.append(A.Compare2Lists(p_featurename="p_mdcategories", cosine_sim_fraction=sim_threshold,
                                               name="P_mdcategories_Q_kwsVectors_" + str(sim_threshold)))
    return actions