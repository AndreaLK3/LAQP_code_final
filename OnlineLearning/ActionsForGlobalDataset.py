import OnlineLearning.BasicActions as A


def get_actionsforbalanced():
    return [a1_title_desc_categs("a1_title_desc_categs"), a2_keywords("a2_desc_keywords"),
            a3("a3"), a4_all("a4_all"), a5_nodesc_notitle("a5_nodesc_notitle")]

class a1_title_desc_categs(A.Action):
    def __init__(self, name):
        self.name = name
        self.Pdesc_Qtext = A.Compare2SingleVectors(p_featurename="p_descvec", cosine_sim_fraction=0.6, name="")
        self.Ptitle_Qtext = A.Compare2SingleVectors(p_featurename="p_titlevec", cosine_sim_fraction=0.6,name="")


    def execute_action(self, x):
        self.Pdesc_Qtext_vote = self.Pdesc_Qtext.execute_action(x)

        self.Ptitle_Qtext_vote = self.Ptitle_Qtext.execute_action(x)

        votes_w = 0.5
        votes_result = votes_w* self.Pdesc_Qtext_vote + votes_w*self.Ptitle_Qtext_vote
        return int(round(votes_result))



class a2_keywords(A.Action):
    def __init__(self, name):
        self.name = name
        self.Pdesc_Qtext =  A.Compare2SingleVectors(p_featurename="p_descvec", cosine_sim_fraction=0.65, name="")
        self.Ptitle_Qtext = A.Compare2SingleVectors(p_featurename="p_titlevec", cosine_sim_fraction=0.65, name="")
        self.Pkeywords_Qtext = A.CompareListToVector(p_featurename="p_kwsVectors", cosine_sim_fraction=0.7, name="")
        self.Pkeywords_Qkeywords = A.Compare2Lists(p_featurename="p_kwsVectors", cosine_sim_fraction=0.7, name="")
        self.Pcategories_Qkeywords = A.Compare2Lists(p_featurename="p_mdcategories", cosine_sim_fraction=0.7,name="")

    def execute_action(self, x):
        self.Pdesc_Qtext_vote = self.Pdesc_Qtext.execute_action(x)

        self.Ptitle_Qtext_vote = self.Ptitle_Qtext.execute_action(x)

        self.Pkeywords_Qtext_vote = self.Pkeywords_Qtext.execute_action(x)

        self.Pkeywords_Qkeywords_vote = self.Pkeywords_Qkeywords.execute_action(x)

        self.Pcategories_Qkeywords_vote = self.Pcategories_Qkeywords.execute_action(x)


        votes_result = 0.1* self.Pdesc_Qtext_vote + 0.1*self.Ptitle_Qtext_vote + 0.3*self.Pkeywords_Qtext_vote + \
                        0.3*self.Pkeywords_Qkeywords_vote + 0.2*self.Pcategories_Qkeywords_vote

        return int(round(votes_result))


class a3(A.Action):
    def __init__(self, name):
        self.name = name
        self.Pdesc_Qtext = A.Compare2SingleVectors(p_featurename="p_descvec", cosine_sim_fraction=0.55,name="")
        self.Ptitle_Qtext = A.Compare2SingleVectors(p_featurename="p_titlevec", cosine_sim_fraction=0.55,name="")
        self.Pcategories_Qtext = A.CompareListToVector(p_featurename="p_mdcategories", cosine_sim_fraction=0.6,name="")

    def execute_action(self, x):
        self.Pdesc_Qtext_vote = self.Pdesc_Qtext.execute_action(x)

        self.Ptitle_Qtext_vote = self.Ptitle_Qtext.execute_action(x)

        self.Pcategories_Qtext_vote = self.Pcategories_Qtext.execute_action(x)

        votes_result = 0.3 * self.Pdesc_Qtext_vote + 0.4 * self.Ptitle_Qtext_vote + 0.3 * self.Pcategories_Qtext_vote
        return int(round(votes_result))


class a4_all(A.Action):
    def __init__(self, name):
        self.name = name
        self.Pdesc_Qtext = A.Compare2SingleVectors(p_featurename="p_descvec", cosine_sim_fraction=0.55,name="")
        self.Ptitle_Qtext = A.Compare2SingleVectors(p_featurename="p_titlevec", cosine_sim_fraction=0.65,name="")
        self.Pcategories_Qtext =  A.CompareListToVector(p_featurename="p_mdcategories", cosine_sim_fraction=0.5,name="")
        self.Pkeywords_Qtext = A.CompareListToVector(p_featurename="p_kwsVectors", cosine_sim_fraction=0.5,name="")
        self.Pkeywords_Qkeywords = A.Compare2Lists(p_featurename="p_kwsVectors", cosine_sim_fraction=0.5,name="")
        self.Pcategories_Qkeywords = A.Compare2Lists(p_featurename="p_mdcategories", cosine_sim_fraction=0.5,name="")

    def execute_action(self, x):
        self.Pdesc_Qtext_vote = self.Pdesc_Qtext.execute_action(x)

        self.Ptitle_Qtext_vote = self.Pdesc_Qtext.execute_action(x)

        self.Pcategories_Qtext_vote =self.Ptitle_Qtext.execute_action(x)

        self.Pkeywords_Qtext_vote = self.Pcategories_Qtext.execute_action(x)

        self.Pkeywords_Qkeywords_vote = self.Pkeywords_Qkeywords.execute_action(x)

        self.Pcategories_Qkeywords_vote = self.Pcategories_Qkeywords.execute_action(x)

        votes_w = 1 / 6
        votes_result = votes_w * self.Pdesc_Qtext_vote + votes_w * self.Ptitle_Qtext_vote + votes_w * self.Pcategories_Qtext_vote \
                       + votes_w * self.Pkeywords_Qtext_vote + votes_w * self.Pkeywords_Qkeywords_vote + votes_w * self.Pcategories_Qkeywords_vote
        return int(round(votes_result))


class a5_nodesc_notitle(A.Action):
    def __init__(self, name):
        self.name = name
        self.Pcategories_Qtext = A.CompareListToVector(p_featurename="p_mdcategories", cosine_sim_fraction=0.6, name="")
        self.Pkeywords_Qtext = A.CompareListToVector(p_featurename="p_kwsVectors", cosine_sim_fraction=0.6,name="")
        self.Pkeywords_Qkeywords = A.Compare2Lists(p_featurename="p_kwsVectors", cosine_sim_fraction=0.6,name="")
        self.Pcategories_Qkeywords = A.Compare2Lists(p_featurename="p_mdcategories", cosine_sim_fraction=0.6, name="")

    def execute_action(self, x):

        self.Pcategories_Qtext_vote = self.Pcategories_Qtext.execute_action(x)

        self.Pkeywords_Qtext_vote = self.Pkeywords_Qtext.execute_action(x)

        self.Pkeywords_Qkeywords_vote = self.Pkeywords_Qkeywords.execute_action(x)

        self.Pcategories_Qkeywords_vote = self.Pcategories_Qkeywords.execute_action(x)

        w = 0.25
        votes_result =  w * self.Pcategories_Qtext_vote + w * self.Pkeywords_Qtext_vote + \
                        w * self.Pkeywords_Qkeywords_vote + w * self.Pcategories_Qkeywords_vote
        return int(round(votes_result))


