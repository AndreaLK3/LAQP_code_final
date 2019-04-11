import ReadQuestions as RQ
import pandas as pd
import MyUtils
import logging

#n: to apply on a dataset of small size, eg. the train subset
def test_order():
    q_df = pd.read_csv(RQ.QA_TRAINSUBSET_DFPATH, sep="_")
    q_df_sorted = q_df.sort_values(by=["asin", "unixTime"], axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last')

    q_df_sorted.to_csv("Experiment_QA_sorted_trainsubset_df.csv", sep="_")
    print(len(q_df_sorted))


def test_extract_sameids():
    MyUtils.init_logging("ExploreQuestions.log")
    q_df = pd.read_csv(RQ.QA_TRAINSUBSET_DFPATH, sep="_")
    q_df_sorted = q_df.sort_values(by=["asin"], axis=0, ascending=True, inplace=False, kind='quicksort',
                                   na_position='last')

    q_df_asins = q_df_sorted["asin"].copy()
    q_df_asins.drop_duplicates(inplace=True)
    print(len(q_df_asins))

    for asin in q_df_asins:
        q_df_subset = q_df_sorted[q_df_sorted["asin"] == asin].copy()
        subset_duplicates = q_df_subset[q_df_subset.duplicated('unixTime', keep=False) == True]
        if len(subset_duplicates)>0:
            subset_duplicates.to_csv("ExploreQuestions.log", mode="a", sep="_")

