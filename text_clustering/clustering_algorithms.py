from strsimpy.levenshtein import Levenshtein
from time_tracker.decorators import time_tracker
import ast
from jellyfish import jaro_similarity, jaro_winkler_similarity
from difflib import SequenceMatcher
import pandas as pd
import ast
import re
from pg.utils.text_proc_utils import cluster_names
from clustering import text_preprocess_and_map

lev = Levenshtein()


def get_similar_score_items(sort_list: list) -> list:
    """
        Desc:
            Return the list of entities which have similar distances or scores
        Example:
            input: [("Bhanpur", 0.8), ("Banpur", 0.8), ("Bhuvanpur", 0.7)]
            output: [("Bhanpur", 0.8), ("Banpur", 0.8)]
    """

    i = 0
    j = 1

    n = len(sort_list)

    similar_word = []

    while(sort_list[i][1] == sort_list[j][1]):
        similar_word.append(sort_list[i])
        i += 1
        j += 1
        if j == n:
            similar_word.append(sort_list[i])
            break
        if sort_list[i][1] != sort_list[j][1]:
            similar_word.append(sort_list[i])
            break

    return similar_word

def get_source_district_list():

    SOURCE_VILLAGE_FILE = "clustering/villages/mh_villages_source.csv"
    df = pd.read_csv(SOURCE_VILLAGE_FILE)
    district_list = df.district.unique().tolist()

    return district_list


def get_source_village(district: str) -> list:
    """
        Desc:
            The function returns the unique list of source villages from a given district
    """

    SOURCE_VILLAGE_FILE = "clustering/villages/mh_villages_source.csv"

    df = pd.read_csv(SOURCE_VILLAGE_FILE)
    df_filtered = df.loc[df["district"] == district]
    df_filtered = df_filtered[["village"]]
    source_list = list(set(list(df_filtered["village"])))

    return source_list


def jaccard_similarity(a, b):
    """
        Desc:
            Returns the score of jaccard_similarity which ranges between 0 and 1
    """

    # convert to set
    a = set(a)
    b = set(b)

    # calucate jaccard similarity
    j = float(len(a.intersection(b))) / len(a.union(b))

    return j


def find_max_jaccard_match(word: str, source: list):
    """
        Desc:
            Returns the most matched word from the source and its corresponding score using
            jaccard similarity alogorithm, with an additional list of helper words which have same score
    """

    new_list = []

    for village in source:
        jac = jaccard_similarity(word, village)
        new_list.append((village, jac))

    sort_list = sorted(new_list, key=lambda x : x[1], reverse=True)

    similar_word = get_similar_score_items(sort_list)

    return (sort_list[0][0], sort_list[0][1], similar_word)


def find_max_ratcliff(word: str, source: list):
    """
        Desc:
            Returns the most matched word from the source and its corresponding score using
            ratcliff alogorithm, with an additional list of helper words which have same score
    """

    new_list = []

    for village in source:
        rat = SequenceMatcher(None, word, village)
        rat = rat.ratio()
        new_list.append((village, rat))

    sort_list = sorted(new_list, key=lambda x : x[1], reverse=False)

    similar_word = get_similar_score_items(sort_list)

    return (sort_list[0][0], sort_list[0][1], similar_word)



def find_min_levenshtein(word: str, source: list):
    """
        Desc:
            Returns the most matched word from the source and its corresponding score using
            levenshtein alogorithm, with an additional list of helper words which have same score
    """

    new_list = []

    for village in source:
        leve = lev.distance(word, village)
        new_list.append((village, leve))

    sort_list = sorted(new_list, key=lambda x : x[1], reverse=False)

    similar_word = get_similar_score_items(sort_list)

    return (sort_list[0][0], sort_list[0][1], similar_word)


def find_max_jaro_distance(word: str, source: list):
    """
        Desc:
            Returns the most matched word from the source and its corresponding score using
            jaro similarity alogorithm, with an additional list of helper words which have same score
    """

    new_list = []

    for village in source:
        jaro = jaro_similarity(word, village)
        new_list.append((village, jaro))

    sort_list = sorted(new_list, key=lambda x : x[1], reverse=True)

    similar_word = get_similar_score_items(sort_list)

    return (sort_list[0][0], sort_list[0][1], similar_word)


def find_max_jaro_winkler(word: str, source: list):
    """
        Desc:
            Returns the most matched word from the source and its corresponding score using
            jaro wrinklar similarity alogorithm, with an additional list of helper words which have same score
    """

    if not source:
        raise Exception(
            "Source is empty. Check for the spelling of district variable or make sure "
            "that the source file has the corresponding district"
        )

    new_list = []

    for village in source:
        jaro = jaro_winkler_similarity(word, village)
        new_list.append((village, jaro))

    sort_list = sorted(new_list, key=lambda x : x[1], reverse=True)

    similar_word = get_similar_score_items(sort_list)

    return (sort_list[0][0], sort_list[0][1], similar_word)

def unsurpervised_clustering_for_unmapped(new_df):

    """
    new df -> supervised classification using jaro winkler
    flag = True where the item does not fall within the threshold against the base source
    where flag is True, get only those rows and do an intra clusters using Affinity Propogation
    """

    df = new_df[new_df["not_in_source_and_below_threshold"] == True]
    token_df = df[["village_transform"]]
    token_df["freq"] = 1
    token_df = token_df.rename(columns={"village_transform":"token"})
    token_df.reset_index(inplace=True)

    # Add the count of the similar tokens and combine the tokens
    aggregation_functions = {"token": "first", "freq": "sum"}
    token_df_group = token_df.groupby(token_df["token"]).aggregate(aggregation_functions)

    token_df_group.columns = ["word", "freq"]
    token_df_group = token_df_group.sort_values("freq", ascending=False)
    token_df_group.reset_index(inplace=True)

    # Clean tokens
    token_df_group["word"] = token_df_group["word"].apply(cluster_names.clean_tokens)
    token_df_group = token_df_group[token_df_group["word"] != ""]

    token_df_group.set_index("word", inplace=True)

    print("The tokens have been created")

    print("alphabet split segment")
    token_dict = cluster_names.alphabet_split(token_df_group)

    cluster_token_dict = cluster_names.text_similarity(
        token_dict=token_dict,
        DAMPING=0.65,
        THRESHOLD=0.8,
    )

    # Dump it in a csv file
    output = "./unsupervised_second_iteration.csv"
    cluster_token_dataframe = pd.DataFrame([cluster_token_dict]).T
    # cluster_token_dataframe.columns = ["Variations"]
    # cluster_token_dataframe.to_csv(output)

    return cluster_token_dataframe


def append_to_cluster_file(word: str, match_word: str, state: str, district: str, entity_type: str):
    """
    """

    file_path = f"clustering/cluster_variations/{state}/{district}_{entity_type}_clusters.csv"

    df = pd.read_csv(file_path)

    new_list = []

    for i, item in df.iterrows():
        new_dict = {}
        item = item.to_dict()
        
        if not item["variations"] or type(item["variations"]) == float:
            item["variations"] = "[]"

        variations = ast.literal_eval(item["variations"])
        centroid = item["centroid"]

        if match_word == centroid:
            if not (word in variations):
                variations.append(word)

        new_dict["variations"] = variations
        new_dict["centroid"] = centroid

        new_list.append(new_dict)

    new_df = pd.DataFrame(new_list)
    new_df = new_df[["centroid", "variations"]]
    new_df.to_csv(file_path, index=False)


def append_centroids_to_cluster_file(new_cluster_df, **kwargs):
    """
        Desc:
            Appends the cluster df to the file
    """

    new_cluster_df = new_cluster_df.reset_index()


    state = kwargs.get("state", "")
    district = kwargs.get("district", "")
    entity_type = kwargs.get("entity_type", "")

    if not (state and district and entity_type):
        raise Exception("state, district and entity_type should be specified")

    cluster_file = f"clustering/cluster_variations/{state}/{district}_{entity_type}_clusters.csv"

    cluster_df = pd.read_csv(cluster_file)
    new_cluster_df.columns = ["centroid", "variations"]
    new_df = pd.concat([cluster_df, new_cluster_df])
    new_df.to_csv(cluster_file, index=False)


def group_into_clusters(new_df, state, district, entity_type):
    """
        Call this function to group the output df into centroid and their clusters
    """

    if not (state and district and entity_type):
        raise Exception("state, district and entity_type should be specified")

    district = district.lower()
    entity_type = entity_type.lower()    

    village_df = new_df[["village_transform", "village_cleaned"]]
    cluster_group = village_df.groupby('village_cleaned')['village_transform'].apply(list)
    cluster_df = pd.DataFrame(cluster_group)
    cluster_df = cluster_df.reset_index()
    cluster_df["village_transform"] = cluster_df["village_transform"].apply(set)
    cluster_df["village_transform"] = cluster_df["village_transform"].apply(list)

    cluster_df = cluster_df.rename(columns={"village_cleaned":"centroid", "village_transform":"variations"})

    cluster_df.to_csv(f"clustering/cluster_variations/{state}/{district}_{entity_type}_clusters.csv", index=False)

def clean_using_algorithm(clean_dict):
    """
        input will be the clean row in dict format
        district = dict["district_transform"]
    """

    print(clean_dict)

    district = clean_dict["district_transform"]
    # district = "Pune"
    threshold = 0.8
    lev_distance = 1
    state = "Maharashtra"

    source = get_source_village(district)

    split_list = []

    clean_dict["village_transform"] = village_preprocess_and_map.preprocess_with_mapping(clean_dict["village_raw"])

    if type(clean_dict["village_transform"]):
            clean_dict["village_transform"] = str(clean_dict["village_transform"])

    new_dict = {}

    for vill in clean_dict["village_transform"].split(","):
        
        flag = False

        new_vill, jaro, similar_words = find_max_jaro_winkler(vill, source)
        if jaro >= threshold:
            split_list.append(new_vill.title())
        else:
            flag = True
            split_list.append(vill.title()) 

        new_dict["village_cleaned"] = ",".join(split_list)
        new_dict["village_raw"] = clean_dict["village_raw"]
        new_dict["village_transform"] = clean_dict["village_transform"]
        if flag:
            new_dict["not_in_source_and_below_threshold"] = True
        else:
            new_dict["not_in_source_and_below_threshold"] = None
        if similar_words:
            new_dict["possiblitity_of_multiple_sources"] = True
            new_dict["count_of_sources"] = len(similar_words)
            new_dict["help_words"] = similar_words
        else:
            new_dict["possiblitity_of_multiple_sources"] = None
            new_dict["count_of_sources"] = None
            new_dict["help_words"] = None

        new_flag_dict = {}
        req_keys = ["not_in_source_and_below_threshold", "possiblitity_of_multiple_sources", "count_of_sources", "help_words"]

        for key in req_keys:
            if new_dict[key]:
                new_dict["manual_check_required"] = True
            else:
                new_dict["manual_check_required"] = None
            new_flag_dict[key] = new_dict[key]

        new_dict["manual_check_entities"] = new_flag_dict

    final_village_mapped = new_dict["village_cleaned"] # -- will be value for output["village_transform"]

    #second check to find where levenshtein distance = 1 and possiblitity_of_multiple_sources is False  
    #do not change mapping! only change flag to True and add help words if condition is satisfied  
    if new_dict["possiblitity_of_multiple_sources"] != True:

        check_split_list = []

        for vill in new_dict["village_transform"].split(","):
            new_vill, lev, similar_words = find_min_levenshtein(vill, source)
            if lev == lev_distance:
                if similar_words:
                    check_split_list.append(new_vill.title())
                    new_dict["help_words"] = similar_words
                    new_dict["possiblitity_of_multiple_sources"] = True
        
        vill = " ".join(check_split_list) 

    final_verification_dict = {}
    final_verification_dict["jaro_winkler_score"] = jaro
    final_verification_dict["below_jaro_winkler_threshold"] = new_dict["not_in_source_and_below_threshold"]
    final_verification_dict["multiple_sources_match"] = new_dict["possiblitity_of_multiple_sources"]
    final_verification_dict["count_of_multiple_source_match"] = new_dict["count_of_sources"]
    final_verification_dict["help_words"] = new_dict["help_words"]

    # if final_village_mapped:
    #     """
    #         append variation to cluster file
    #     """
    #     append_to_cluster_file(clean_dict["village_transform"], final_village_mapped, state, district.lower(), 'village')

    reclean_status = "1"
    
    print(final_village_mapped)
    print(final_verification_dict)


    return (final_village_mapped, final_verification_dict, reclean_status)


## --- TESTING -----
"""
    The functions below can be used to test the output from the clustering algorithms:
        1. clean_using_algorithm_test -> returns output in a dataframe into a csv file, contains all relevant columns
        2. test_single_word -> return the mapped village after clustering
"""
def clean_using_algorithm_test():
    """
        Newly added for mapping words with numbers

            Gurgram-1 ----> Gurugram 1
    """

    district = "Pune"
    threshold = 0.8
    lev_distance = 1
    state = "Maharashtra"

    source = get_source_village(district)

    if not source:
        print(f"Source Villages are empty for district '{district}'")
        return

    INPUT = f"clustering/villages/clean/preprocessed_{district.lower()}_villages.csv"
    OUTPUT = f"clustering/villages/clean/corrected_{district.lower()}_villages.csv"

    df = pd.read_csv(INPUT)

    new_list = []

    for i, item in df.iterrows():

        new_dict = {}
        item = item.to_dict()

        split_list = []

        item["village_transform"] = village_preprocess_and_map.preprocess_with_mapping(item["village_raw"])

        if type(item["village_transform"]):
            item["village_transform"] = str(item["village_transform"])

        for vill in item["village_transform"].split(","):

            flag = False

            new_vill, jaro, similar_words = find_max_jaro_winkler(vill, source)
            if jaro >= threshold:
                split_list.append(new_vill.title())
            else:
                flag = True
                split_list.append(vill.title())

        new_dict["village_cleaned"] = ",".join(split_list)
        new_dict["village_raw"] = item["village_raw"]
        new_dict["village_transform"] = item["village_transform"]
        if flag:
            new_dict["not_in_source_and_below_threshold"] = True
        else:
            new_dict["not_in_source_and_below_threshold"] = None
        if similar_words:
            new_dict["possiblitity_of_multiple_sources"] = True
            new_dict["count_of_sources"] = len(similar_words)
            new_dict["help_words"] = similar_words
        else:
            new_dict["possiblitity_of_multiple_sources"] = None
            new_dict["count_of_sources"] = None
            new_dict["help_words"] = None

        new_flag_dict = {}
        req_keys = ["not_in_source_and_below_threshold", "possiblitity_of_multiple_sources", "count_of_sources", "help_words"]

        for key in req_keys:
            if new_dict[key]:
                new_dict["manual_check_required"] = True
            else:
                new_dict["manual_check_required"] = None
            new_flag_dict[key] = new_dict[key]

        new_dict["manual_check_entities"] = new_flag_dict

        new_list.append(new_dict)

    new_df = pd.DataFrame(new_list)

    new_df = new_df[["village_raw", "village_transform", "village_cleaned", "not_in_source_and_below_threshold", "possiblitity_of_multiple_sources", "count_of_sources", "help_words"]]

    #second check to find where levenshtein distance = 1 and possiblitity_of_multiple_sources is False
    old_df = new_df[new_df["possiblitity_of_multiple_sources"] == True]
    check_df = new_df[new_df["possiblitity_of_multiple_sources"] != True]

    check_split_list = []

    for i, item in check_df.iterrows():
        check_dict = {}
        item = item.to_dict()
        check_new_vill_split = []

        for vill in item["village_transform"].split(","):
            new_vill, lev, similar_words = find_min_levenshtein(vill, source)
            if lev == lev_distance:
                if similar_words:
                    check_dict["not_in_source_and_below_threshold"] = item["not_in_source_and_below_threshold"]
                    check_dict["possiblitity_of_multiple_sources"] = True
                    check_dict["count_of_sources"] = len(similar_words)
                    check_dict["help_words"] = similar_words
                    check_new_vill_split.append(new_vill.title())
                else:
                    check_dict["not_in_source_and_below_threshold"] = item["not_in_source_and_below_threshold"]
                    check_dict["possiblitity_of_multiple_sources"] = None
                    new_dict["count_of_sources"] = None
                    new_dict["help_words"] = None
                    check_new_vill_split.append(vill.title())
            else:
                check_dict["not_in_source_and_below_threshold"] = item["not_in_source_and_below_threshold"]
                check_dict["possiblitity_of_multiple_sources"] = None
                new_dict["count_of_sources"] = None
                new_dict["help_words"] = None
                check_new_vill_split.append(vill.title())
        
        vill = ",".join(check_new_vill_split)
        check_dict["village_cleaned"] = vill
        check_dict["village_transform"] = item["village_transform"]
        check_dict["village_raw"] = item["village_raw"]
        check_split_list.append(check_dict)

    new_check_df = pd.DataFrame(check_split_list)
    new_df = pd.concat([old_df, new_check_df])

    new_df.to_csv(OUTPUT, index=False)

    group_into_clusters(new_df, state, district, "village") #make the initial cluster file in cluster_variations/Maharashtra/ folder

    unsupervised_cluster_df = unsurpervised_clustering_for_unmapped(new_df)

    append_centroids_to_cluster_file(unsupervised_cluster_df, state=state, district=district.lower(), entity_type="village")

def test_single_word():

    item = {}

    item["village_raw"] = "Umara"
    item["village_transform"] = "Umara"
    
    district = "Thane"
    threshold = 0.8
    lev_distance = 1

    split_list = []
    new_dict = {}

    source = get_source_village(district)

    for vill in item["village_transform"].split(","):

        flag = False

        new_vill, jaro, similar_words = find_max_jaro_winkler(vill, source)
        if jaro >= threshold:
            split_list.append(new_vill.title())
        else:
            flag = True
            split_list.append(vill.title())

    new_dict["village_cleaned"] = ",".join(split_list)
    new_dict["village_raw"] = item["village_raw"]
    new_dict["village_transform"] = item["village_transform"]

    if flag:
        new_dict["not_in_source_and_below_threshold"] = True
    else:
        new_dict["not_in_source_and_below_threshold"] = None

    if similar_words:
        new_dict["possiblitity_of_multiple_sources"] = True
        new_dict["count_of_sources"] = len(similar_words)
        new_dict["help_words"] = similar_words
    else:
        new_dict["possiblitity_of_multiple_sources"] = None
        new_dict["count_of_sources"] = None
        new_dict["help_words"] = None

    item = {}

    for key in new_dict:
        item[key] = new_dict[key]

    if item["possiblitity_of_multiple_sources"] != True:
        check_dict = {}

        check_new_vill_split = []

        for vill in item["village_transform"].split(","):

            new_vill, lev, similar_words = find_min_levenshtein(vill, source)

            if lev == lev_distance:
                if similar_words:
                    check_dict["not_in_source_and_below_threshold"] = item["not_in_source_and_below_threshold"]
                    check_dict["possiblitity_of_multiple_sources"] = True
                    check_dict["count_of_sources"] = len(similar_words)
                    check_dict["help_words"] = similar_words
                    check_new_vill_split.append(new_vill.title())
                else:
                    check_dict["not_in_source_and_below_threshold"] = item["not_in_source_and_below_threshold"]
                    check_dict["possiblitity_of_multiple_sources"] = None
                    new_dict["count_of_sources"] = None
                    new_dict["help_words"] = None
                    check_new_vill_split.append(vill.title())
            else:
                check_dict["not_in_source_and_below_threshold"] = item["not_in_source_and_below_threshold"]
                check_dict["possiblitity_of_multiple_sources"] = None
                new_dict["count_of_sources"] = None
                new_dict["help_words"] = None
                check_new_vill_split.append(vill.title())
        
        vill = ",".join(check_new_vill_split)

        check_dict["village_cleaned"] = vill
        check_dict["village_transform"] = item["village_transform"]
        check_dict["village_raw"] = item["village_raw"]

    print(check_dict)

if __name__ == "__main__":

    d = {
        "district_transform":"Pune",
        "village_raw":"Chanholi bu",
        # "village_transform":"Saalai"
    }

    # clean_using_algorithm_test()

    clean_using_algorithm(d)
    # test_single_word()
    # append_to_cluster_file("Sawalai", "Sawali", "Maharashtra", "wardha", "village")
