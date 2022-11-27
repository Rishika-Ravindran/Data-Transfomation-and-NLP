"""
    preprocessing
    precompute cluster and keep the results

"""


import re
import pandas as pd

from general_transformer.entity_mapper import EntityMapper
from address_transformer.address_translator import AddressTranslator


translator = AddressTranslator(languages=["marathi"])
village_mapper = EntityMapper(mapper_type="village", languages=["marathi"])


def write_to_csv(df, file_path: str, col_order: list = []):
    if col_order:
        df = df[col_order]
    df.to_csv(file_path, index=False)


def preprocess(text):

    if not text:
        return ""

    text = text.split(',')
    text_list = [i.strip() for i in text if i.strip()]
    new_list = []
    for text in text_list:
        text = re.sub(r" {2,}", " ", text)
        text = re.sub(rf"\d\)", f", ", text)
        text = re.sub(rf"\d$", f" ", text)
        text = re.sub(rf" ane ", ",", text.lower())
        text = re.sub(r" no\.", " ", text)
        text = re.sub(r" no", " ", text)
        # text = re.sub(r"\.", " ", text)
        text = text.strip()
        new_list.append(text)
    text = list(set(new_list))
    text = ','.join(text).strip()

    return text


def remove_characters(word: str, from_: str, to_: str):
    """
    """

    start = word.find(from_)

    if start == -1:
        return word

    end = word.find(to_)

    if end == -1:
        del_str = word[start:]
    else:
        del_str = word[start: end+1]

    # print(word)
    # print(del_str)

    word = word.replace(del_str, "")

    return word


def second_preprocess(text: str):
    """
    """

    if not text:
        return ""

    text = re.sub(r"^\,", "", text)
    text = remove_characters(text, "(", ")")
    text = remove_characters(text, "(", ")")
    text = re.sub(r"\)", " ", text)
    text = re.sub(r"\(", " ", text)
    text = remove_characters(text, "[", "]")
    text = remove_characters(text, "[", "]")
    text = re.sub(r"\]", " ", text)
    text = re.sub(r"\[", " ", text)
    text = re.sub(r" नं(\.)", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r" न ", " no ", text).strip()

    return text

def postprocess(word: str):
    if not word:
        return ""
    word = re.sub(r"\(\)", " ", word)
    word_list = word.split(",")
    new_list = []
    for w in word_list:
        w = w.strip().title()
        new_list.append(w)

    processed_word = ",".join(new_list)
    processed_word = re.sub("Mauja", " ", processed_word).strip()
    processed_word = re.sub("Mauje", " ", processed_word).strip()
    processed_word = re.sub("Budruk", "Bk", processed_word).strip()
    processed_word = re.sub("Khurd", "Kh", processed_word).strip()
    processed_word = re.sub("While", "Vela", processed_word).strip()
    processed_word = re.sub("Vaayangia", "Waigaon", processed_word).strip()

    return processed_word

def clean_after_all_mapping(word: str):
    """
    """
    if not word:
        return ""

    word = re.sub(r"मौजे", " ", word)
    word = re.sub(r"विभागाचे", " ", word)
    word = re.sub(r"नव्याने", " ", word)
    word = re.sub(r"समाविष्ट", " ", word)

    word = word.strip().title()

    return word


def map_words_with_numbers(word: str):
    word_list = word.split()

    new_list = []

    for w in word_list:
        w = w.strip()
        mapped = map_whole_phrase(w)
        if mapped:
            new_list.append(mapped)
        else:
            new_list.append(w)

    processed_word = " ".join(new_list)

    return processed_word


def flag_multiple(word: str, stats: dict, new_dict):
    if not word:
        return

    if "," in word:
        new_dict["multiple"] = True


def flag_english_words_not_transformed(word: str, stats: dict, new_dict):
    """
    """

    # if new_dict["transformed"] is not None:
    #     return

    # if new_dict["mapped"] is not None:
    #     return

    if re.search(r'[a-zA-Z]+', word):
        stats["mapped"] += 1
        new_dict["transformed"] = False
        new_dict["mapped"] = True
        new_dict["transliterated"] = False
        new_dict["village_transform"] = word
        flag_multiple(word, stats, new_dict)


def flag_english_words_transformed(word: str, stats:dict, new_dict):
    """
    """

    if not word:
        return ""

    # if new_dict["transformed"] is not None:
    #     return

    # if new_dict["mapped"] is not None:
    #     return

    if re.search(r'[a-zA-Z]+', word):
        stats["mapped"] += 1
        new_dict["transformed"] = True
        new_dict["mapped"] = True
        new_dict["transliterated"] = False
        new_dict["village_transform"] = word.title()
        flag_multiple(word, stats, new_dict)


def transliterate_word(word: str):
    """
    """

    return translator.transliterate_address(word)


def map_whole_phrase(word: str):
    """
    """

    return village_mapper.map_entity(word)


def map_multiple_words(word: str, stats: dict, new_dict):
    """
    """

    if not ("," in  word):
        return

    TRANSLITERATED = False

    string_join_list = []

    for i in word.split(','):
        split_mapped = village_mapper.map_entity(i)
        if not split_mapped:
            i = transliterate_word(i)
            string_join_list.append(i)
            TRANSLITERATED = True
            new_dict["mapped"] = False
        else:
            string_join_list.append(split_mapped)

    final_str = ",".join(string_join_list)

    if TRANSLITERATED:
        stats["unmapped"] += 1
        new_dict["village_transform"] = final_str
        new_dict["mapped"] = False
        new_dict["transliterated"] = True
    else:
        stats["mapped"] += 1
        new_dict["village_transform"] = final_str
        new_dict["mapped"] = True
        new_dict["transliterated"] = False

    flag_multiple(final_str, stats, new_dict)


def preprocess_with_mapping(village_raw: str):
    """
        Desc:
            Returns:
                A preprocessed and KB mapped village string
    """

    stats = {
        'mapped' : 0,
        'unmapped' : 0,
        'multiple' : 0
    }

    item = {}

    item["village_raw"] = village_raw

    new_dict = {
        "village_raw": None,
        "village_transform": None,
        "mapped": None,
        "transformed": None,
        "transliterated": None,
        "multiple": None,
    }

    village_raw = item["village_raw"]
    village_raw = preprocess(village_raw)
    flag_english_words_not_transformed(village_raw, stats, new_dict)
    mapped_string = map_whole_phrase(village_raw)
    flag_english_words_transformed(mapped_string, stats, new_dict)

    if mapped_string:
        new_dict["village_raw"] = item["village_raw"]
        for key in new_dict:
            if new_dict[key] is None:
                new_dict[key] = False
        
        return new_dict["village_transform"]

    map_multiple_words(village_raw, stats, new_dict)

    for key in new_dict:
        if new_dict[key] is None:
            new_dict[key] = False

    new_dict["village_raw"] = item["village_raw"]

    if new_dict["transformed"] is False:

        new_dict["village_transform"] = item["village_raw"]
        new_dict["village_transform"] = new_dict["village_transform"].split(',')
        name_list = []
        for i in new_dict["village_transform"]:
            name = i.strip().title()
            name_list.append(name)
        name_list = list(set(name_list))
        new_dict["village_transform"] = name_list
        new_dict["village_transform"] = ', '.join(new_dict["village_transform"])

        # New change in the process flow

        village_raw = new_dict["village_transform"]
        village_raw = postprocess(village_raw)
        
        village_raw = second_preprocess(village_raw)

        temp_list = []
        for i in village_raw.split(","):
            name = i.strip().title()
            temp_list.append(name)

        village_raw = ",".join(temp_list)

        mapped_string = map_whole_phrase(village_raw)


        new_dict["village_transform"] = mapped_string

        if not mapped_string:
            village_trans = map_words_with_numbers(village_raw)
            new_dict["village_transform"] = village_trans

        new_dict["village_transform"] = clean_after_all_mapping(new_dict["village_transform"])

        marking = new_dict["village_transform"]

        if re.search(r"[^\x00-\x7F]", marking):
            new_dict["transliterated"] = True
        if new_dict["village_transform"]:
            new_dict["village_transform"] = new_dict["village_transform"].title()
        new_dict["village_transform"] = translator.transliterate_address(new_dict["village_transform"])

        new_dict["village_transform"] = postprocess(new_dict["village_transform"])

        new_dict["village_transform"] = re.sub(r"\d+$", " ", new_dict["village_transform"]).strip()

        if new_dict["village_transform"]:
            new_dict["village_transform"] = new_dict["village_transform"].title()

        return new_dict["village_transform"]

def preprocess_with_mapping_test():

    path_dict = {
        "pune": {
            "input": "clustering/villages/raw/pune_villages.csv",
            "output": "clustering/villages/clean/preprocessed_pune_villages.csv",
        },
        "solapur": {
            "input": "clustering/villages/raw/solapur_villages.csv",
            "output": "clustering/villages/clean/preprocessed_solapur_villages.csv"
        },
        "thane": {
            "input": "clustering/villages/raw/thane_villages.csv",
            "output": "clustering/villages/clean/preprocessed_thane_villages.csv",
        },
        "mumbai": {
            "input": "clustering/villages/raw/mumbai_villages.csv",
            "output": "clustering/villages/clean/preprocessed_mumbai_villages.csv"
        },
        "wardha": {
            "input": "clustering/villages/raw/wardha_villages.csv",
            "output": "clustering/villages/clean/preprocessed_wardha_villages.csv"
        },
        "raigad": {
            "input": "clustering/villages/raw/raigad_villages.csv",
            "output": "clustering/villages/clean/preprocessed_raigad_villages.csv"
        }
    }

    geography = "pune"

    INPUT_FILE_PATH = path_dict[geography]["input"]
    OUTPUT_FILE_PATH = path_dict[geography]["output"]

    stats = {
        'mapped' : 0,
        'unmapped' : 0,
        'multiple' : 0
    }

    df = pd.read_csv(INPUT_FILE_PATH)

    new_list = []

    for i, item in df.iterrows():
        new_dict = {
            "village_raw": None,
            "village_transform": None,
            "mapped": None,
            "transformed": None,
            "transliterated": None,
            "multiple": None,
        }
        item = item.to_dict()
        village_raw = item["village_raw"]
        village_raw = preprocess(village_raw)
        flag_english_words_not_transformed(village_raw, stats, new_dict)
        mapped_string = map_whole_phrase(village_raw)
        flag_english_words_transformed(mapped_string, stats, new_dict)

        if mapped_string:
            new_dict["village_raw"] = item["village_raw"]
            new_list.append(new_dict)
            for key in new_dict:
                if new_dict[key] is None:
                    new_dict[key] = False
            continue

        map_multiple_words(village_raw, stats, new_dict)

        for key in new_dict:
            if new_dict[key] is None:
                new_dict[key] = False

        new_dict["village_raw"] = item["village_raw"]
        if new_dict["transformed"] is False:
            # new_dict["village_transform"] = translator.transliterate_address(item["village_raw"])
            new_dict["village_transform"] = item["village_raw"]
            new_dict["village_transform"] = new_dict["village_transform"].split(',')
            name_list = []
            for i in new_dict["village_transform"]:
                name = i.strip().title()
                name_list.append(name)
            name_list = list(set(name_list))
            new_dict["village_transform"] = name_list
            new_dict["village_transform"] = ', '.join(new_dict["village_transform"])
            new_dict["village_transform"] = re.sub("Budruk", "Bk", new_dict["village_transform"])
            new_dict["village_transform"] = re.sub("Khurd", "Kh", new_dict["village_transform"])
            new_dict["village_transform"] = re.sub("Mauje", " ", new_dict["village_transform"]).strip()
            new_dict["village_transform"] = re.sub("Mauja", " ", new_dict["village_transform"]).strip()
            new_dict["village_transform"] = re.sub("While", "Vela", new_dict["village_transform"]).strip()

        # New change in the process flow

        village_raw = new_dict["village_transform"]
        village_raw = postprocess(village_raw)
        village_raw = second_preprocess(village_raw)

        temp_list = []
        for i in village_raw.split(","):
            name = i.strip().title()
            temp_list.append(name)

        village_raw = ",".join(temp_list)

        mapped_string = map_whole_phrase(village_raw)

        new_dict["village_transform"] = mapped_string

        if not mapped_string:
            village_trans = map_words_with_numbers(village_raw)
            new_dict["village_transform"] = village_trans

        new_dict["village_transform"] = clean_after_all_mapping(new_dict["village_transform"])

        marking = new_dict["village_transform"]

        if re.search(r"[^\x00-\x7F]", marking):
            new_dict["transliterated"] = True
        if new_dict["village_transform"]:
            new_dict["village_transform"] = new_dict["village_transform"].title()
        new_dict["village_transform"] = translator.transliterate_address(new_dict["village_transform"])

        new_dict["village_transform"] = postprocess(new_dict["village_transform"])

        new_dict["village_transform"] = re.sub(r"\d+$", " ", new_dict["village_transform"]).strip()
        
        if new_dict["village_transform"]:
            new_dict["village_transform"] = new_dict["village_transform"].title()
        # print(new_dict["village_transform"])

        new_list.append(new_dict)

    new_df = pd.DataFrame(new_list)
    # write_to_csv(new_df, OUTPUT_FILE_PATH, ["village_raw", "village_transform", "mapped", "transformed", "transliterated", "multiple"])
    write_to_csv(new_df, OUTPUT_FILE_PATH, ["village_raw", "village_transform", "transliterated"])



if __name__ == "__main__":

    # word = "महादापूर , वेळा ऊर्फ वेला"

    # a = preprocess_with_mapping(word)

    # print(a)

    preprocess_with_mapping_test()