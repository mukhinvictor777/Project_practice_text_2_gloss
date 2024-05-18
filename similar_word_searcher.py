from json import load, dump
import numpy as np
from data_preparator import search_homonyms


class SimilarWordSearcher:
    def __init__(self, gloss_filename, test_filename):
        self.gloss_filename = gloss_filename
        self.test_filename = test_filename

    def __read_json_to_dict(self, filename) -> dict:
        with open(filename, 'r', encoding='utf-8') as json_file:
            data_dict = load(json_file)

        for key, value in data_dict.items():
            data_dict[key] = np.array(value, dtype=np.float32)

        return data_dict

    def find_similar_words(self, not_found, trash_hold=0.5):
        gloss_dict = self.__read_json_to_dict(self.gloss_filename)
        test_dict = self.__read_json_to_dict(self.test_filename)
        result_dict = {}

        # Добавим ненайденные слова в результаты
        if isinstance(not_found, list):
            for word in not_found:
                result_dict[word] = 'None'

        for word in test_dict:
            if word in gloss_dict:
                result_dict[word] = word
            else:
                similarity_lst = []
                for gloss, vec in gloss_dict.items():
                    cos_sim = np.dot(test_dict[word], gloss_dict[gloss]) / (
                            np.linalg.norm(test_dict[word]) * np.linalg.norm(gloss_dict[gloss]))
                    if cos_sim > trash_hold:
                        similarity_lst.append((gloss, cos_sim))

                if len(similarity_lst) == 0:
                    result_dict[word] = 'None'
                else:
                    similarity_lst = sorted(similarity_lst, key=lambda x: x[1], reverse=True)[:10]
                    similarity_dict = {x[0]: x[1] for x in similarity_lst}
                    result_dict[word] = search_homonyms(similarity_dict, word)
        return result_dict

    def save_dict_to_json(self, result_filename, result_dict):
        with open(result_filename, 'w', encoding='utf-8') as json_file:
            dump(result_dict, json_file, ensure_ascii=False, indent=2)
