from json import load, dump
import numpy as np
from numpy import dot
from numpy.linalg import norm
from data_preparator import search_homonyms


class SimilarWordSearcher:
    """
    Класс для поиска наиболее похожего слова в словаре глоссов
    """

    def __init__(self, gloss_filename: str, test_filename: str) -> None:
        """
        Функция инициализации класса
        Parameters
        ----------
        gloss_filename: str
            путь к json-файлу с векторным представлением глоссов
        test_filename: str
            путь к json-файлу с векторным представлением тестовых слов
        """
        self.gloss_filename = gloss_filename
        self.test_filename = test_filename

    def __read_json_to_dict(self, filename) -> dict:
        """
        Функция для чтения json-файл в словарь

        Parameters
        ----------
        filename: str
            путь к json-файлу

        Returns
        -------
        dict
            словарь с векторным представлением слова
        """
        with open(filename, 'r', encoding='utf-8') as json_file:
            data_dict = load(json_file)

        for key, value in data_dict.items():
            data_dict[key] = np.array(value, dtype=np.float32)

        return data_dict

    def find_similar_words(self, not_found: list, trash_hold: float = 0.5) -> dict:
        """
        Функция для нахождения наиболее похожего слова среди слов из словаря

        Parameters
        ----------
        not_found: list
            список слов, которые не нашлись в модели
        trash_hold: float
            порог, для отбора слов по косинусному расстоянию
            остаются слова со значением больше trash_hold

        Returns
        -------
        dict
            словарь "входное слово": "наиболее похожее слово"
        """
        gloss_dict = self.__read_json_to_dict(self.gloss_filename)
        test_dict = self.__read_json_to_dict(self.test_filename)
        result_dict = {}

        # Добавим ненайденные слова в результаты
        if isinstance(not_found, list):
            for word in not_found:
                result_dict[word] = 'None'

        for word in test_dict:
            # проверяем есть ли слово в словаре глоссов
            if word in gloss_dict:
                result_dict[word] = word
            else:
                similarity_lst = []
                for gloss, vec in gloss_dict.items():
                    # считаем косинусное расстояние
                    cos_sim = dot(test_dict[word], gloss_dict[gloss]) / (
                            norm(test_dict[word]) * norm(gloss_dict[gloss]))
                    if cos_sim > trash_hold:
                        similarity_lst.append((gloss, cos_sim))

                if len(similarity_lst) == 0:
                    result_dict[word] = 'None'
                else:
                    # оставляем топ 10 отсортированных по косинусному расстоянию слов
                    similarity_lst = sorted(similarity_lst, key=lambda x: x[1], reverse=True)[:10]
                    similarity_dict = {x[0]: x[1] for x in similarity_lst}
                    # ищем среди наиболее близких однокоренные
                    result_dict[word] = search_homonyms(similarity_dict, word)
        return result_dict

    def save_dict_to_json(self, result_filename: str, result_dict: dict) -> None:
        """
        Функция для сохранения словаря в json

        Parameters
        ----------
        result_filename: str
            путь до файла с результатом
        result_dict: dict
            словарь с результатами
        """
        with open(result_filename, 'w', encoding='utf-8') as json_file:
            dump(result_dict, json_file, ensure_ascii=False, indent=2)
        print(f"Done! Result is saved into {result_filename}")
