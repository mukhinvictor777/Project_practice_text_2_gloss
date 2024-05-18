import pandas as pd
import numpy as np
from json import dump
import re
from word_analyzer import WordAnalyzer


def clean_text(text):
    """Функция оставляет только буквы, цифры, пробелы и слэш с помощью регулярного выражения
    :param text: str
        текст для обработки
    :return:
    """
    pattern = r'[^a-zA-Zа-яА-Я0-9 /]'
    cleaned_text = re.sub(pattern, '', text)
    return cleaned_text


def set_type(row):
    if row["composite"]:
        return "composite"
    elif row["synonym"]:
        return "synonym"
    else:
        return "single"


def search_homonyms(similar_words: dict, word: str):
    analyzer = WordAnalyzer()

    homonyms = analyzer.find_related_words(word,
                                           list(
                                               similar_words.keys()))  # Преобразуем представление ключей в список
    homonyms_dict = {word: similar_words[word] for word in similar_words.keys() if word in homonyms}

    if len(homonyms) < 1:
        return list(similar_words.keys())[0]
    else:
        return list(homonyms_dict.keys())[0]


class DataPreparator:
    def __init__(self,
                 model,
                 input_filename="RSL_class_list.txt",
                 output_filename="gloss_vec.json",
                 ):
        """Функция инициализации класса"""

        self.model = model
        self.input_filename = input_filename
        self.output_filename = output_filename
        self.not_found_words = None

    def __load_data(self):
        df = pd.read_csv(self.input_filename,
                         sep="\t",
                         names=['word'])
        return df

    def __words_typing(self, df):

        df['word'] = df['word'].apply(lambda x: clean_text(x))
        df['word'] = df['word'].apply(lambda x: x.replace("ё", "е"))

        df['composite'] = df['word'].apply(lambda x: True if len(x.split()) > 1 else False)
        df['synonym'] = df['word'].apply(lambda x: True if '/' in x else False)

        df = df[df['word'].str.split().apply(len) < 4]
        df = df[df['word'].str.split("/").apply(len) < 4]

        letters_to_remove = list('бгдежзйлмнопртфхцчьэыъю')
        # Создаем маску для удаления строк
        mask = df['word'].isin(letters_to_remove)
        # Удаляем строки
        df = df[~mask]

        df = df.query('word != ""')

        df = df.assign(word_type=df.apply(set_type, axis=1))
        return df

    def __search_similar_word_in_model(self, word: str) -> str:
        try:
            most_similar = self.model.most_similar(word)
            if len(most_similar) < 1:
                return None
            else:
                similar_words = {similar[0]: similar[1] for similar in most_similar}  # создание словаря похожих слов
                return search_homonyms(similar_words, word)
        except (ValueError, KeyError):
            return None

    def __get_vector(self, x, column):
        key = x[column]
        if key in self.model:
            return self.model[key]
        else:
            return self.__search_similar_word_in_model(key)

    def __add_synonyms_composit_columns(self, df):
        # Создаем столбцы для синонимов и для их векторных представлений
        for i in range(1, 4):
            df[f'synonym_{i}'] = df.apply(
                lambda x: x['word'].split('/')[i - 1] if x['synonym'] and len(x['word'].split('/')) >= i else None,
                axis=1)
            df[f'synonym_{i}_vec'] = df.apply(lambda x: self.__get_vector(x, f'synonym_{i}') if x['synonym'] else '',
                                              axis=1)

        # Создаем столбцы для составных слов и для их векторных представлений
        for i in range(1, 4):
            df[f'composite_{i}'] = df.apply(
                lambda x: x['word'].split()[i - 1] if x['composite'] and i <= len(x['word'].split()) < 4 else None,
                axis=1)
            df[f'composite_{i}_vec'] = df.apply(
                lambda x: self.__get_vector(x, f'composite_{i}') if x['composite'] else None,
                axis=1)

        return df

    def __get_vec_for_composite(self, df):
        df_comp = df.query("word_type == 'composite'")
        df_comp = df_comp.fillna("")

        df_comp["composite_3_vec"] = df_comp["composite_3_vec"].apply(lambda x: [0.0] * 100 if x == "" else x)

        # Добавление новой колонки с суммой векторов
        df_comp['vec'] = df_comp.apply(lambda row:
                                       np.round(np.array(row['composite_1_vec']) +
                                                np.array(row['composite_2_vec']) +
                                                np.array(row['composite_3_vec']), 6), axis=1)

        return df_comp[["word", "vec"]]

    def __get_vec_for_synonym(self, df):
        df_filtered = df.query("word_type == 'synonym'")

        df1 = df_filtered.query("synonym_1 != ''")
        df1 = df1[["word", "synonym_1_vec"]]
        df1 = df1.rename(columns={"synonym_1_vec": "vec"})

        df2 = df_filtered.query("synonym_2 != ''")
        df2 = df2[["word", "synonym_2_vec"]]
        df2 = df2.rename(columns={"synonym_2_vec": "vec"})

        df3 = df_filtered.query("synonym_3 != ''")
        df3 = df3[["word", "synonym_3_vec"]]
        df3 = df3.rename(columns={"synonym_3_vec": "vec"})
        df3 = df3[~df3["vec"].isna()]

        res = pd.concat([df1, df2, df3], ignore_index=True)
        return res

    def __get_all_vec(self, df):
        df_single = df.query("word_type == 'single'")
        df_single['vec'] = df_single.apply(lambda x: self.__get_vector(x, 'word'), axis=1)
        df_single = df_single[["word", "vec"]]

        synonyms = self.__get_vec_for_synonym(df)
        composite = self.__get_vec_for_composite(df)

        res = pd.concat([df_single, synonyms, composite], ignore_index=True)

        res_not_na = res[~res["vec"].isna()]
        res_na = res[res["vec"].isna()]["word"].to_list()

        return res_not_na, res_na

    def __dataframe_to_json(self, df):
        df['vec'] = df['vec'].apply(list)
        dict_vec = df.set_index('word')['vec'].to_dict()
        for key, val in dict_vec.items():
            new_val = list(map(str, val))
            dict_vec[key] = new_val
        with open(self.output_filename, 'w', encoding='utf-8') as json_file:
            dump(dict_vec, json_file, ensure_ascii=False, indent=2)

    def create_vected_words(self):
        df_raw = self.__load_data()
        df_typed = self.__words_typing(df_raw)
        df_with_extra_cols = self.__add_synonyms_composit_columns(df_typed)
        df_vec, lst_na = self.__get_all_vec(df_with_extra_cols)
        self.not_found_words = lst_na

        self.__dataframe_to_json(df_vec)

