import pandas as pd
import numpy as np
from json import dump
import re

from pandas import Series, DataFrame

from word_analyzer import WordAnalyzer
from typing import Union, Tuple, Any


def clean_text(text: str) -> str:
    """
    Функция оставляет только буквы, цифры, пробелы и слэш с помощью регулярного выражения

    Parameters
    ----------
    text: str
        текст для обработки

    Returns
    -------
    str
        текст очищенный от лишних символов
    """
    pattern = r'[^a-zA-Zа-яА-Я0-9 /]'
    cleaned_text = re.sub(pattern, '', text)
    return cleaned_text


def set_type(row) -> str:
    """
    Функция определяет к какому типу относится слово - single, composite или synonym

    Parameters
    ----------
    row:
        строка датафрейма

    Returns
    -------
    str
        тип слова
    """
    if row["composite"]:
        return "composite"
    elif row["synonym"]:
        return "synonym"
    else:
        return "single"


def search_homonyms(similar_words: dict, word: str) -> str:
    """
    Функция для нахождения однокоренных слов

    Parameters
    ----------
    similar_words: dict
        словарь с наиболее близкими по косинусному расстоянию словами
    word: str
        слово, которому ищем ближайшее

    Returns
    -------
    str
        наиболее подходящее слово
    """
    analyzer = WordAnalyzer()

    # поиск однокоренных слов среди наиболее близких
    homonyms = analyzer.find_related_words(word, list(similar_words.keys()))
    homonyms_dict = {word: similar_words[word] for word in similar_words.keys() if word in homonyms}

    # если однокоренных нет
    if len(homonyms) < 1:
        return list(similar_words.keys())[0]
    else:
        return list(homonyms_dict.keys())[0]


class DataPreparator:
    """
    Класс для обработки данных.
    На вход подается файл .txt, на выходе .json,
    где ключ - слово, значение - векторное представление
    """
    def __init__(self,
                 model,
                 input_filename="input/RSL_class_list.txt",
                 output_filename="vected_data/gloss_vec.json",
                 ):
        """
        Функция инициализации класса

        Parameters
        ----------
        model:
            модель для создания векторных представлений
        input_filename: str
            название входного .txt файла
        output_filename: str
            название выходного .json файла
        """

        self.model = model
        self.input_filename = input_filename
        self.output_filename = output_filename

        # здесь будем хранить слова, которые не нашлись в модели
        self.not_found_words = None

    def __load_data(self) -> pd.DataFrame:
        """
        Функция для загрузки данных в формате .txt

        Returns
        -------
        pd.DataFrame
            pandas датафрейм со столбцом word
        """
        df = pd.read_csv(self.input_filename,
                         sep="\t",
                         names=['word'])
        return df

    def __words_typing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Функция для очистки данных и определения типа слова:
        одиночное (single), синоним (synonym), составное (composite)

        Parameters
        ----------
        df: pd.DataFrame
            датафрейм со словами, которые надо обработать

        Returns
        -------
        pd.DataFrame
            датафрейм с очищенными данными и определенным типом для каждого слова
        """

        # удаляем лишние символы
        df['word'] = df['word'].apply(lambda x: clean_text(x))
        df['word'] = df['word'].apply(lambda x: x.replace("ё", "е"))

        # создаем дополнительные столбцы
        # если слово состоит из нескольких - composite
        # если содержит '/' - synonym
        df['composite'] = df['word'].apply(lambda x: True if len(x.split()) > 1 else False)
        df['synonym'] = df['word'].apply(lambda x: True if '/' in x else False)

        df = df[df['word'].str.split().apply(len) < 4]
        df = df[df['word'].str.split("/").apply(len) < 4]

        # удаляем слова из одной буквы
        letters_to_remove = list('бгдежзйлмнопртфхцчьэыъю')
        # маска для удаления строк
        mask = df['word'].isin(letters_to_remove)
        df = df[~mask]

        df = df.query('word != ""')

        # создаем столбец с типом слова
        df = df.assign(word_type=df.apply(set_type, axis=1))
        return df

    def __search_similar_word_in_model(self, word: str) -> str:
        """
        Функция для поиска наиболее похожих по вектору слов в модели

        Parameters
        ----------
        word: str
            слово для поиска

        Returns
        -------
        str
            возвращает наиболее похожее слово или None
        """
        try:
            most_similar = self.model.most_similar(word)
            if len(most_similar) < 1:
                return None
            else:
                # создание словаря похожих слов
                similar_words = {similar[0]: similar[1] for similar in most_similar}
                return search_homonyms(similar_words, word)
        except (ValueError, KeyError):
            return None

    def __get_vector(self, x, column: str) -> Union[list, str]:
        """
        Функция для нахождения векторного представления слова в модели

        Parameters
        ----------
        x:
            строка в датафрейме
        column: str
            колонка со словом, для которого нужно найти векторное представление

        Returns
        -------
        list
            векторное представление для слова
        """
        key = x[column]
        if key in self.model:
            return self.model[key]
        else:
            return self.__search_similar_word_in_model(key)

    def __add_synonyms_composit_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Функция добавляет дополнительные столбцы для составных слов и синонимов,
        чтобы в дальнейшем получить для каждого отдельного слова векторное представление

        Parameters
        ----------
        df: pd.DataFrame
            датафрейм с входными данными и определенным типом слова

        Returns
        -------
        pd.DataFrame
            датафрейм с дополнительными колонками для составных слов и синонимов
        """

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

    def __get_vec_for_composite(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Функция для получения векторного представления составных слов

        Parameters
        ----------
        df: pd.DataFrame
            датафрейм со словами

        Returns
        -------
        pd.DataFrame
            датафрейм со словами и их векторными представлениями
        """

        df_comp = df.query("word_type == 'composite'")
        df_comp = df_comp.fillna("")

        # не все составные слова состоят из 3 слов
        # поэтому заполняем пропуски вектором из ста нулей для третьего слова
        df_comp["composite_3_vec"] = df_comp["composite_3_vec"].apply(lambda x:
                                                                      [0.0] * 100 if isinstance(x, str) and x == ""
                                                                      else x)

        # Добавление новой колонки с суммой векторов
        df_comp['vec'] = df_comp.apply(lambda row:
                                       np.round(np.array(row['composite_1_vec']) +
                                                np.array(row['composite_2_vec']) +
                                                np.array(row['composite_3_vec']), 6), axis=1)

        return df_comp[["word", "vec"]]

    def __get_vec_for_synonym(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Функция для получения векторного представления синонимов
        Вектор ищется отдельно для каждого из синонимов

        Parameters
        ----------
        df: pd.DataFrame
            датафрейм со словами

        Returns
        -------
        pd.DataFrame
            датафрейм со словами и их векторными представлениями
        """
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

    def __get_all_vec(self, df: pd.DataFrame) -> tuple[Any, Any]:
        """
        Функция для сбора векторов всех типов слов в один датафрейм

        Parameters
        ----------
        df: pd.DataFrame
            датафрейм с входными данными

        Returns
        -------
        pd.DataFrame
            датафрейм со словами и их векторными представлениями
        list
            ненайденные слов
        """
        df_single = df.query("word_type == 'single'")
        df_single['vec'] = df_single.apply(lambda x: self.__get_vector(x, 'word'), axis=1)
        df_single = df_single[["word", "vec"]]

        synonyms = self.__get_vec_for_synonym(df)
        composite = self.__get_vec_for_composite(df)

        res = pd.concat([df_single, synonyms, composite], ignore_index=True)

        res_not_na = res[~res["vec"].isna()]
        res_na = res[res["vec"].isna()]["word"].to_list()

        return res_not_na, res_na

    def __dataframe_to_json(self, df: pd.DataFrame) -> None:
        """
        Функция для сохранения датафрейма в json.
        Ключ - слово, значение - векторное представление

        Parameters
        ----------
        df: pd.DataFrame
            датафрейм с словами и векторными представлениями

        Returns
        -------
        None
        """
        df['vec'] = df['vec'].apply(list)
        dict_vec = df.set_index('word')['vec'].to_dict()
        for key, val in dict_vec.items():
            new_val = list(map(str, val))
            dict_vec[key] = new_val
        with open(self.output_filename, 'w', encoding='utf-8') as json_file:
            dump(dict_vec, json_file, ensure_ascii=False, indent=2)

    def create_vected_words(self) -> None:
        """
        Функция для запуска всего пайплайна обработки слов:
        Загрузка сырых слов
        -> Очистка и проставление словам типов
        -> Добавление дополнительных колонок для составных слов и синонимов
        -> Добавление векторных представлений
        -> Сохранение результата в json

        Returns
        -------
        None
        """
        df_raw = self.__load_data()
        df_typed = self.__words_typing(df_raw)
        df_with_extra_cols = self.__add_synonyms_composit_columns(df_typed)
        df_vec, lst_na = self.__get_all_vec(df_with_extra_cols)

        # запись ненайденных слов в атрибут класса
        self.not_found_words = lst_na

        self.__dataframe_to_json(df_vec)

