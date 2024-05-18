import pymorphy2
from nltk.stem.snowball import SnowballStemmer
import inspect


class WordAnalyzer:
    def __init__(self):
        """
        Инициализация объекта класса WordAnalyzer.
        """
        self.morph = self._init_morph()
        self.stemmer = SnowballStemmer("russian")

    def _init_morph(self):
        """
        Инициализация объекта MorphAnalyzer с передачей языка из параметров конструктора.
        """
        init_params = inspect.signature(pymorphy2.MorphAnalyzer.__init__).parameters
        lang_param = init_params.get('lang')
        lang = 'ru' if lang_param.default == inspect.Parameter.empty else lang_param.default
        return pymorphy2.MorphAnalyzer(lang=lang)

    def extract_normal_form(self, word: str) -> str:
        """
        Извлекает нормальную форму слова.

        Параметры:
            word (str): Слово, для которого нужно найти нормальную форму.

        Возвращает:
            str: Нормальная форма слова.
        """
        parsed_word = self.morph.parse(word)[0]
        return parsed_word.normal_form

    def _get_param_names(self):
        signature = inspect.signature(self.__init__)
        return [p.name for p in signature.parameters.values()]

    def stemming(self, word: str) -> str:
        """
        Выполняет стемминг слова.

        Параметры:
            word (str): Слово, которое нужно стеммить.

        Возвращает:
            str: Основа слова после стемминга.
        """
        return self.stemmer.stem(word)

    def levenshtein_distance(self, word1: str, word2: str) -> int:
        """
        Вычисляет расстояние Левенштейна между двумя словами.

        Параметры:
            word1 (str): Первое слово.
            word2 (str): Второе слово.

        Возвращает:
            int: Расстояние Левенштейна между словами.
        """
        len1, len2 = len(word1), len(word2)
        dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]
        for i in range(len1 + 1):
            dp[i][0] = i
        for j in range(len2 + 1):
            dp[0][j] = j
        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                cost = 0 if word1[i - 1] == word2[j - 1] else 1
                dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
        return dp[len1][len2]

    def compare_words(self, word1: str, word2: str, threshold_percent: float = 0.5) -> bool:
        """
        Сравнивает два слова на основе их нормальных форм, стеммов и расстояния Левенштейна.

        Параметры:
            word1 (str): Первое слово.
            word2 (str): Второе слово.
            threshold_percent (float): Процент от длины слова для определения порогового значения.
                                      По умолчанию 0.7.

        Возвращает:
            bool: True, если слова считаются однокоренными, False в противном случае.
        """
        normal_form1 = self.extract_normal_form(word1)
        normal_form2 = self.extract_normal_form(word2)
        stem1 = self.stemming(normal_form1)
        stem2 = self.stemming(normal_form2)
        distance_threshold = max(len(stem1), len(stem2)) * threshold_percent
        distance = self.levenshtein_distance(stem1, stem2)
        return distance <= distance_threshold

    def find_related_words(self, word: str, dictionary: list[str]) -> list[str]:
        """
        Находит однокоренные слова к поданному слову из заданного словаря.

        Параметры:
            word (str): Слово, для которого нужно найти однокоренные слова.
            dictionary (List[str]): Список слов из словаря.

        Возвращает:
            List[str]: Список однокоренных слов к поданному слову из словаря.
        """
        related_words = [dict_word for dict_word in dictionary if self.compare_words(word, dict_word)]
        return related_words
