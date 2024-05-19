from model_loader import ModelLoader
from data_preparator import DataPreparator
from similar_word_searcher import SimilarWordSearcher
import warnings

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    # если модель не скачана, то загрузка будет около 15 мин ~2.5 ГБ
    # можно скачать заранее с http://vectors.nlpl.eu/repository/#
    # распаковать в ./models/65
    model = ModelLoader(model_id="65").load_model()

    # подготовка словаря глоссов
    preparator_gloss = DataPreparator(model=model,
                                      input_filename="input_data/RSL_class_list.txt",
                                      output_filename="vected_data/gloss.json")
    preparator_gloss.create_vected_words()

    # подготовка тестовых слов
    preparator_test = DataPreparator(model=model,
                                     input_filename="input_data/test.txt",
                                     output_filename="vected_data/test.json")
    preparator_test.create_vected_words()

    # поиск похожих слов
    searcher = SimilarWordSearcher(gloss_filename=preparator_gloss.output_filename,
                                   test_filename=preparator_test.output_filename)

    res = searcher.find_similar_words(not_found=preparator_test.not_found_words,
                                      trash_hold=0.5)
    # сохранение результата
    searcher.save_dict_to_json("results/result.json", res)
