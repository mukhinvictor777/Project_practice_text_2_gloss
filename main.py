from model_loader import ModelLoader
from data_preparator import DataPreparator
from similar_word_searcher import SimilarWordSearcher
import warnings

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    model = ModelLoader(model_id="65").load_model()

    preparator_gloss = DataPreparator(model=model,
                                      input_filename="RSL_class_list.txt",
                                      output_filename="gloss.json")
    preparator_gloss.create_vected_words()

    preparator_test = DataPreparator(model=model,
                                     input_filename="test.txt",
                                     output_filename="test.json")
    preparator_test.create_vected_words()

    searcher = SimilarWordSearcher(gloss_filename=preparator_gloss.output_filename,
                                   test_filename=preparator_test.output_filename)

    res = searcher.find_similar_words(not_found=preparator_test.not_found_words,
                                      trash_hold=0.5)
    searcher.save_dict_to_json("result.json", res)
