import os
import gensim
import wget
import zipfile


class ModelLoader:
    def __init__(self, model_id="65"):
        """Функция инициализации класса
        :param model_id: str
            id модели
        """
        self.model_id = model_id

    def __download_model_rusvectores(self):
        """Функция скачивает zip с моделью с сайта rusvectores"""
        if not os.path.exists("models"):
            os.mkdir("models")
        if os.path.exists(f"models/{self.model_id}"):
            print("Model is downloaded already")
        else:
            model_url = f'http://vectors.nlpl.eu/repository/11/{self.model_id}.zip'
            m = wget.download(model_url)
            with zipfile.ZipFile(f'{self.model_id}.zip', 'r') as zip_ref:
                unzip_path = "models/" + str(self.model_id)
                zip_ref.extractall(unzip_path)

    def load_model(self):
        """
        Функция загружает модель.
        Работает только с word2vec моделями в формате model.bin
        """
        self.__download_model_rusvectores()

        model_path = f"models/{self.model_id}/model.bin"
        model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=True)
        return model
