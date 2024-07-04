import os
import gensim
import wget
import zipfile


class ModelLoader:
    """Класс для скачивания и загрузки модели"""
    def __init__(self, model_id="65"):
        """
        Функция инициализации класса

        Parameters
        ----------
        model_id: str
            id модели
        """
        self.model_id = model_id

    def __download_model_rusvectores(self):
        """Функция скачивает zip с моделью с сайта rusvectores"""
        # создаем папку для модели если ее нет
        if not os.path.exists("models"):
            os.mkdir("models")
        if os.path.exists(f"models/{self.model_id}"):
            print("Model is downloaded already")
        else:
            model_url = f'http://vectors.nlpl.eu/repository/20/{self.model_id}.zip'
            print("Start to download model")
            m = wget.download(model_url)
            with zipfile.ZipFile(f'{self.model_id}.zip', 'r') as zip_ref:
                print(f"Start to unzip model into models/{self.model_id}")
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
        print("Model has been loaded")

        # Удаление лишнего zip архива
        zip_file_path = f'{self.model_id}.zip'
        if os.path.exists(zip_file_path):
            os.remove(zip_file_path)
            print(f"Removed zip file: {zip_file_path}")

        return model
