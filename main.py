import fire
import torch
import torchaudio
import json
import glob
import numpy as np
import os
from sklearn.svm import SVC
import pickle
from transformers import AutoFeatureExtractor, AutoModel
import hashlib
import tqdm


def sha256hash(file_path):
    with open(file_path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


class SpeakerIdentification:
    def __init__(self,
                 working_directory,
                 model_name='rinna/japanese-hubert-base',
                 sampling_rate=16000):
        self.working_directory = working_directory
        self.feature_directory = os.path.join(
            working_directory, 'features')
        self.model_path = os.path.join(working_directory, 'model.pkl')

        os.makedirs(self.feature_directory, exist_ok=True)

        self.load_hubert_model(model_name)
        self.sampling_rate = sampling_rate
        self.model = None

    def load_hubert_model(self, model_name):
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            model_name)
        self.hubert_model = AutoModel.from_pretrained(model_name)
        self.hubert_model.eval()

    def extract_features(self, file_path):
        # NOTE: This function has side effects to save processed audio file
        feature_filename = sha256hash(file_path) + '.npy'
        preprocessed_file_path = os.path.join(
            self.feature_directory, feature_filename)

        # NOTE: only use mean of hidden states
        def preprocess(x): return x.mean(axis=0).reshape(-1)

        if os.path.exists(preprocessed_file_path):
            return preprocess(np.load(preprocessed_file_path))

        waveform, sample_rate = torchaudio.load(file_path)
        if sample_rate != self.sampling_rate:
            waveform = torchaudio.transforms.Resample(
                sample_rate, self.sampling_rate)(waveform)

        input_values = self.feature_extractor(
            waveform[0], return_tensors='pt',
            sampling_rate=self.sampling_rate)

        with torch.no_grad():
            output = self.hubert_model(**input_values)
        # retval = output.last_hidden_state.mean(1).detach().numpy()[0]
        retval = output.last_hidden_state.detach().numpy()[0]

        np.save(preprocessed_file_path, retval)

        return preprocess(retval)

    def create_training_data(self, dataset_folder):
        folders = glob.glob(dataset_folder + '/*')

        class_names = [os.path.basename(folder) for folder in folders]
        x_train = []
        y_train = []

        for dataset_folder, class_name in zip(folders, class_names):
            print(f'Processing {class_name}...')
            files = glob.glob(dataset_folder + '/*.wav')
            for file in tqdm.tqdm(files):
                features = self.extract_features(file)

                x_train.append(features)
                y_train.append(class_name)

        return np.array(x_train), np.array(y_train)

    def predict_single(self, file_path):
        if self.model is None:
            raise ValueError('Model is not loaded')

        features = self.extract_features(file_path)
        predicted_class = self.model.predict(features.reshape(1, -1))[0]
        return predicted_class

    def learn_model(self, dataset_folder):
        x_train, y_train = self.create_training_data(dataset_folder)
        os.makedirs(self.working_directory, exist_ok=True)

        self.model = SVC(kernel='linear', C=1)
        self.model.fit(x_train, y_train)

        # show accuracy
        print(self.model.score(x_train, y_train))

        self.save_model()

    def save_model(self):
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.model, f)

    def load_model(self, model_path):
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)


class Command:
    def learn_model(self, dataset_folder, working_directory):
        manager = SpeakerIdentification(working_directory)
        manager.learn_model(dataset_folder)
        manager.save_model()

    def predict_single(self, file_path, working_directory):
        manager = SpeakerIdentification(working_directory)
        manager.load_model(manager.model_path)
        predicted_class = manager.predict_single(file_path)
        print(f'{file_path} is {predicted_class}')

    def predict_folder(self, folder, working_directory):
        manager = SpeakerIdentification(working_directory)
        manager.load_model(manager.model_path)

        pair = {}
        for file in glob.glob(folder + '/*.wav'):
            predicted_class = manager.predict_single(file)
            print(f'{file} is {predicted_class}')

            pair[file] = predicted_class

        basename = os.path.basename(os.path.dirname(folder))
        result_path = os.path.join(
            working_directory, f'result-{basename}.json')

        with open(result_path, 'w') as f:
            json.dump(pair, f)


def main():
    fire.Fire(Command)


if __name__ == '__main__':
    main()
