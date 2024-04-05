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


def load_hubert_model():
    model_name = 'rinna/japanese-hubert-base'
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    hubert_model = AutoModel.from_pretrained(model_name)
    hubert_model.eval()

    return feature_extractor, hubert_model


def extract_features(file_path, feature_extractor, hubert_model, working_directory):
    # NOTE: This function has side effects to save processed audio file
    basename = os.path.basename(file_path)
    preprocessed_dir = os.path.join(working_directory, 'preprocessed')
    preprocessed_file_path = os.path.join(preprocessed_dir, basename + '.npy')

    if os.path.exists(preprocessed_file_path):
        return np.load(preprocessed_file_path)

    os.makedirs(preprocessed_dir, exist_ok=True)

    SAMPLING_RATE = 16000

    waveform, sample_rate = torchaudio.load(file_path)
    if sample_rate != SAMPLING_RATE:
        waveform = torchaudio.transforms.Resample(
            sample_rate, SAMPLING_RATE)(waveform)

    input_values = feature_extractor(
        waveform[0], return_tensors='pt', sampling_rate=SAMPLING_RATE)

    with torch.no_grad():
        output = hubert_model(**input_values)
    retval = output.last_hidden_state.mean(1).detach().numpy()[0]

    np.save(preprocessed_file_path, retval)

    return retval


def create_training_data(folder, working_directory):
    folders = glob.glob(folder + '/*')
    feature_extractor, hubert_model = load_hubert_model()

    class_names = [os.path.basename(folder) for folder in folders]
    x_train = []
    y_train = []

    for folder, class_name in zip(folders, class_names):
        print(f'Processing {class_name}...')
        for file in glob.glob(folder + '/*.wav'):
            print(file)
            features = extract_features(
                file, feature_extractor, hubert_model, working_directory)

            x_train.append(features)
            y_train.append(class_name)

    return np.array(x_train), np.array(y_train)


def predict_single(model, feature_extractor, file_path, hubert_model, working_directory):
    features = extract_features(
        file_path, feature_extractor, hubert_model, working_directory)
    predicted_class = model.predict(features.reshape(1, -1))[0]
    return predicted_class


def save_model(model, output_model_path):
    with open(output_model_path, 'wb') as f:
        pickle.dump(model, f)


def load_model(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model


class Command:
    def learn_model(self, dataset_folder, working_directory):
        os.makedirs(working_directory, exist_ok=True)

        output_model_path = os.path.join(working_directory, 'model.pkl')
        x_train, y_train = create_training_data(
            dataset_folder, working_directory)

        model = SVC(kernel='linear', C=1)
        model.fit(x_train, y_train)

        # show accuracy
        print(model.score(x_train, y_train))

        save_model(model, output_model_path)

    def predict_single(self, file_path, working_directory):
        output_model_path = os.path.join(working_directory, 'model.pkl')

        model = load_model(output_model_path)
        feature_extractor, hubert_model = load_hubert_model()

        predicted_class = predict_single(
            model, feature_extractor, file_path, hubert_model, working_directory)
        print(f'{file_path} is {predicted_class}')

    def predict_folder(self, folder, working_directory):
        output_model_path = os.path.join(working_directory, 'model.pkl')

        model = load_model(output_model_path)
        feature_extractor, hubert_model = load_hubert_model()

        pair = {}
        for file in glob.glob(folder + '/*.wav'):
            predicted_class = predict_single(
                model, feature_extractor, file, hubert_model, working_directory)
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
