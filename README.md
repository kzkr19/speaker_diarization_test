# Speaker Diarization Test

## Abstract
Snippet for speaker diarization with HuBERT model(rinna/japanese-hubert-base).
This repository is made for my studying, you should use pyannote-audio.

## Usage

Setup: Clone repsitory and install depended libraries. Specify torch with cuda if you need.
```bash
git clone git@github.com:kzkr19/speaker_identification_test.git
cd speaker_identification_test.
pip install fire torch torchaudio json transformers scikit-learn
```

Prepare Dataset: Put `.wav` files in separate folders for each speaker like below.
```text
+ dataset_directory
├── speaker009
│   ├── 000_000.wav
│   ├── 000_001.wav
│   ├── :
├── speaker001
│   ├── 001_000.wav
│   ├── 001_001.wav
│   ├── :
├── speaker002
│   ├── 002_000.wav
│   ├── 002_001.wav
│   ├── :
├── :
```

Run(Learning): Learn speaker identification model with SVM. Learned model will be saved `{working_directory}/model.pkl`. Features of `.wav` files are saved in `{working_directory}/preprocessed`.
```bash
python main.py learn_model ./dataset_folder ./working_directory
```

Run(Prediction): Predict speaker with learned model.
```bash
python main.py predict_single ./xxx.wav ./working_directory
```

Predict files in specified folder.
```bash
python main.py predict_folder ./target_folder ./working_directory
```

## Reference
* [rinna/japanese-hubert-base · Hugging Face](https://huggingface.co/rinna/japanese-hubert-base)
* [GitHub - pyannote/pyannote-audio: Neural building blocks for speaker diarization](https://github.com/pyannote/pyannote-audio?tab=readme-ov-file#pyannoteaudio-speaker-diarization-toolkit)