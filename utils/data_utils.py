import os
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
import torchaudio.compliance.kaldi as ta_kaldi
import librosa
import json
import pandas
import utils.eval_metrics as em


class ESDD_Dataset(Dataset):
    def __init__(self, args, train, path):
        if train and args.vocoder_aug=='True':
            labels, list_IDs = genSpoof_list_VOCODER_AUG(path)
        else:
            labels, list_IDs = genSpoof_list(path)

        self.list_IDs = list_IDs
        self.labels = labels
        self.args = args
        self.cut = 64600  # ~4 seconds of audio (64600 samples)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        filepath = self.list_IDs[index]
        X, sr = librosa.load(filepath, sr=16000)
        X_pad = pad(X, self.cut)
        x_inp = Tensor(X_pad)  
        fbank = preprocess(x_inp)
        target = self.labels[filepath]

        return fbank, target

def genSpoof_list(dir_meta):
    d_meta = {}    # key : file_path, value : label(0 or 1)
    file_list = [] # list of file_path

    # Read the JSON file
    with open(dir_meta, 'r') as f:
        data = json.load(f)
    
    for item in data:
        key = item['file_path']
        label = item['label']
        file_list.append(key)
        d_meta[key] = 1 if label == 'real' else 0

    return d_meta, file_list

def genSpoof_list_VOCODER_AUG(dir_meta):
    d_meta = {}    # key : file_path, value : label(0 or 1)
    file_list = [] # list of file_path

    # Read the JSON file
    with open(dir_meta, 'r') as f:
        data = json.load(f)

    for item in data:
        key = item['file_path']
        label = item['label']
        file_list.append(key)
        d_meta[key] = 1 if label == 'real' else 0

    ### additional generated fake audio ###
    gen_list = []
    for file in file_list:
        if d_meta[file]==1:
            # hifigan generated fake data
            gen_audio_1 = '/Data/data/ESDD2026/generated/hifigan/' + '/'.join(file.split('/')[-2:])
            if os.path.exists(gen_audio_1):
                gen_list.append(gen_audio_1)

            # bigvgan_v2 generated fake data
            gen_audio_2 = '/Data/data/ESDD2026/generated/bigvgan_v2/' + '/'.join(file.split('/')[-2:])
            if os.path.exists(gen_audio_2):
                gen_list.append(gen_audio_2)

            # univnet generated fake data
            gen_audio_3 = '/Data/data/ESDD2026/generated/univnet/' + '/'.join(file.split('/')[-2:])
            if os.path.exists(gen_audio_3):
                gen_list.append(gen_audio_3)

    print(f'Generated Fake audio : {len(gen_list)}')
    for gen in gen_list:
        file_list.append(gen)
        d_meta[gen] = 0   ### label : fake 

    return d_meta, file_list


def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len)+1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x	

def preprocess(
        waveform: torch.Tensor,
        fbank_mean: float = 15.41663,
        fbank_std: float = 6.55582,
    ) -> torch.Tensor:
    
        waveform = waveform.unsqueeze(0) * 2**15
        fbank = ta_kaldi.fbank(
            waveform,
            num_mel_bins=128,
            sample_frequency=16000,
            frame_length=1615 / 64.6,
            frame_shift=646 / 64.6,
        )

        fbank = (fbank - fbank_mean) / (2 * fbank_std)
        return fbank

def eval_to_score_file(score_file, key_json_file):
    """
    Evaluate scores and calculate EER based on ground truth JSON file.
    """
    # Load ground truth from JSON file
    with open(key_json_file, 'r') as f:
        cm_data = json.load(f)

    # Load the submission scores
    submission_scores = pandas.read_csv(score_file, sep='|', header=None, skipinitialspace=True)

    if len(submission_scores.columns) != 2:
        print("Error: Submission file must have exactly 2 columns (file_name and score).")
        exit(1)

    # Convert cm_data JSON to a DataFrame
    cm_df = pandas.DataFrame(cm_data)

    #print(submission_scores.columns)
    #print(cm_df.columns)
    # Merge scores with ground truth
    cm_scores = pandas.concat([submission_scores,cm_df],axis=1, join='inner')
    #cm_scores = submission_scores.merge(cm_df, left_on=0, right_on="file_path", how="inner")

    # Extract bona-fide and spoof scores
    bona_cm = cm_scores[cm_scores["label"] == "real"][1].values
    spoof_cm = cm_scores[cm_scores["label"] == "fake"][1].values

    # Compute EER
    eer_cm, threshold = em.compute_eer(bona_cm, spoof_cm)
    #print(f"EER: {eer_cm * 100:.2f}%")

    return eer_cm, threshold