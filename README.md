# BEAT2AASIST

This is the official repository for the paper **"BEAT2AASIST: BEATs Feature Splitting with Dual-Branch AASIST for Environmental Sound Deepfake Detection"**, accepted to the *2026 International Joint Conference on Artificial Intelligence and European Conference on Artificial Intelligence (IJCAI-ECAI 2026)*, Bremen, Germany. 

BEAT2AASIST is an enhanced deepfake detection framework developed for the [Environmental Sound Deepfake Detection (ESDD) 2026 Challenge](https://sites.google.com/view/esdd-challenge/esdd-challenges/esdd-1/). It addresses the limitations of the existing BEATs-AASIST baseline and effectively detects various spoofing artifacts under unseen generator and black-box conditions.

The proposed approach achieved competitive performance in the ESDD 2026 Challenge, ranking **3rd in Track 2 (Black-Box)** and **4th in Track 1 (Unseen Generators)**.

---

## Overview

BEAT2AASIST improves detection performance through three main architectural and training strategies:

*   **Dual-Branch AASIST Architecture**: The 1D token sequence extracted from the BEATs encoder is explicitly split along either the frequency or channel dimension and processed in parallel by two independent AASIST branches. This design enables specialized modeling of distinct spoofing characteristics.
*   **Multi-layer Fusion**: Instead of relying solely on the final transformer layer of BEATs, it aggregates information from the top-k layers to capture richer and more hierarchical acoustic cues. Supported fusion mechanisms include Concatenation, CNN-gated, and SE-gated strategies.
*   **Vocoder-based Data Augmentation**: Generalization to unseen audio generation models and robustness in black-box scenarios are improved by applying data augmentation using multiple high-fidelity neural vocoders, including HiFi-GAN, UnivNet, and BigVGAN.

---

## Datasets

The experiments are based on data provided by the **ICASSP 2026 Environmental Sound Deepfake Detection Challenge (ESDD) Challenge**.

The datasets can be accessed from [here](https://sites.google.com/view/esdd-challenge/esdd-challenges/esdd-1/dataset?authuser=0)

After downloading the data from the link above, place the data in your own directory as shown below.

```text
Datasets/
├── development/      
├── dev_track2/      
├── eval_track1/
├── eval_track2/
├── test_track1/
└── test_track2/
```

Run the code below to generate the data list (JSON files) based on your custom directory path.

```
python jsons/get_jsons.py \
  --root_path /path/to/BEAT2AASIST \
  --data_path /path/to/Datasets \
```

Running the code above will generate the following files in the jsons folder.
```
jsons/
├── dev_track1_train.json
├── dev_track1_valid.json
├── dev_track2_train.json
├── dev_track2_valid.json
├── eval_track1.json
├── eval_track2.json
├── test_track1.json
└── test_track2.json
```

---

## Challenge Results

Despite using fewer ensemble components than top-ranked systems, BEAT2AASIST demonstrated top-tier performance on the EnvSDD dataset in the ESDD 2026 Challenge.

| Track | Scenario | Test EER (%) | Rank |
| :--- | :--- | :---: | :---: |
| **Track 1** | Generalization to Unseen Generators | 1.60% | 4th |
| **Track 2** | Black-box, Low-resource | 0.35% | 3rd |

*For detailed experimental results and ablation studies, please refer to the paper.

---

## Repository Structure

```text
BEAT2AASIST/
├── train_BEAT2AASIST.py      # Main training and evaluation script
├── train.sh                  # Example training command
├── networks/
│   ├── BEAT2AASIST.py        # BEAT2AASIST model definition
│   ├── AASIST.py             # AASIST-based back-end network
│   └── beats/                # BEATs encoder implementation
├── utils/
│   ├── data_utils.py         # Dataset loading, feature extraction, and EER evaluation utilities
│   └── eval_metrics.py       # Evaluation metrics
├── jsons/
│   └── get_jsons.py          # Utility script for preparing JSON metadata
└── metadata/                 # ESDD metadata files
```

---

## Training

An example training command is:
```
python train_BEAT2AASIST.py \
  --root_path /path/to/BEAT2AASIST \
  --pre_trained_path /path/to/BEATs_iter3.pt \
  --track 1 \
  --feature_split Frequency \
  --multi_layer_fusion CNN-gate \
  --top_k 4 \
  --vocoder_aug True \
  --device cuda:0
```

Main arguments:

| Argument               | Description                                                                              |
| ---------------------- | ---------------------------------------------------------------------------------------- |
| `--root_path`          | Root directory containing `jsons/`, `metadata/`, and experiment outputs                  |
| `--pre_trained_path`   | Path to the pre-trained BEATs checkpoint. If not provided, BEATs is trained from scratch |
| `--track`              | ESDD challenge track number, either `1` or `2`                                           |
| `--feature_split`      | Feature splitting strategy: `Frequency`, `Channel`, or `None`                            |
| `--multi_layer_fusion` | Layer fusion strategy: `CNN-gate`, `SE-gate`, `Concat`, or `None`                        |
| `--top_k`              | Number of top BEATs transformer layers used for multi-layer fusion                       |
| `--vocoder_aug`        | Whether to use vocoder-generated fake audio augmentation                                 |
| `--time_mask`          | Whether to apply time masking during training                                            |
| `--freq_mask`          | Whether to apply frequency masking during training                                       |
| `--batch_size`         | Training batch size                                                                      |
| `--num_epochs`         | Number of training epochs                                                                |
| `--lr`                 | Learning rate                                                                            |
| `--device`             | Device used for training and inference                                                   |

<!--
The training script expects JSON files under
```
{root_path}/jsons/
```

The expected files are:
```
dev_track{track}_train.json
dev_track{track}_valid.json
eval_track{track}.json
test_track{track}.json
```

Each JSON file should contain a list of items with the following format:
```
[
  {
    "file_path": "/path/to/audio.wav",
    "label": "real"
  },
  {
    "file_path": "/path/to/fake_audio.wav",
    "label": "fake"
  }
]
```
-->

### Output Files
Training results are saved under:
```
{root_path}/exp_track{track}/BEAT2AASIST_Split_{feature_split}_Fusion_{multi_layer_fusion}_Topk_{top_k}/
```

The directory contains:
```
log.txt
ckpts/
├── saved_model.pt
├── val_scores.txt
├── eval_scores.txt
└── test_scores.txt
```
The best checkpoint is selected based on the lowest validation EER during training.


---

## Citation

If you use this code or find it helpful in your research, please cite the corresponding paper:

### IJCAI-ECAI 2026
```
@inproceedings{chung2026beat2aasist,
  title     = {{BEAT2AASIST}: {BEATs} Feature Splitting with Dual-Branch
               {AASIST} for Environmental Sound Deepfake Detection},
  author    = {Chung, Sanghyeok and Oh, Seungsang and Kim, Donggun and
               You, Jeongbin and Kwak, Il-Youp and Kim, Eujin and
               Heo, Gaeun and Lee, Nahyun and Choi, Sunmook and Han, Soyul},
  booktitle = {Proceedings of the Thirty-Fifth International Joint
               Conference on Artificial Intelligence (IJCAI-ECAI 2026)},
  year      = {2026},
  address   = {Bremen, Germany},
  note      = {To appear}
}
```

### ICASSP 2026 Challenge track
```
@INPROCEEDINGS{11461593,
  author={Chung, Sanghyeok and Kim, Eujin and Kim, Donggun and Heo, Gaeun and You, Jeongbin and Lee, Nahyun and Choi, Sunmook and Han, Soyul and Oh, Seungsang and Kwak, Il-Youp},
  booktitle={ICASSP 2026 - 2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={BEAT2AASIST Model with Layer Fusion for ESDD 2026 Challenge}, 
  year={2026},
  volume={},
  number={},
  pages={21763-21765},
  keywords={Deepfakes;Vocoders;Videos;Protocols;HTTP;Communication equipment;Telephone equipment;Fuses;Electronic components;Convolutional neural networks;ESDD;BEATs;Multi-layer fusion;Vocoder},
  doi={10.1109/ICASSP55912.2026.11461593}}
```
