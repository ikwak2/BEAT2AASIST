# BEAT2AASIST: BEATs Feature Splitting with Dual-Branch AASIST

This is the official repository for the paper **"BEAT2AASIST: BEATS Feature Splitting with Dual-Branch AASIST for Environmental Sound Deepfake Detection"**. 

BEAT2AASIST is an enhanced deepfake detection framework developed for the [Environmental Sound Deepfake Detection (ESDD) 2026 Challenge](https://sites.google.com/view/esdd-challenge/esdd-challenges/esdd-1/). It addresses the limitations of the existing BEATs-AASIST baseline and effectively detects various spoofing artifacts under unseen generator and black-box conditions.

The proposed approach achieved competitive performance in the ESDD 2026 Challenge, ranking **3rd in Track 2 (Black-Box)** and **4th in Track 1 (Unseen Generators)**.

---

## Overview

BEAT2AASIST maximizes detection performance through three main architectural and training strategies:

*   **Dual-Branch AASIST Architecture**: The 1D token sequence extracted from the BEATs encoder is explicitly split along either the frequency or channel dimension and processed in parallel by two independent AASIST branches. This design enables specialized modeling of distinct spoofing characteristics.
*   **Multi-layer Fusion**: Instead of relying solely on the final transformer layer of BEATs, it aggregates information from the top-k layers to capture richer and more hierarchical acoustic cues. Supported fusion mechanisms include Concatenation, CNN-gated, and SE-gated strategies.
*   **Vocoder-based Data Augmentation**: Generalization to unseen audio generation models and robustness in black-box scenarios are improved by applying data augmentation using multiple high-fidelity neural vocoders, including HiFi-GAN, UnivNet, and BigVGAN.

---

## Datasets

The experiments are based on data provided by the **ICASSP 2026 Environmental Sound Deepfake Detection Challenge (ESDD) Challenge**,

The datasets can be accessed from [here](https://sites.google.com/view/esdd-challenge/esdd-challenges/esdd-1/dataset?authuser=0)

---

## Challenge Results

Despite using fewer ensemble components than top-ranked systems, BEAT2AASIST demonstrated top-tier performance on the EnvSDD dataset in the ESDD 2026 Challenge.

| Track | Scenario | Test EER (%) | Rank |
| :--- | :--- | :---: | :---: |
| **Track 1** | Generalization to Unseen Generators | 1.60% | 4th |
| **Track 2** | Black-box, Low-resource | 0.35% | 3rd |

*For detailed experimental results and ablation studies, please refer to the paper.

---

## Code 관련 설명: 
상혁학생 코드 업데이트 후 여기부분 작성 부탁합니다. :)


---

## Citation

If you use this code or find it helpful in your research, please cite the corresponding paper:

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
