pip install librosa
pip install pandas
pip install timm==0.9.12


python train_BEAET2AASIST.py --root_path '/Data/data/sanghyeok/IJCAI_OFFICIAL' --pre_trained_path '/Data/data/sanghyeok/IJCAI_OFFICIAL/networks/BEATs_iter3.pt' --track 1 --feature_split 'Frequency' --multi_layer_fusion 'CNN-gate' --top_k 4