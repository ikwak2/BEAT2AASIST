import os, json, argparse
import pandas as pd
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate data list')
    parser.add_argument('--root_path',   type=str, default='/Data/data/sanghyeok/IJCAI_OFFICIAL', help='Root directory path for the project.')
    parser.add_argument('--data_path',   type=str, default='/Data/data/ESDD2026',                 help='Data directory path.')
    args = parser.parse_args()

    TYPE_list = ['dev_track1', 'dev_track2', 'eval_track1', 'eval_track2', 'test_track1', 'test_track2']

    dev_track1_audio = os.path.join(args.data_path,  'development')
    dev_track2_audio = os.path.join(args.data_path,  'dev_track2')

    eval_track1_audio = os.path.join(args.data_path,  'eval_track1')
    eval_track2_audio = os.path.join(args.data_path,  'eval_track2')

    test_track1_audio = os.path.join(args.data_path,  'test_track1')
    test_track2_audio = os.path.join(args.data_path,  'test_track2')

    for TYPE in TYPE_list:

        if TYPE == "dev_track1":
            train_meta_path = f"{args.root_path}/jsons/dev_track1_train.json"
            valid_meta_path = f"{args.root_path}/jsons/dev_track1_valid.json"
            train_data, valid_data = [], []
            metadata = pd.read_csv(f'{args.root_path}/metadata/dev_track1.csv')
            nums = metadata.shape[0]
            for i in tqdm(range(nums)):
                wavename = metadata.iloc[i,0]
                dataset = metadata.iloc[i,1]
                usage = metadata.iloc[i,-1]
                real_info = {
                    'file_path': f'{dev_track1_audio}/real_audio/{dataset}/{wavename}',
                    'label': 'real',
                    'attack_type': 'real',
                    'generative_model': 'real',
                }
                if usage == 'train':
                    train_data.append(real_info)
                    for model_name in ['audiogen', 'audioldm1', 'audioldm2']:
                        train_data.append({
                            'file_path': f'{dev_track1_audio}/fake_audio/TTA/{model_name}/{dataset}/{wavename}',
                            'label': 'fake',
                            'attack_type': 'tta',
                            'generative_model': model_name,
                        })
                    for model_name in ['audioldm1']:
                        train_data.append({
                            'file_path': f'{dev_track1_audio}/fake_audio/ATA/{model_name}/{dataset}/{wavename}',
                            'label': 'fake',
                            'attack_type': 'ata',
                            'generative_model': model_name,
                        })
                elif usage == 'validation':
                    valid_data.append(real_info)
                    for model_name in ['audiogen', 'audioldm1', 'audioldm2']:
                        valid_data.append({
                            'file_path': f'{dev_track1_audio}/fake_audio/TTA/{model_name}/{dataset}/{wavename}',
                            'label': 'fake',
                            'attack_type': 'tta',
                            'generative_model': model_name,
                        })
                    for model_name in ['audioldm1']:
                        valid_data.append({
                            'file_path': f'{dev_track1_audio}/fake_audio/ATA/{model_name}/{dataset}/{wavename}',
                            'label': 'fake',
                            'attack_type': 'ata',
                            'generative_model': model_name,
                        })
        elif TYPE == "dev_track2":
            train_meta_path = f"{args.root_path}/jsons/dev_track2_train.json"
            valid_meta_path = f"{args.root_path}/jsons/dev_track2_valid.json"
            train_data, valid_data = [], []
            # get data of dev track 1
            metadata = pd.read_csv(f'{args.root_path}/metadata/dev_track1.csv')
            nums = metadata.shape[0]
            for i in tqdm(range(nums)):
                wavename = metadata.iloc[i,0]
                dataset = metadata.iloc[i,1]
                usage = metadata.iloc[i,-1]
                real_info = {
                    'file_path': f'{dev_track1_audio}/real_audio/{dataset}/{wavename}',
                    'label': 'real',
                    'attack_type': 'real',
                    'generative_model': 'real',
                }
                if usage == 'train':
                    train_data.append(real_info)
                    for model_name in ['audiogen', 'audioldm1', 'audioldm2']:
                        train_data.append({
                            'file_path': f'{dev_track1_audio}/fake_audio/TTA/{model_name}/{dataset}/{wavename}',
                            'label': 'fake',
                            'attack_type': 'tta',
                            'generative_model': model_name,
                        })
                    for model_name in ['audioldm1']:
                        train_data.append({
                            'file_path': f'{dev_track1_audio}/fake_audio/ATA/{model_name}/{dataset}/{wavename}',
                            'label': 'fake',
                            'attack_type': 'ata',
                            'generative_model': model_name,
                        })
                elif usage == 'validation':
                    valid_data.append(real_info)
                    for model_name in ['audiogen', 'audioldm1', 'audioldm2']:
                        valid_data.append({
                            'file_path': f'{dev_track1_audio}/fake_audio/TTA/{model_name}/{dataset}/{wavename}',
                            'label': 'fake',
                            'attack_type': 'tta',
                            'generative_model': model_name,
                        })
                    for model_name in ['audioldm1']:
                        valid_data.append({
                            'file_path': f'{dev_track1_audio}/fake_audio/ATA/{model_name}/{dataset}/{wavename}',
                            'label': 'fake',
                            'attack_type': 'ata',
                            'generative_model': model_name,
                        })
            # get data from dev track 2
            metadata = pd.read_csv(f'{args.root_path}/metadata/dev_track2.csv')
            nums = metadata.shape[0]
            for i in tqdm(range(nums)):
                wavename, label, usage = metadata.iloc[i,0], metadata.iloc[i,1], metadata.iloc[i,2]
                if usage == "train":
                    train_data.append({
                        'file_path': f'{dev_track2_audio}/{wavename}',
                        'label': label,
                        'attack_type': 'xxx',
                        'generative_model': 'xxx',
                    })
                elif usage == "validation":
                    valid_data.append({
                        'file_path': f'{dev_track2_audio}/{wavename}',
                        'label': label,
                        'attack_type': 'xxx',
                        'generative_model': 'xxx',
                    })
                    
        elif TYPE == "eval_track1":
            eval_meta_path = f"{args.root_path}/jsons/eval_track1.json"
            eval_data = []
            metadata = pd.read_csv(f'{args.root_path}/metadata/eval_track1.csv')
            nums = metadata.shape[0]
            for i in tqdm(range(nums)):
                wavename, label = metadata.iloc[i,0], metadata.iloc[i,1]
                eval_data.append({
                        'file_path': f'{eval_track1_audio}/{wavename}',
                        'label': label,
                        'attack_type': 'xxx',
                        'generative_model': 'xxx',
                    })
        elif TYPE == "eval_track2":
            eval_meta_path = f"{args.root_path}/jsons/eval_track2.json"
            eval_data = []
            metadata = pd.read_csv(f'{args.root_path}/metadata/eval_track2.csv')
            nums = metadata.shape[0]
            for i in tqdm(range(nums)):
                wavename, label = metadata.iloc[i,0], metadata.iloc[i,1]
                eval_data.append({
                        'file_path': f'{eval_track2_audio}/{wavename}',
                        'label': label,
                        'attack_type': 'xxx',
                        'generative_model': 'xxx',
                    })
        elif TYPE == "test_track1":
            test_meta_path = f"{args.root_path}/jsons/test_track1.json"
            test_data = []
            metadata = pd.read_csv(f'{args.root_path}/metadata/test_track1.csv')
            nums = metadata.shape[0]
            for i in tqdm(range(nums)):
                wavename, source_wavename, source_dataset, faketype, generator = \
                metadata.iloc[i,0],metadata.iloc[i,1],metadata.iloc[i,2],metadata.iloc[i,3],metadata.iloc[i,4]
                if faketype == "real":
                    label = "real"
                else:
                    label = "fake"
                test_data.append({
                        'file_path': f'{test_track1_audio}/{wavename}',
                        'label': label,
                        'attack_type': faketype,
                        'generative_model': generator,
                    })
        elif TYPE == "test_track2":
            test_meta_path = f"{args.root_path}/jsons/test_track2.json"
            test_data = []
            metadata = pd.read_csv(f'{args.root_path}/metadata/test_track2.csv')
            nums = metadata.shape[0]
            for i in tqdm(range(nums)):
                wavename, source_wavename, source_dataset, faketype, generator = \
                metadata.iloc[i,0],metadata.iloc[i,1],metadata.iloc[i,2],metadata.iloc[i,3],metadata.iloc[i,4]
                if faketype == "real":
                    label = "real"
                else:
                    label = "fake"
                test_data.append({
                        'file_path': f'{test_track2_audio}/{wavename}',
                        'label': label,
                        'attack_type': faketype,
                        'generative_model': generator,
                    })


        if "test" in TYPE:
            # show information
            print(f"Type: {TYPE}")
            print(f'Test Samples: {len(test_data)}, saved in {test_meta_path}')
            # # Save metadata to JSON files
            with open(test_meta_path, "w") as f:
                json.dump(test_data, f, indent=4)
        elif "eval" in TYPE:
            # show information
            print(f"Type: {TYPE}")
            print(f'Eval Samples: {len(eval_data)}, saved in {eval_meta_path}')
            # # Save metadata to JSON files
            with open(eval_meta_path, "w") as f:
                json.dump(eval_data, f, indent=4)
        else:
            # show information
            print(f"Type: {TYPE}")
            print(f'Train Samples: {len(train_data)}, saved in {train_meta_path}')
            print(f'Validation Samples: {len(valid_data)}, saved in {valid_meta_path}')
            # # Save metadata to JSON files
            with open(train_meta_path, "w") as f:
                json.dump(train_data, f, indent=4)
            with open(valid_meta_path, "w") as f:
                json.dump(valid_data, f, indent=4)