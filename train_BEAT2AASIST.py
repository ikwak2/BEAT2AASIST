import os, time, random, argparse
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from networks.BEAT2AASIST import BEAT2AASIST as BEAT2AASIST
from utils.data_utils import ESDD_Dataset, eval_to_score_file
from utils.eval_metrics import accuracy


def Train_Epoch(train_loader, model, args, optimizer, epoch):
    train_loss = 0
    train_acc = 0
    num_total = 0.0

    weight = torch.FloatTensor([args.fake, args.real]).to(args.device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    
    model.BEATs.model.train_mode = True
    model.train()
    model.to(args.device)
    for batch_x, batch_y in tqdm(train_loader, total=len(train_loader), desc='Training'): 
        batch_size = batch_x.size(0)
        num_total += batch_size
        
        batch_x = batch_x.to(args.device)
        batch_y = batch_y.view(-1).type(torch.int64).to(args.device)

        batch_out = model(batch_x)
        batch_loss = criterion(batch_out, batch_y)
        
        train_loss += (batch_loss.item() * batch_size)
        train_acc += accuracy(batch_out, batch_y) * batch_size

        optimizer.learning_rate = args.lr*(args.lr_decay**epoch)
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
       
    train_loss /= num_total
    train_acc /= num_total
    
    return train_loss, train_acc

def Evaluate(data_loader, model, args, output, path):
    val_loss = 0.0
    val_acc = 0.0
    num_total = 0.0

    weight = torch.FloatTensor([args.fake, args.real]).to(args.device)
    criterion = nn.CrossEntropyLoss(weight=weight)

    model.BEATs.model.train_mode = False
    model.eval()
    with open(output, 'w') as fh:  # Open file once in write mode
        with torch.no_grad():
            for batch_x, batch_y in data_loader:        
                batch_size = batch_x.size(0)
                num_total += batch_size

                batch_x = batch_x.to(args.device)
                _batch_y = batch_y.view(-1).type(torch.int64).to(args.device)
                
                batch_out = model(batch_x)
                batch_loss = criterion(batch_out, _batch_y)

                val_loss += (batch_loss.item() * batch_size)
                val_acc += accuracy(batch_out, batch_y) * batch_size

                batch_score = batch_out[:, 1].data.cpu().numpy().ravel()  # Extract spoof confidence scores
                for f, cm in zip(batch_y, batch_score):
                    fh.write(f'{f}|{cm}\n')  # Save file name and score
        
    val_loss /= num_total
    val_acc  /= num_total

    eer, threshold = eval_to_score_file(output, path )

    return val_loss, val_acc, eer, threshold


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BEAT2AASIST')

    ### Model Hyperparameters
    #parser.add_argument('--pre_trained_path',   type=str, default='/Data/data/sanghyeok/IJCAI/networks/BEATs_iter3.pt')
    parser.add_argument('--pre_trained_path',   type=str, default=None,        help='Path to the pre-trained BEATs model parameters.')
    parser.add_argument('--feature_split',      type=str, default='Frequency', help='Feature Split method      : Frequency/Channel/None.')
    parser.add_argument('--multi_layer_fusion', type=str, default='CNN-gate',  help='Multi-layer Fusion method : CNN-gate/SE-gate/Concat/None.')
    parser.add_argument('--top_k',              type=int, default=4,           help='Top-k layers for multi-layer fusion method : 1 ~ 12.') ####### 
    parser.add_argument('--vocoder_aug',        type=str, default='True',     help='Vocoder Augmentation if you have vocoder generated fake audio : True/False.') ####### 
    
    ### Basic Hyperparameters
    parser.add_argument('--device',      type=str,   default='cuda:0', help='Device to use for model training and inference.')
    parser.add_argument('--num_workers', type=int,   default=2,        help='Number of worker processes for parallel data loading.')
    parser.add_argument('--seed',        type=int,   default=2025)

    parser.add_argument('--num_epochs',  type=int,   default=1,        help='Number of train epochs.')
    parser.add_argument('--batch_size',  type=int,   default=32,       help='Batch size.')
    parser.add_argument('--lr',          type=float, default=0.000001, help='Learning rate.')
    parser.add_argument('--lr_decay',    type=float, default=0.9,      help='Learning rate decay (linear).')

    parser.add_argument('--fake',        type=float, default=0.1,      help='Class weight for the fake label in CrossEntropyLoss.')
    parser.add_argument('--real',        type=float, default=0.9,      help='Class weight for the real label in CrossEntropyLoss.')
    
    # Masking Augmentation
    parser.add_argument('--time_mask',      type=str,   default='True', help='Time masking augmentation : True/False.')
    parser.add_argument('--freq_mask',      type=str,   default='True', help='Frequency masking augmentation : True/False.')
    parser.add_argument('--max_time_mask',  type=int,   default=30,     help='Maximum width of the time mask.')
    parser.add_argument('--max_freq_mask',  type=int,   default=40,     help='Maximum width of the frequency mask.')
    parser.add_argument('--max_mask_ratio', type=float, default=0.5,    help='Maximum proportion of time steps to mask.')
    parser.add_argument('--mask_iid',       type=str,   default='True', help='Apply independent masks across channels : True/False.')

    ### Auxiliary arguments
    parser.add_argument('--track',       type=int, default=1, help='ESDD1 Challenge Track : 1/2.')
    parser.add_argument('--root_path',   type=str, default='/Data/data/sanghyeok/IJCAI_OFFICIAL', help='Root directory path for the project.')
    
    args = parser.parse_args()

    ### Make experiment reproducible
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    ### Result saving folder & path
    result_folder = f'{args.root_path}/exp_track{args.track}/BEAT2AASIST_Split_{args.feature_split}_Fusion_{args.multi_layer_fusion}_Topk_{args.top_k}'  # 결과 저장폴더
    model_save_path = f'{result_folder}/ckpts'
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path, exist_ok=True)
    val_output = os.path.join(model_save_path,  'val_scores.txt')
    eval_output = os.path.join(model_save_path, 'eval_scores.txt')
    test_output = os.path.join(model_save_path, 'test_scores.txt')
    
  
    ### Model
    model = BEAT2AASIST(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0001)
    
    ### Train dataloader
    train_path   = f'{args.root_path}/jsons/dev_track{args.track}_train.json'
    train_set    = ESDD_Dataset(args=args, train=True ,path= train_path)
    train_loader = DataLoader(train_set, batch_size = args.batch_size, num_workers = args.num_workers, shuffle = True, drop_last = True, pin_memory=True)
    
    ### Dev(validation) dataloader
    dev_path   = f'{args.root_path}/jsons/dev_track{args.track}_valid.json'
    dev_set    = ESDD_Dataset(args=args, train=False, path= dev_path)
    dev_loader = DataLoader(dev_set, batch_size=args.batch_size, num_workers= args.num_workers, shuffle=False)
    
    ### Evaluation dataloader
    eval_path   = f'{args.root_path}/jsons/eval_track{args.track}.json'
    eval_set    = ESDD_Dataset(args=args, train=False, path= eval_path)
    eval_loader = DataLoader(eval_set, batch_size=args.batch_size, num_workers= args.num_workers, shuffle=False)

    ### Test dataloader
    test_path   = f'{args.root_path}/jsons/test_track{args.track}.json'
    test_set    = ESDD_Dataset(args=args, train=False, path= test_path)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers= args.num_workers, shuffle=False)


    ### Result saving file
    score_file = open(f'{result_folder}/log.txt', "a+")

    score_file.write(f"### Model Information ### \n")
    score_file.write(f"Model                     : BEAT2AASIST \n")
    score_file.write(f"Pre-trained BEATs         : {'YES' if args.pre_trained_path else 'NO'} \n")
    score_file.write(f"Feature Split method      : {args.feature_split} \n")
    score_file.write(f"Multi-layer Fusion method : {args.multi_layer_fusion} \n")
    score_file.write(f"Top-k layers              : {args.top_k} \n")
    score_file.write(f"Vocoder Augmentation      : {args.vocoder_aug} \n")
    score_file.write(f"num_params(learnable)     : {sum([param.view(-1).size()[0] for param in model.parameters()])} ({sum(p.numel() for p in model.parameters() if p.requires_grad)}) \n\n")

    score_file.write(f"### Data Information ### \n")
    score_file.write(f"Training data   : {len(train_set.list_IDs)} \n")
    score_file.write(f"Validation data : {len(dev_set.list_IDs)} \n")
    score_file.write(f"Evaluation data : {len(eval_set.list_IDs)} \n")
    score_file.write(f"Test data       : {len(test_set.list_IDs)} \n\n")
    score_file.write(f"====== Train Start ====== \n")

    score_file.flush()

    ### Print log
    print(f"### Model Information ###")
    print(f"Model                     : BEAT2AASIST")
    print(f"Pre-trained BEATs         : {'YES' if args.pre_trained_path else 'NO'}")
    print(f"Feature Split method      : {args.feature_split}")
    print(f"Multi-layer Fusion method : {args.multi_layer_fusion}")
    print(f"Top-k layers              : {args.top_k}")
    print(f"Vocoder Augmentation      : {args.vocoder_aug}")
    print(f"num_params(learnable)     : {sum([param.view(-1).size()[0] for param in model.parameters()])} ({sum(p.numel() for p in model.parameters() if p.requires_grad)})\n")

    print(f"### Data Information ###")
    print(f"Training data   : {len(train_set.list_IDs)}")
    print(f"Validation data : {len(dev_set.list_IDs)}")
    print(f"Evaluation data : {len(eval_set.list_IDs)}")
    print(f"Test data       : {len(test_set.list_IDs)}\n")
    print(f"====== Train Start ======")

    min_test_eer = 10000
    min_epoch = -10

    for epoch in range(args.num_epochs):
        start = time.time()
        running_loss, running_acc = Train_Epoch(train_loader, model, args, optimizer, epoch)
        val_loss,  val_acc,  val_eer,  _         = Evaluate(dev_loader,  model, args, output=val_output,  path = dev_path)
        eval_loss, eval_acc, eval_eer, _         = Evaluate(eval_loader, model, args, output=eval_output, path = eval_path)
        test_loss, test_acc, test_eer, threshold = Evaluate(test_loader, model, args, output=test_output, path = test_path)
        end = time.time()

        score_file.write(f"[EPOCH: {epoch+1}]  train_loss={running_loss : .6f}|train_acc={running_acc : .3f}|  val_loss={val_loss : .6f}|val_acc={val_acc : .3f}|val_eer={val_eer * 100:.5f}|  eval_loss={eval_loss : .6f}|eval_acc={eval_acc : .3f}|eval_eer={eval_eer * 100:.5f} \n")
        score_file.write(f"            test_loss={test_loss : .6f}|test_acc={test_acc : .3f}|test_eer={test_eer * 100:.5f}|thres={threshold:.3f}    |time:{(end-start)/60:.2f} \n")
        score_file.flush()

        print(f"[EPOCH: {epoch+1}]  train_loss={running_loss : .6f}|train_acc={running_acc : .3f}|  val_loss={val_loss : .6f}|val_acc={val_acc : .3f}|val_eer={val_eer * 100:.5f}|  eval_loss={eval_loss : .6f}|eval_acc={eval_acc : .3f}|eval_eer={eval_eer * 100:.5f}")
        print(f"            test_loss={test_loss : .6f}|test_acc={test_acc : .3f}|test_eer={test_eer * 100:.5f}|thres={threshold:.3f}    |time:{(end-start)/60:.2f}")

        if test_eer < min_test_eer:
            min_test_eer = test_eer
            min_epoch = epoch
            torch.save({'model': model.state_dict()}, f'{model_save_path}/saved_model.pt')
            score_file.write("model saved \n")
            score_file.flush()
            print("model saved")

score_file.write(f"\n====== Train Finish ====== \n")
score_file.write(f"TEST EER [epoch: {min_epoch+1}] : {min_test_eer * 100:.5f} \n")
score_file.flush()

print('\n====== Train Finish ======')
print(f"TEST EER [epoch: {min_epoch+1}] : {min_test_eer * 100:.5f}")