import transformers, torch, time, argparse
import numpy as np
from tqdm import tqdm
from scipy import stats
from src.model import *
from src.loss import *
from src.datasets import *
from src.optimizer import *
from src.utils import *
transformers.logging.set_verbosity_error()

def eval(model, data, loss_function, if_dev, args):
    scores = dict()
    all_y_pred = None
    all_labels = None
    model.eval() 
    loss = 0.
    for batch in data:
        with torch.no_grad():
            data = To_tensor(batch)
            y_pred = model(**data) 
            labels = data['labels']
            label_tensor = torch.tensor(labels)
            label_tensor = label_tensor.to(args.device)
            loss += loss_function(y_pred, label_tensor).item()
            y_pred_arr = y_pred.detach().cpu().numpy()
            ls = np.array(labels) 
            if all_y_pred is None:  
                all_y_pred = y_pred_arr
                all_labels = ls
            else:  
                all_y_pred = np.concatenate((all_y_pred, y_pred_arr), 0)
                all_labels = np.concatenate((all_labels, ls), 0)
    pred_labels = all_y_pred.squeeze()  
    true_labels = all_labels
    scores['mse'] = (np.square(true_labels - pred_labels)).mean()
    scores['pearson r'] = stats.pearsonr(pred_labels, true_labels)[0]
    if if_dev:
        data_name = "DEV"
    else:
        data_name = "TRAIN"
    print("Evaling on \"{}\" data".format(data_name))
    for s_name, s_val in scores.items(): 
        print("{}: {}".format(s_name, s_val)) 
    return scores


def train(exp_name, model, train_data, dev_data, loss_function, optimizer, scheduler, args):
    best_dev_score = -9999
    for e in range(args.epochs):
        model.train()
        print(f'----------------epoches{e}----------------')
        loss = 0.  # clear the loss
        start_time = time.time()
        for i, batch in enumerate(tqdm(train_data)):
            model.zero_grad()
            data = To_tensor(batch)
            y_pred = model(**data) 
            labels = data['labels']
            label_tensor = torch.tensor(labels)
            label_tensor = label_tensor.to(args.device)
            graph_loss = loss_function(y_pred, label_tensor)
            loss += graph_loss.item()  # update loss 
            graph_loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            # if i % args.check_steps==0 and i>0:
        end_time = time.time()
        print(f'\t epoch: {e} | Took: {((end_time - start_time)/60.):.1f} min')
        train_scores = eval(model, train_data, loss_function, False, args)
        dev_scores = eval(model, dev_data, loss_function, True, args)
        f = open('{}/{}.all_scores.txt'.format(args.result_path, exp_name), 'a')
        f.write(' ==================================================  Epoch: {}  ==================================================\n'.format(e))
        f.write('TrainScore: \n{}\nEvalScore: \n{}\n'.format(json.dumps(train_scores), json.dumps(dev_scores))) 
        f.close()
        if dev_scores['pearson r'] >= best_dev_score:
            best_dev_score = dev_scores['pearson r']
            save(e+1, model, exp_name, args.checkpoint_path, 'BEST')
    save(e+1, model, exp_name, args.checkpoint_path, 'FINAL')






if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_dir", help="The input data dir.", 
                        default='./data/preprocess_train.csv')
    parser.add_argument("-t", "--test_data_dir", help="The test data dir.", 
                        default='./data/preprocess_train.csv')
    parser.add_argument("-c", "--checkpoint_path", help="The output directory where the model checkpoints will be written.",
                        default='./checkpoints/')
    parser.add_argument("-r", "--result_path", help="The output directory where the model results will be written.",
                        default='./results')
    parser.add_argument("-m", "--model_name_or_path", help="Path to pretrained model or model identifier from huggingface.co/models.",
                        default='cardiffnlp/twitter-xlm-roberta-base-sentiment')
    parser.add_argument("-n", "--name", help="The model name for saving.",
                        default='XLMModel')
    parser.add_argument("-hd", "--hidden_dim", type=int, help="The hidden_dim of FFNN.",
                        default=113)
    parser.add_argument("-de", "--device", help="The device of training model.",
                        default='cuda:0')
    parser.add_argument("--embed_dim", type=int, help="Embedding dimension of pre-training language model.",
                        default=768)
    parser.add_argument("--weight_decay", type=float, help="Weight decay if we apply some.", 
                        default=0.0, )
    parser.add_argument("--adam_epsilon", type=float, help="Epsilon for Adam optimizer.", 
                        default=1e-8,)
    parser.add_argument("--warmup_proportion", type=float, help="Linear warmup over warmup_steps.",
                        default=0.1)
    parser.add_argument("-ld", "--load_model", type=str, help="load existing dumped model parameter.",
                        default='')
    parser.add_argument("-ml", "--max_tok_len", type=int, default=50)
    parser.add_argument("-l", "--learning_rate", type=float, default=1e-5)
    parser.add_argument("-b", "--batchsize", type=int, default=32)
    parser.add_argument("-drop", "--dropout", type=float, default=0.5)
    parser.add_argument("-e", "--epochs", type=int, default=10)
    parser.add_argument("-s", "--seed", type=int, default=1234)
    args = parser.parse_args()
    
    setup_seed(args.seed)
    train_df, dev_df = build_data(args.data_dir)
    # train_df.to_csv('train_df.csv',sep=',', header=True)
    train_data = BertData('TRAIN', train_df, args)
    dev_data = BertData('DEV', dev_df, args)
    print('The size of the training dataset: {}'.format(len(train_data)))
    print('The size of the Validation dataset: {}'.format(len(dev_data)))
    dataloader = BatchData(train_data,  batch_size=args.batchsize)
    dev_dataloader = BatchData(dev_data, batch_size=args.batchsize)
    num_training_steps = len(train_data) * int(args.epochs) 

    model = Baseline(args)

    stance_loss_function = Intimacy_Loss()
    optimizer, scheduler = build_optimizer(model, num_training_steps, args)
    
    exp_name = '{}_D-{}_B-{}_H-{}_E-{}_Lr-{}_SEED-{}'.format(args.name, args.dropout, args.batchsize, args.hidden_dim, args.epochs, args.learning_rate, args.seed)
    if args.load_model != '':
        model.load_state_dict(torch.load(args.load_model, args.device)['model_state_dict'])
    train(exp_name, model, dataloader, dev_dataloader, stance_loss_function, optimizer, scheduler, args)