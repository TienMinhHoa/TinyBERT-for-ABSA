import os
import logging
from tqdm import tqdm
import argparse
import random
import json
from math import ceil
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.optim import Adam,SGD
from torch.nn import init
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import data_utils
from sklearn.metrics import precision_score,recall_score,f1_score,accuracy_score
from data_utils import ABSATokenizer
from pathlib import Path
from sklearn.preprocessing import MultiLabelBinarizer
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam
# from optimization import BertAdam
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel, BertLayer, BertPreTrainedModel, BertConfig
from transformers import AutoModel
from torchcrf import CRF


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")


class GRoIE(nn.Module):
    def __init__(self, count, config, num_labels,drop = False):
        super(GRoIE, self).__init__()
        self.count = count
        self.num_labels = num_labels
        self.pre_layers = torch.nn.ModuleList()
        self.crf_layers = torch.nn.ModuleList()
        self.classifier = torch.nn.Linear(config.hidden_size, num_labels)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        for i in range(count):
            self.pre_layers.append(BertLayer(config))
            self.crf_layers.append(CRF(num_labels))

    def forward(self, layers, attention_mask, labels):
        losses = []
        logitses = []
        for i in range(self.count):
            layer = self.pre_layers[i](layers[-i-1], attention_mask)
            layer = self.dropout(layer)
            logits = self.classifier(layer)
            
            # print(logits.shape,labels.shape)
            if labels is not None:
                loss = self.crf_layers[i](logits.view(logits.shape[0], -1, self.num_labels), labels.view(labels.shape[0], -1))
                losses.append(loss)
            logitses.append(logits)
        if labels is not None:
            total_loss = torch.sum(torch.stack(losses), dim=0)
        else:
            total_loss = torch.Tensor(0)
        avg_logits = torch.sum(torch.stack(logitses), dim=0)/self.count
        return -total_loss, avg_logits


class BertForABSA(BertPreTrainedModel):
    def __init__(self, config, num_labels=3):
        super(BertForABSA, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.bert.load_state_dict(torch.load("/home/student/hoa/NLP/TinyBert/pytorch_model.pt"))
        self.groie = GRoIE(2, config, num_labels)
        # self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        layers, _ = self.bert(input_ids, token_type_ids, 
                                                        attention_mask=attention_mask, 
                                                        output_all_encoded_layers=True)
        mask =  attention_mask.unsqueeze(1).unsqueeze(2)
        mask = mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        mask = (1.0 - mask) * -10000.0
        
        loss, logits = self.groie(layers, mask, labels)
        if labels is not None:
            return loss
        else:
            return logits

def train(args):
    processor = data_utils.AeProcessor()
    label_list = processor.get_labels()
    tokenizer = ABSATokenizer.from_pretrained("/home/student/hoa/NLP/TinyBert/vocab.txt")
    train_examples = processor.get_train_examples(args.data_dir)
    num_train_steps = int(len(train_examples) / args.batch_size) * args.num_epochs

    train_features = data_utils.convert_examples_to_features(
        train_examples, label_list, 128, tokenizer, "ae")
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_examples))
    logger.info("  Batch size = %d", args.batch_size)
    logger.info("  Num steps = %d", num_train_steps)
    
    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
    
    train_data = TensorDataset(all_input_ids, all_segment_ids, all_input_mask, all_label_ids)

    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size)
    
    #>>>>> validation

    valid_examples = processor.get_dev_examples(args.data_dir)
    valid_features=data_utils.convert_examples_to_features(
        valid_examples, label_list, 128, tokenizer, "ae")
    valid_all_input_ids = torch.tensor([f.input_ids for f in valid_features], dtype=torch.long)
    valid_all_segment_ids = torch.tensor([f.segment_ids for f in valid_features], dtype=torch.long)
    valid_all_input_mask = torch.tensor([f.input_mask for f in valid_features], dtype=torch.long)
    valid_all_label_ids = torch.tensor([f.label_id for f in valid_features], dtype=torch.long)
    valid_data = TensorDataset(valid_all_input_ids, valid_all_segment_ids, valid_all_input_mask, valid_all_label_ids)

    logger.info("***** Running validations *****")
    logger.info("  Num orig examples = %d", len(valid_examples))
    logger.info("  Num split examples = %d", len(valid_features))
    logger.info("  Batch size = %d", args.batch_size)

    valid_sampler = SequentialSampler(valid_data)
    valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=args.batch_size)    

    best_valid_loss=float('inf')
    #<<<<< end of validation declaration

    model = BertForABSA(config=BertConfig("/home/student/hoa/NLP/TinyBert/config.json"), num_labels = len(label_list))
    if args.re_init:
        for module in model.bert.encoder.layer[0].modules():
            if isinstance(module, torch.nn.Linear):
                init.xavier_uniform_(module.weight)  # Khởi tạo trọng số
                if module.bias is not None:
                    init.zeros_(module.bias)  # Khởi tạo bias
            elif isinstance(module, torch.nn.LayerNorm):
                init.ones_(module.weight)  # Khởi tạo gamma
                init.zeros_(module.bias)
    
    model.to(device)
    
    
    
    ##>>>>>>>> freezing
    # for param in model.bert.parameters(): 
    #     param.requires_grad = False
    ##>>>>>>>>>>>
    # Prepare optimizer
    param_optimizer = [(k, v) for k, v in model.named_parameters()]
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    # no_decay = []
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    t_total = num_train_steps
    if args.bert_adam:
        optimizer = BertAdam(optimizer_grouped_parameters,
                            lr=args.learning_rate,
                            warmup=args.warmup_proportion,
                            t_total=t_total)
    else:
        optimizer = Adam(optimizer_grouped_parameters,lr=args.learning_rate)
    

    global_step = 0
    
    log= {"train_loss":[],
          "val_loss":[],
          "train_precision":[],
          "val_precision":[],
          "train_recall":[],
          "val_recall":[],
          "train_f1":[],
          "val_f1":[],
          "train_acc":[],
          "val_acc":[]}
    model.train()
    for epoch in tqdm(range(args.num_epochs)):
        total_train_loss = 0
        for step, batch in tqdm(enumerate(train_dataloader)):
            batch = tuple(t.to(device) for t in batch)
            input_ids, segment_ids, input_mask, label_ids = batch
            
            # print(input_ids.shape)
            # print(input_mask.shape)

            loss = model(input_ids, segment_ids, input_mask, label_ids)
            
            total_train_loss+=loss.item()
            loss.backward()
            
            lr_this_step = args.learning_rate * warmup_linear(global_step/t_total, args.warmup_proportion)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_this_step
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
            #>>>> perform validation at the end of each epoch .
        print("training loss: ", loss.item(), epoch+1)
        log["train_loss"].append(total_train_loss/len(train_dataloader))
        new_dirs = os.path.join(args.output_dir, str(epoch+1))
        Path(new_dirs).mkdir(parents = True,exist_ok = True)
                
        #>>do validation
        model.eval()
        with torch.no_grad():
            train_size=0
            full_logits =[]
            full_label_ids = []
            for step, batch in enumerate(train_dataloader):
                batch = tuple(t.to(device) for t in batch) # multi-gpu does scattering it-self
                input_ids, segment_ids, input_mask, label_ids = batch
                loss = model(input_ids, segment_ids, input_mask, label_ids)
                logits = model(input_ids, segment_ids, input_mask)
                train_size+=input_ids.size(0)
                
                logits = logits.detach().cpu().numpy()
                label_ids = label_ids.cpu().numpy()

                full_logits.extend(logits.tolist())
                full_label_ids.extend(label_ids.tolist())
            precision,recall,f1,acc = eval(full_label_ids,full_logits)
            log["train_precision"].append(precision)
            log["train_recall"].append(recall)
            log["train_f1"].append(f1)
            log["train_acc"].append(acc)
            
            
            losses=[]
            total_val_loss = 0
            valid_size=0
            full_logits =[]
            full_label_ids = []
            for step, batch in enumerate(valid_dataloader):
                batch = tuple(t.to(device) for t in batch) # multi-gpu does scattering it-self
                input_ids, segment_ids, input_mask, label_ids = batch
                loss = model(input_ids, segment_ids, input_mask, label_ids)
                total_val_loss+=loss.item()
                logits = model(input_ids, segment_ids, input_mask)
                losses.append(loss.item() )
                valid_size+=input_ids.size(0)
                
                logits = logits.detach().cpu().numpy()
                label_ids = label_ids.cpu().numpy()

                full_logits.extend(logits.tolist())
                full_label_ids.extend(label_ids.tolist())
            precision,recall,f1,acc = eval(full_label_ids,full_logits)
            valid_loss=sum(losses)/len(valid_dataloader)
            logger.info("validation loss: %f, epoch: %d", total_val_loss/len(valid_dataloader), epoch+1)
            # valid_losses.append(valid_loss)
            log["val_loss"].append(total_val_loss/len(valid_dataloader))
            log["val_precision"].append(precision)
            log["val_recall"].append(recall)
            log["val_f1"].append(f1)
            log['val_acc'].append(acc)
            torch.save(model, os.path.join(new_dirs, "model.pt"))
            if epoch == args.num_epochs-1:
                torch.save(model, os.path.join(args.output_dir, "model.pt"))
        if valid_loss<best_valid_loss:
            best_valid_loss=valid_loss
        model.train()
        torch.save(model, os.path.join(args.output_dir, "model.pt")) 
    torch.save(model, os.path.join(args.output_dir, "model.pt"))  
    with open(os.path.join(args.output_dir,"log.json"), "w") as fw:
        json.dump(log,fw)
    visual(log) 

def test(args):  # Load a trained model that you have fine-tuned (we assume evaluate on cpu)    
    processor = data_utils.AeProcessor()
    label_list = processor.get_labels()
    tokenizer = ABSATokenizer.from_pretrained("/home/student/hoa/NLP/TinyBert/vocab.txt")
    
    data_dir = args.data_dir
    eval_examples = processor.get_test_examples(data_dir)
    eval_features = data_utils.convert_examples_to_features(eval_examples, label_list, 128, tokenizer, "ae")

    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.batch_size)
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
    eval_data = TensorDataset(all_input_ids, all_segment_ids, all_input_mask, all_label_ids)
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.batch_size)

    model = torch.load(args.model_path)
    model.to(device)
    model.eval()
    
    full_logits=[]
    full_label_ids=[]
    for step, batch in enumerate(eval_dataloader):
        batch = tuple(t.to(device) for t in batch)
        input_ids, segment_ids, input_mask, label_ids = batch
        
        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask)

        logits = logits.detach().cpu().numpy()
        label_ids = label_ids.cpu().numpy()

        full_logits.extend(logits.tolist())
        full_label_ids.extend(label_ids.tolist())
        
        

    output_eval_json = os.path.join(args.output_dir, "test_predictions.json")  
    with open(output_eval_json, "w") as fw:
        assert len(full_logits)==len(eval_examples)
        #sort by original order for evaluation
        recs={}
        for qx, ex in enumerate(eval_examples):
            recs[int(ex.guid.split("-")[1]) ]={"sentence": ex.text_a, "idx_map": ex.idx_map, "logit": full_logits[qx][1:]} #skip the [CLS] tag.
        full_logits=[recs[qx]["logit"] for qx in range(len(full_logits))]
        raw_X=[recs[qx]["sentence"] for qx in range(len(eval_examples) )]
        idx_map=[recs[qx]["idx_map"] for qx in range(len(eval_examples))]
        json.dump({"logits": full_logits, "raw_X": raw_X, "idx_map": idx_map}, fw)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-b",'--batch_size',default=16,type = int)
    parser.add_argument("-e",'--num_epochs',default=50,type=int)
    parser.add_argument("-l","--learning_rate",default=3e-5,type=float)
    parser.add_argument("-o","--output_dir",type=str)
    parser.add_argument("-w","--warmup_proportion",default=0.1,type = float)
    parser.add_argument("--do_test",default=False,action="store_true")
    parser.add_argument("--do_train",default=False,action="store_true")
    parser.add_argument("--model_path",type = str)
    parser.add_argument("--data_dir",type=str)
    parser.add_argument("--do_contrastive",default=False,action="store_true")
    parser.add_argument("--bert_adam",default=False,action="store_true")
    parser.add_argument("--drop",default=False,action="store_true")
    parser.add_argument("--re_init",default=False,action="store_true")
    args= parser.parse_args()
    return args

def eval(label,logits):
    truth = []
    for i in tqdm(label):
        tmp = []
        for charac in i:
            if charac!=-1:
                tmp.append(charac)
        truth.append(tmp)
    mlb = MultiLabelBinarizer()
    # print(label)
    
    preds = []
    for idx,logit in enumerate(logits):
        # print(len(logit))
        # print(label[idx])
        pred=[0]*len(truth[idx])
        for id,tmp in enumerate(truth[idx]):
            lb = np.argmax(logit[id])
            if lb==1: #B
                pred[id]=1
            elif lb==2: #I
                if pred[id]==0: #only when O->I (I->I and B->I ignored)
                    pred[id]=2
        preds.append(pred)
    label= mlb.fit_transform(truth)
    
    preds = mlb.transform(preds)
    precision = precision_score(label,preds,average='macro')
    recall = recall_score(label,preds,average='macro')
    f1 = f1_score(label,preds,average='macro')
    acc = accuracy_score(label,preds)
    return precision,recall,f1,acc
        
def visual(log):
    Path(os.path.join(args.output_dir, 'plot_train')).mkdir(parents=True,exist_ok=True)
    for metric in ['loss', 'acc', 'precision', 'recall', 'f1']:
        # plot the training loss
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(log["train_" + metric], label="train_" + metric)
        plt.plot(log["val_" + metric], label="val_" + metric)
        plt.title("Training " + metric + " on Dataset")
        plt.xlabel("Epoch #")
        plt.ylabel(metric)
        plt.legend(loc="lower left")
        plt.savefig(os.path.join(args.output_dir, f'plot_train/{metric}.png'))
        plt.close()
if __name__=="__main__":
    args= main()
    if args.do_train:
        train(args=args)
    if args.do_test:
        test(args=args)