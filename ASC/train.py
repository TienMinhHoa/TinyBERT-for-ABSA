import os
import logging
import argparse
import random
import json
from torch.optim import Adam,SGD
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import precision_score,recall_score,f1_score,accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.modeling import BertModel, BertLayer,BertPooler, BertPreTrainedModel, BertConfig
from transformers import AutoModel
from torch.nn import init
import data_utils
from data_utils import ABSATokenizer
from math import ceil
from pytorch_metric_learning.losses import NTXentLoss


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)
mlb = MultiLabelBinarizer()
def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
contrastive_critetion = NTXentLoss(temperature=0.10).to(device)

class GRoIE(nn.Module):
    def __init__(self, count, config, num_labels,drop = False):
        super(GRoIE, self).__init__()
        self.count = count
        self.num_labels = num_labels
        self.pooler = BertPooler(config)
        self.pre_layers = torch.nn.ModuleList()
        self.loss_fct = torch.nn.ModuleList()
        self.classifier = torch.nn.Linear(config.hidden_size, num_labels)
        self.drop = drop
        # self.sm = torch.nn.Softmax(dim=1)
        self.drop = torch.nn.Dropout(p=0.3)
        for i in range(count):
            self.pre_layers.append(BertLayer(config))
            self.loss_fct.append(torch.nn.CrossEntropyLoss(ignore_index=-1))

    def forward(self, layers, attention_mask, labels):
        losses = []
        logitses = []
        out_embedd = []
        for i in range(self.count):
            layer = self.pre_layers[i](layers[-i-1], attention_mask)
            # print(layer.shape)
            layer = self.pooler(layer)
            if self.drop:
                layer = self.drop(layer)
            # print(layer.shape)
            out_embedd.append(layer)
            logits = self.classifier(layer)
            # print(logits.shape)
            if labels is not None:
                loss = self.loss_fct[i](logits.view(-1, self.num_labels), labels.view(-1))
                losses.append(loss)
            logitses.append(logits)
        if labels is not None:
            total_loss = torch.sum(torch.stack(losses), dim=0)
        else:
            total_loss = torch.Tensor(0)
        avg_logits = torch.sum(torch.stack(logitses), dim=0)/self.count
        # print(avg_logits.shape,torch.stack(logitses).shape)
        # avg_logits = self.sm(avg_logits)
        embedd = torch.sum(torch.stack(out_embedd),dim=0)
        # print(embedd.shape)
        return total_loss, avg_logits, embedd/self.count


class BertForABSA(BertPreTrainedModel):
    def __init__(self, config, num_labels=3,drop = False):
        super(BertForABSA, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.bert.load_state_dict(torch.load("/home/student/hoa/NLP/TinyBert/tinybert.pt"))
        # self.bert = torch.load("/home/student/hoa/NLP/TinyBert/ASC/OUT/Rest/Test/model.pt")
        self.groie = GRoIE(2, config, num_labels,drop)
        self.pooler = BertPooler(config)
        self.classifier = torch.nn.Linear(config.hidden_size, num_labels)
        self.loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-1)
        # self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        layers, _ = self.bert(input_ids, token_type_ids, 
                                                        attention_mask=attention_mask, 
                                                        output_all_encoded_layers=True)
        # print(layers[0].shape)
        mask =  attention_mask.unsqueeze(1).unsqueeze(2)
        mask = mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        mask = (1.0 - mask) * -10000.0
        loss, logits,embedds = self.groie(layers, mask, labels)
        # layers = self.pooler(layers[-1])
        # embedds = layers
        
        # logits = self.classifier(layers)
        # # print(logits.shape)
        # if labels is not None:
        #     loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        if labels is not None:
            return loss,embedds
        else:
            return logits

def train(args):
    processor = data_utils.AscProcessor()
    label_list = processor.get_labels()
    tokenizer = ABSATokenizer.from_pretrained("/home/student/hoa/NLP/TinyBert/vocab.txt")
    train_examples = processor.get_train_examples(args.data_dir)
    num_train_steps = int(len(train_examples) / args.batch_size) * args.num_epochs

    train_features = data_utils.convert_examples_to_features(
        train_examples, label_list, 512, tokenizer, "asc")
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
        valid_examples, label_list, 128, tokenizer, "asc")
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

    model = BertForABSA(config=BertConfig("/home/student/hoa/NLP/TinyBert/tinyBertConfig.json"), num_labels = len(label_list),drop=args.drop)
    # model.load_state_dict(torch.load("/home/student/hoa/NLP/TinyBert/ASC/OUT/Rest/Test/state_dict.pt"))
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
    # param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
    # no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    no_decay = []
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

            loss,embedds = model(input_ids, segment_ids, input_mask, label_ids)
            
            if args.do_contrastive:
                contrastive_loss = contrastive_critetion(embedds,label_ids)
                # contrastive_loss.backward()
                loss+=contrastive_loss
            
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
                loss,_ = model(input_ids, segment_ids, input_mask, label_ids)
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
                loss,embedds = model(input_ids, segment_ids, input_mask, label_ids)
                if args.do_contrastive:
                    contrastive_loss = contrastive_critetion(embedds,label_ids)
                    loss+=contrastive_loss
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
    torch.save(model.state_dict(), os.path.join(args.output_dir, "state_dict.pt"))  
    visual(log) 
    with open(os.path.join(args.output_dir,"log.json"), "w") as fw:
        json.dump(log, fw)

def test(args):  # Load a trained model that you have fine-tuned (we assume evaluate on cpu)    
    processor = data_utils.AscProcessor()
    label_list = processor.get_labels()
    tokenizer = ABSATokenizer.from_pretrained("/home/student/hoa/NLP/TinyBert/vocab.txt")
    
    data_dir = args.data_dir
    eval_examples = processor.get_test_examples(data_dir)
    eval_features = data_utils.convert_examples_to_features(eval_examples, label_list, 128, tokenizer, "asc")

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
        json.dump({"logits": full_logits, "label_ids": full_label_ids}, fw)
    


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
    pred = []
    for i in logits:
        lb=np.argmax(i)
        pred.append(lb)
    precision = precision_score(label,pred,average='macro')
    recall = recall_score(label,pred,average='macro')
    f1 = f1_score(label,pred,average='macro')
    acc = accuracy_score(label,pred)
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