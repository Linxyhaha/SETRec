import argparse
import html
import json
import os
import random
import re
import torch
import numpy as np
from transformers import AutoModel, Qwen2Tokenizer

import ipdb

def clean_text(raw_text):
    if isinstance(raw_text, list):
        new_raw_text=[]
        for raw in raw_text:
            raw = html.unescape(raw)
            raw = re.sub(r'</?\w+[^>]*>', '', raw)
            raw = re.sub(r'["\n\r]*', '', raw)
            new_raw_text.append(raw.strip())
        cleaned_text = ' '.join(new_raw_text)
    else:
        if isinstance(raw_text, dict):
            cleaned_text = str(raw_text)[1:-1].strip()
        else:
            cleaned_text = raw_text.strip()
        cleaned_text = html.unescape(cleaned_text)
        cleaned_text = re.sub(r'</?\w+[^>]*>', '', cleaned_text)
        cleaned_text = re.sub(r'["\n\r]*', '', cleaned_text)
    index = -1
    while -index < len(cleaned_text) and cleaned_text[index] == '.':
        index -= 1
    index += 1
    if index == 0:
        cleaned_text = cleaned_text + '.'
    else:
        cleaned_text = cleaned_text[:index] + '.'
    if len(cleaned_text) >= 2000:
        cleaned_text = cleaned_text[:2000]
    return cleaned_text

def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def set_device(gpu_id):
    if gpu_id == -1:
        return torch.device('cpu')
    else:
        return torch.device(
            'cuda:' + str(gpu_id) if torch.cuda.is_available() else 'cpu')

def load_plm(model_path):

    tokenizer = Qwen2Tokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", use_fast=True)
    print("Load Model:", model_path)

    model = AutoModel.from_pretrained(model_path,low_cpu_mem_usage=True,)
    return tokenizer, model

def load_json(file):
    with open(file, 'r') as f:
        data = json.load(f)
    return data


def load_data(args):

    item2feature_path = os.path.join(args.root, f'{args.dataset}.item.json')
    item2feature = load_json(item2feature_path)

    return item2feature

def generate_text(item2feature, features):
    item_text_list = []

    for item in item2feature:
        data = item2feature[item]
        text = []
        for meta_key in features:
            if meta_key in data:
                meta_value = clean_text(data[meta_key])
                text.append(meta_value.strip())

        item_text_list.append([int(item), text])

    return item_text_list

def preprocess_text(args):
    print('Process text data: ')
    print(' Dataset: ', args.dataset)

    item2feature = load_data(args)
    # load item text and clean
    item_text_list = generate_text(item2feature, ['title', 'description', 'brand', 'categories'])
    return item_text_list

def generate_item_embedding(args, item_text_list, tokenizer, model, word_drop_ratio=-1):
    print(f'Generate Text Embedding: ')
    print(' Dataset: ', args.dataset)

    items, texts = zip(*item_text_list)
    order_texts = [[0]] * len(items)
    for item, text in zip(items, texts):
        order_texts[item] = text
    for text in order_texts:
        assert text != [0]

    embeddings = []
    start, batch_size = 0, 1
    while start < len(order_texts):
        if (start+1)%100==0:
            print("==>",start+1)
        field_texts = order_texts[start: start + batch_size]
        # print(field_texts)
        field_texts = zip(*field_texts)

        field_embeddings = []
        for item, sentences in enumerate(field_texts):
            sentences = list(sentences)
            # print(sentences)
            if word_drop_ratio > 0:
                print(f'Word drop with p={word_drop_ratio}')
                new_sentences = []
                for sent in sentences:
                    new_sent = []
                    sent = sent.split(' ')
                    for wd in sent:
                        rd = random.random()
                        if rd > word_drop_ratio:
                            new_sent.append(wd)
                    new_sent = ' '.join(new_sent)
                    new_sentences.append(new_sent)
                sentences = new_sentences

            encoded_sentences = tokenizer(sentences, max_length=args.max_sent_len,
                                              truncation=True, return_tensors='pt',padding="longest").to(args.device)

            with torch.no_grad():
                outputs = model(input_ids=encoded_sentences.input_ids,
                                attention_mask=encoded_sentences.attention_mask)

            masked_output = outputs.last_hidden_state * encoded_sentences['attention_mask'].unsqueeze(-1)
            mean_output = masked_output.sum(dim=1) / encoded_sentences['attention_mask'].sum(dim=-1, keepdim=True)

            mean_output = mean_output.detach().cpu()
            field_embeddings.append(mean_output)
            
        field_mean_embedding = torch.stack(field_embeddings, dim=0).mean(dim=0)
        embeddings.append(field_mean_embedding)
        start += batch_size

    embeddings = torch.cat(embeddings, dim=0).numpy()
    print('Embeddings shape: ', embeddings.shape)

    file = os.path.join(args.root, args.dataset + '.emb-' + args.plm_name + "-td" + ".npy")
    np.save(file, embeddings)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='toys', help='toys / beauty / sports / steam')
    parser.add_argument('--root', type=str, default="./")
    parser.add_argument('--gpu_id', type=int, default=0, help='ID of running GPU')
    parser.add_argument('--plm_name', type=str, default='qwen')
    parser.add_argument('--plm_checkpoint', type=str, default='/your/path/to/pre-trained/model/')
    parser.add_argument('--max_sent_len', type=int, default=2048)
    parser.add_argument('--word_drop_ratio', type=float, default=-1, help='word drop ratio, do not drop by default')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    args.root = os.path.join(args.root, args.dataset)

    device = set_device(args.gpu_id)
    args.device = device

    item_text_list = preprocess_text(args)

    plm_tokenizer, plm_model = load_plm(args.plm_checkpoint)
    if plm_tokenizer.pad_token_id is None:
        plm_tokenizer.pad_token_id = 0
    plm_model = plm_model.to(device)

    generate_item_embedding(args, item_text_list,plm_tokenizer,
                            plm_model, word_drop_ratio=args.word_drop_ratio)


