from sentence_transformers import SentenceTransformer
import numpy as np
import torch
import ipdb

sentences = ["This is an example sentence", "Each sentence is converted"]

# TODO: check dataset and outputdir
dataset="steam"

items = np.load(f'./{dataset}/combine_tdcb_maps.npy', allow_pickle=True).item()
item_names = list(items['recid2combine'].values())
item_ids = [_ for _ in range(len(item_names))]

movie_dict = dict(zip(item_names, item_ids))
result_dict = dict()


def batch(list, batch_size=1):
    chunk_size = (len(list) - 1) // batch_size + 1
    for i in range(chunk_size):
        yield list[batch_size * i: batch_size * (i + 1)]

item_embeddings = []
from tqdm import tqdm

model = SentenceTransformer('sentence-transformers/sentence-t5-base')

model.eval()

for i, name in tqdm(enumerate(batch(item_names, 8))):
        input = name
        embeddings = model.encode(name)
        item_embeddings.append(torch.tensor(embeddings))

item_embeddings = torch.cat(item_embeddings, dim=0)

# save item_embeddings
item_embeddings = item_embeddings.numpy()
np.save(f'./{dataset}/{dataset}.emb-t5-tdcb.npy', item_embeddings)