import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
from Q_t5 import *
from AE.models.ae import AE
from transformers import T5Tokenizer

def generate_position_ids(total_length, i, j, n, device=None):
    """
    :param total_length: seq_length
    :param i: instruction token length
    :param j: response token length
    :param n: n_query within each item
    :return: PyTorch tensor
    """
    assert i + j < total_length, "i + j should be smaller than total_length"
    # create base vector
    vector = torch.arange(total_length, dtype=torch.long, device=device)
    # mask middle item sequence part
    middle_mask = (vector >= i) & (vector < total_length - j)
    # change the middle item sequence part, let the position id be the same within the items
    vector[middle_mask] = i + torch.div((vector[middle_mask] - i), n, rounding_mode='floor')
    return vector

def new_compute_bias(self, seq_length, key_length, device=None):
    """Compute binned relative position bias"""
    if device is None:
        device = self.relative_attention_bias.weight.device

    context_position = generate_position_ids(seq_length, self.n_instruct, self.n_response, self.n_query, device=device)[:, None]
    memory_position = generate_position_ids(key_length, self.n_instruct, self.n_response, self.n_query, device=device)[None, :]

    relative_position = memory_position - context_position  # shape (query_length, key_length)
    relative_position_bucket = self._relative_position_bucket(
        relative_position,  # shape (query_length, key_length)
        bidirectional=(not self.is_decoder),
        num_buckets=self.relative_attention_num_buckets,
        max_distance=self.relative_attention_max_distance,
    )
    values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
    values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
    return values

def get_position_ids_for_qdecoder(self, seq_length, key_length, device=None):
    context_position = torch.zeros(seq_length, dtype=torch.long, device=device)[:, None]
    memory_position = torch.zeros(key_length, dtype=torch.long, device=device)[None, :]
    relative_position = memory_position - context_position  # shape (query_length, key_length)
    relative_position_bucket = self._relative_position_bucket(
        relative_position,  # shape (query_length, key_length)
        bidirectional=(not self.is_decoder),
        num_buckets=self.relative_attention_num_buckets,
        max_distance=self.relative_attention_max_distance,
    )
    values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
    values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
    return values

class T54Rec(nn.Module):
    def __init__(self, **args):
        super(T54Rec, self).__init__()
        self.args = args
        self.n_query = args["n_query"]
        self.n_cf = args["n_cf"]
        self.n_sem = args["n_sem"]
        self.alpha = args["alpha"] # coefficient of tokenizer loss

        print(f'Initializing language decoder ...')

        self.t5_model = QT5.from_pretrained(self.args['base_model'], local_files_only=True, cache_dir=args['cache_dir'], device_map=self.args['device_map'], n_query=self.args['n_query'])
        
        if args.get("inference", None) is None:
            nn.init.normal_(self.t5_model.decoder.query_emb.weight.data, mean=0, std=0.02) # manualy initialize, otherwise the initialization gives a very large range
        
        self.t5_model.config.use_cache = False

        self.t5_tokenizer = T5Tokenizer.from_pretrained(self.args['base_model'], use_fast=False, local_files_only=True, cache_dir=args['cache_dir'])
        self.t5_tokenizer.pad_token_id = 0
        self.t5_tokenizer.padding_side = "left"
        self.instruct_ids, self.instruct_mask = self.t5_tokenizer(self.args['instruction_text'][0],
                                                                     truncation=True, padding=False,
                                                                     return_tensors='pt', add_special_tokens=False).values()
        self.response_ids, self.response_mask = self.t5_tokenizer(self.args['instruction_text'][1],
                                                                     truncation=True, padding=False,
                                                                     return_tensors='pt', add_special_tokens=False).values()
        self.input_dim=args['input_dim']
        print("instruction:", self.args['instruction_text'][0])
        print("response_split:", self.args['instruction_text'][1])

        self.replace_position_func()
        print('Language decoder initialized.')

        self.m_item = self.args['m_item']
        self.emb_dim = self.t5_model.config.hidden_size
        
        # initialize CF token
        padding_embed = torch.zeros(1, self.input_dim)
        concat_embed = torch.cat([padding_embed, self.args['input_embeds']], dim=0)
        self.input_embeds = nn.ModuleList([nn.Embedding.from_pretrained(concat_embed, freeze=True)])

        # initialize CF projector (cf token dim -> LLM token dim)
        self.input_proj = nn.Linear(self.input_dim, self.t5_model.config.hidden_size)

        # initialize sem tokenizer
        if self.n_sem:
            self.tokenizer = AE(in_dim=args["in_dim"],
                                e_dim=self.t5_model.config.hidden_size * (self.n_query-self.n_cf),
                                layers=args["layers"],
                                dropout_prob=args["dropout_prob"],
                                bn=args["bn"],
                                loss_type=args["loss_type"],
                                item_feature=args["item_feature"]
                                )
            
        # initialize CE Loss for recommendation
        self.criterion = nn.CrossEntropyLoss(reduction='mean')

    def replace_position_func(self):
        # replace compute_bias function
        for layer in self.t5_model.encoder.block:
            layer.layer[0].SelfAttention.n_query = self.n_query
            layer.layer[0].SelfAttention.n_instruct = self.instruct_ids.shape[-1]
            layer.layer[0].SelfAttention.n_response = self.response_ids.shape[-1]
            layer.layer[0].SelfAttention.compute_bias = new_compute_bias.__get__(layer.layer[0].SelfAttention)

        for layer in self.t5_model.decoder.block:
            layer.layer[0].SelfAttention.compute_bias = get_position_ids_for_qdecoder.__get__(layer.layer[0].SelfAttention)
            layer.layer[1].EncDecAttention.compute_bias = get_position_ids_for_qdecoder.__get__(layer.layer[1].EncDecAttention)

    def tokenize_all(self):
        '''
        This function tokenize all item ids into the semantic token embeddings. 
        '''
        idx_tensor = torch.arange(self.m_item, dtype=torch.long).cuda()
        self.tokenizer.item_feature = self.tokenizer.item_feature.to(idx_tensor.device)
        sem_input = self.tokenizer.item_feature[idx_tensor] # (n_unique, in_dim)
        sem_emb, recon_emb = self.tokenizer(sem_input) # (n_item, dim*n_sem) 
        # expected shape (n_item, n_sem, dim)
        sem_emb = sem_emb.view(sem_emb.shape[0], self.n_sem, -1)
        # expected self.all_sem shape # (n_sem, n_item, dim)
        self.all_sem = sem_emb.transpose(0,1)
        return recon_emb
    
    def get_batch_sem(self, item_id, all_recon_emb):
        '''
        This function takes the item id as input, and extract these items' semantic token embeddings from all sem token embeddings. 
        '''
        # initialize a zero tensor
        item_sem_token = torch.zeros(item_id.shape[0], item_id.shape[1], self.n_sem, self.emb_dim, dtype=self.all_sem.dtype).cuda() # (bs, seq_len, n_sem, dim)
        
        non_zero_mask = self.batch_nonzero_mask # (bs, seq_len)
        non_zero_ids = item_id[non_zero_mask] #  (n_nonzero, )

        item_sem_token[non_zero_mask] = self.all_sem.transpose(0,1)[non_zero_ids-1] # (n_nonzero, n_sem, dim)

        sem_input = self.tokenizer.item_feature[non_zero_ids-1] # (n_nonzero, 4096)
        recon_emb = all_recon_emb[non_zero_ids-1] # (n_nonzero, 4096)

        sem_emb = self.all_sem.transpose(0,1)[non_zero_ids-1] # (n_nonzero, n_sem, dim)
        return sem_input, sem_emb, recon_emb, item_sem_token
        
    
    def predict(self, inputs_id, inputs_mask, inference=False):
        self.batch_nonzero_mask = inputs_id!=0

        if not inference and self.n_sem: 
            self.recon_all = self.tokenize_all()

        x, unique_sem_emb, x_hat, item_sem_token = None, None, None, None

        # 1. get semantic tokens
        if self.n_sem:
            # put current batch item into the AE to get the sem emb
            x, unique_sem_emb, x_hat, item_sem_token = self.get_batch_sem(inputs_id, self.recon_all)
                                                        # x: (n_nonzero, input_dim)
                                                        # unique_sem_emb (n_nonzero, n_sem, dim)
                                                        # x_hat (n_nonzero, input_dim)
                                                        # item_sem_token (bs, seq_len, n_sem, dim)

        bs = inputs_id.shape[0]
        instruct_embeds = self.t5_model.shared(self.instruct_ids.cuda()).expand(bs, -1, -1)
        response_embeds = self.t5_model.shared(self.response_ids.cuda()).expand(bs, -1, -1)
        answer_embeds = self.t5_model.decoder.query_emb(self.t5_model.decoder.query_input_ids.cuda()).expand(bs, -1, -1)

        instruct_mask = self.instruct_mask.cuda().expand(bs, -1)
        response_mask = self.response_mask.cuda().expand(bs, -1)

        # 2. get cf tokens
        inputs = self.input_proj(self.input_embeds[0](inputs_id))# (bs, seq_len, dim) get the cf embedding

        # 3. concat cf emb and sem emb
        inputs = inputs.unsqueeze(-2)
        inputs = torch.cat([inputs, item_sem_token], dim=2) if self.n_sem else inputs # (bs, seq_len, n_sem+n_cf(1), dim) 
        inputs = inputs.view(bs, -1, self.t5_model.config.hidden_size)

        inputs = torch.cat([instruct_embeds, inputs, response_embeds], dim=1)
        attention_mask = torch.cat([instruct_mask, inputs_mask, response_mask], dim=1)
        assert attention_mask.size()[0] == inputs.size()[0] and attention_mask.size()[1] == inputs.size()[1]

        # 4. query-guided simultaneous decoding (t5 forward)
        outputs = self.t5_model(inputs_embeds=inputs, attention_mask=attention_mask, decoder_inputs_embeds=answer_embeds, return_dict=True, output_hidden_states=True)
        
        # 5. grounding
        query_output = outputs.decoder_hidden_states[-1][:,-self.n_query:] # (bs, n_query, dim)
        output_emb = query_output.transpose(0,1) # (n_query, bs, emb_dim)
        
        if not inference:
            # get all cf and all sem
            idx_tensor = torch.arange(self.m_item, dtype=torch.long).cuda()
            self.all_cf = self.input_proj(self.input_embeds[0](idx_tensor+1)).unsqueeze(0) # (n_cf(1), n_item, dim)
            # concat cf and sem, then dot product
            mat = torch.cat([self.all_cf, self.all_sem], dim=0) if self.n_sem else self.all_cf # (n_query, n_item, dim)
            mat = mat.transpose(1,2)  # (n_query, dim, n_item)
            output = torch.bmm(output_emb, mat) # (n_query, bs, n_item)
            return output_emb, outputs, output, x, unique_sem_emb, x_hat
        
        if inference:
            # concat cf and sem, then dot product
            mat = torch.cat([self.all_cf, self.all_sem], dim=0) if self.n_sem else self.all_cf # (n_query, n_item, dim)
            mat = mat.transpose(1,2)  # (n_query, dim, n_item)
            output = torch.bmm(output_emb, mat) # (n_query, bs, n_item)
            weight = self.beta[:, None, None].to(self.t5_model.device)
            score = torch.sum(output * weight, dim=0).squeeze()
            return outputs, score, x, unique_sem_emb, x_hat


    def forward(self, inputs, inputs_mask, labels):
        _, outputs, scores, x, _, x_hat = self.predict(inputs, inputs_mask) # prediction (bs, n_query, dim)

        loss = None
        logits = None
        if labels is not None:
            # generative recommendation loss
            scores = torch.cat([_ for _ in scores],dim = 0)
            all_item_gold = labels.squeeze().repeat(1,self.n_query).squeeze() #(n_query*bs)
            loss = self.criterion(scores, all_item_gold)

            # tokenizer loss (i.e., autoencoder loss)
            loss_ae = 0
            if self.n_sem:
                loss_ae = F.mse_loss(x, x_hat, reduction='mean')
            
            # Eq.(8) in paper
            loss += self.alpha * loss_ae

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
        )
     