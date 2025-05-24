import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen2Tokenizer
from Q_qwen import QQwen2Model
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
from AE.models.ae import AE

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_int8_training,
)


class Qwen4Rec(nn.Module):
    def __init__(self, **args):
        super(Qwen4Rec, self).__init__()
        self.args = args
        self.input_dim = args['input_dim']
        self.m_item = self.args['m_item']
        self.n_query = args["n_query"]
        self.n_cf = args["n_cf"]
        self.n_sem = args["n_sem"]
        self.alpha = args["alpha"]

        # add the lora module
        peft_config = LoraConfig(
            task_type='FEATURE_EXTRACTION',
            r=self.args['lora_r'],
            lora_alpha=self.args['lora_alpha'],
            lora_dropout=self.args['lora_dropout'],
            target_modules=self.args['lora_target_modules'],
            bias='none',
        )

        self.qwen_model = QQwen2Model.from_pretrained(self.args['base_model'], load_in_8bit=True, torch_dtype=torch.float16,
                                                      local_files_only=False, cache_dir=args['cache_dir'],
                                                      device_map=self.args['device_map'])
        self.qwen_model.n_query = args['n_query']
        self.qwen_model = prepare_model_for_int8_training(self.qwen_model)
        self.qwen_model = get_peft_model(self.qwen_model, peft_config)
        self.qwen_model.print_trainable_parameters()
        self.qwen_model.config.use_cache = False

        self.qwen_tokenizer = Qwen2Tokenizer.from_pretrained(self.args['base_model'], use_fast=True)
        
        self.qwen_tokenizer.pad_token_id = 151643
        self.qwen_tokenizer.padding_side = "right"
        self.instruct_ids, self.instruct_mask = self.qwen_tokenizer(self.args['instruction_text'][0],
                                                                     truncation=True, padding=False,
                                                                     return_tensors='pt', add_special_tokens=False).values()
        self.response_ids, self.response_mask = self.qwen_tokenizer(self.args['instruction_text'][1],
                                                                     truncation=True, padding=False,
                                                                     return_tensors='pt', add_special_tokens=False).values()
        self.emb_dim = self.qwen_model.config.hidden_size

        # initialize CF embedding
        padding_embed = torch.zeros(1, self.input_dim)
        concat_embed = torch.cat([padding_embed, self.args['input_embeds']], dim=0)
        self.input_embeds = nn.ModuleList([nn.Embedding.from_pretrained(concat_embed, freeze=True)])

        # initialize CF projector (cf token dim -> LLM token dim)
        self.input_proj = nn.Linear(self.input_dim, self.qwen_model.config.hidden_size)

        # initialize guiding query 
        self.query = nn.Embedding(self.n_query, self.qwen_model.config.hidden_size)
        self.query_mask = torch.ones(self.n_query, dtype=self.response_mask.dtype)

        # initialize semantic tokenizer
        if self.n_sem:
            self.tokenizer = AE(in_dim=args["in_dim"],
                                e_dim=self.qwen_model.config.hidden_size * (self.n_query-self.n_cf),
                                layers=args["layers"],
                                dropout_prob=args["dropout_prob"],
                                bn=args["bn"],
                                loss_type=args["loss_type"],
                                item_feature=args["item_feature"]
                                )
        
    def get_tokenizer(self):
        return self.qwen_tokenizer
    
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
        # initialize
        item_sem_token = torch.zeros(item_id.shape[0], item_id.shape[1], self.n_sem, self.emb_dim, dtype=self.all_sem.dtype).cuda() # (bs, seq_len, n_sem, dim)
        
        non_zero_mask = self.batch_nonzero_mask # (bs, seq_len)
        non_zero_ids = item_id[non_zero_mask] #  (n_nonzero, )

        item_sem_token[non_zero_mask] = self.all_sem.transpose(0,1)[non_zero_ids-1] # (n_nonzero, n_sem, dim)

        sem_input = self.tokenizer.item_feature[non_zero_ids-1] # (n_nonzero, 4096)
        recon_emb = all_recon_emb[non_zero_ids-1] # (n_nonzero, 4096)

        sem_emb = self.all_sem.transpose(0,1)[non_zero_ids-1] # (n_nonzero, n_sem, dim)
        return sem_input, sem_emb, recon_emb, item_sem_token
        
    def predict(self, inputs, inputs_mask, inference=False):
        self.batch_nonzero_mask = inputs!=0

        if not inference and self.n_sem: 
            self.recon_all = self.tokenize_all()

        x, unique_sem_emb, x_hat, item_sem_token = None, None, None, None

        # 1. get semantic tokens
        if self.n_sem:
            # put current batch item into the AE to get the sem emb
            x, unique_sem_emb, x_hat, item_sem_token = self.get_batch_sem(inputs, self.recon_all)
                                                        # x: (n_nonzero, input_dim)
                                                        # unique_sem_emb (n_nonzero, n_sem, dim)
                                                        # x_hat (n_nonzero, input_dim)
                                                        # item_sem_token (bs, seq_len, n_sem, dim)

        bs = inputs.shape[0]
        instruct_embeds = self.qwen_model.model.embed_tokens(self.instruct_ids.cuda()).expand(bs, -1, -1)
        response_embeds = self.qwen_model.model.embed_tokens(self.response_ids.cuda()).expand(bs, -1, -1)
        instruct_mask = self.instruct_mask.cuda().expand(bs, -1)
        response_mask = self.response_mask.cuda().expand(bs, -1)
        query_mask = self.query_mask.cuda().expand(bs, -1)

        # 2. get cf tokens
        inputs = self.input_proj(self.input_embeds[0](inputs))

        # 3. concat cf emb and sem emb
        inputs = inputs.unsqueeze(-2) # (bs, seq_len, n_cf, dim)
        inputs = torch.cat([inputs, item_sem_token], dim=2) if self.n_sem else inputs # (bs, seq_len, n_sem+n_cf(1), dim) 
        inputs = inputs.view(bs, -1, self.qwen_model.config.hidden_size)

        history_len = inputs.shape[1]

        # concate the query with the original input
        indices = torch.arange(0,self.n_query, device=inputs.device) 
        querys = self.query(indices).unsqueeze(0).repeat(inputs.shape[0],1,1) # shape (bs, n_query, dim)

        inputs = torch.cat([instruct_embeds, inputs, response_embeds, querys], dim=1) 
        attention_mask = torch.cat([instruct_mask, inputs_mask, response_mask, query_mask], dim=1)

        assert attention_mask.size()[0] == inputs.size()[0] and attention_mask.size()[1] == inputs.size()[1]

        history_idx_st = instruct_embeds.shape[1]
        answer_idx_st = instruct_embeds.shape[1] + history_len

        # 4. query-guided simultaneous decoding (qwen forward)
        outputs = self.qwen_model(inputs_embeds=inputs, attention_mask=attention_mask, return_dict=True, history_idx_st=history_idx_st, answer_idx_st=answer_idx_st)
        
        # 5. grounding
        query_output = outputs.last_hidden_state[:, -self.n_query:, :] # (bs, n_query, dim)
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
            weight = self.beta[:, None, None].to(self.qwen_model.device)
            score = torch.sum(output * weight, dim=0).squeeze()
            return outputs, score, x, unique_sem_emb, x_hat

    def forward(self, inputs, inputs_mask, labels):
        _, outputs, scores, x, _, x_hat = self.predict(inputs, inputs_mask) # prediction (bs, n_query, dim)

        loss = None
        logits = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # generative recommendation loss
            scores = torch.cat([_ for _ in scores],dim = 0)
            all_item_gold = labels.squeeze().repeat(1,self.n_query).squeeze() #(n_query*bs)
            loss = loss_fct(scores, all_item_gold)

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
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
