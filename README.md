# SETRec
This is the pytorch implementation of our paper 
> [Order-agnostic Identifier for Large Language Model-based Generative Recommendation](https://arxiv.org/pdf/2502.10833)

> [Xinyu Lin](https://scholar.google.com/citations?view_op=list_works&hl=en&hl=en&user=0O_bs3UAAAAJ&sortby=pubdate), [Haihan Shi](https://data-science.ustc.edu.cn/_upload/tpl/15/04/5380/template5380/people.html), [Wenjie Wang](https://wenjiewwj.github.io/), [Fuli Feng](https://fulifeng.github.io/), [Qifan Wang](https://wqfcr.github.io/), [See-Kiong Ng](https://www.comp.nus.edu.sg/~ngsk/), [Tat-Seng Chua](https://www.chuatatseng.com/)

## Environment
- Anaconda 3
- Python 3.8.0
- pytorch 2.0.1
- transformers 4.41.0

## Usage

### Data
The experimental data are in './data' folder, including Toys, Beauty, Sports, and Steam.

#### :arrow_forward: CF Tokenizer
A pre-trained CF tokenizer is utilized in SETRec to obtain the CF token. You can either

- **Approach 1** - train your own CF tokenizer (e.g., SASRec)

- **Approach 2** - directly use our provided CF tokens (e.g., ``SASRec_item_embed.pkl`` under each specific dataset folder)

Skip this step if you directly use our provided ones (*Approach 2*).

#### :arrow_forward: Semantic Tokenizer

A semantic tokenizer will be created and trained during the SETRec training. But before that, we need **item semantic representation** for tokenization. To obtain the semantic representation, you can either

- **Approach 1** - extract manually
The item semantic representation is extracted by pre-trained language model (e.g., T5 or Qwen). We provide the scripts for extracting semantic representation in "./data" folder, ``extract_item_semantic_rep.py``. 

- **Approach 2** - use our provided semantic representations under each specific dataset folder.

Skip this step if you directly use our provided ones (*Approach 2*).

### :red_circle: Training 

First direct to the './code' folder
```
cd code
```

For T5 backend, run the command

```
bash scripts/train_t5.sh <dataset> <lr> <n_sem> <alpha>
```

For Qwen backend, run the command

```
bash scripts/train_qwen.sh <dataset> <lr> <n_sem> <alpha>
```

- The log file will be in the './log/' folder. 
- The explanation of hyper-parameters can be found in './code/parse_utils.py'. 

### :large_blue_circle: Inference
Get the results of SETRec by running inference.py:

- Infer with T5 backend
```
bash scripts/inference_t5.sh <dataset> <n_sem> <ckpt_path>
```
- Infer with Qwen backend (Please stay tuned for the inference script for Qwen. Currently you can get the evaluation results at the end of training.)
```
bash scripts/inference_qwen.sh <dataset> <n_sem> <ckpt_path>
```


### :white_circle: Examples

1. Train SETRec (T5) on Toys dataset 

```
cd ./code
bash scripts/train_t5.sh toys 1e-3 4 0.7
```

2. Inference

```
cd ./code
sh scripts/inference_t5.sh toys 4 <ckpt_path> 
```

## Citation
If you find our work is useful for your research, please consider citing: 
```
@inproceedings{lin2025order,
  title={Order-agnostic Identifier for Large Language Model-based Generative Recommendation},
  author={Lin, Xinyu and Shi, Haihan and Wang, Wenjie and Feng, Fuli and Wang, Qifan and Ng, See-Kiong and Chua, Tat-Seng},
  booktitle={SIGIR},
  year={2025}
}
```

## License

NUS © [NExT++](https://www.nextcenter.org/)