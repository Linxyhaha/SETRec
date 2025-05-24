def train_args(parser):
    parser.add_argument("--base_model", type=str, default="")
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--cache_dir", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--seed", type=int, default=2024)

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--micro_batch_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--cutoff_len", type=int, default=4096)
    parser.add_argument("--val_set_size", type=int, default=2000)
    parser.add_argument("--lr_scheduler", type=str, default="cosine")
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--group_by_length", action="store_true")

    parser.add_argument("--resume_from_checkpoint", type=str, default=None) # either training checkpoint or final adapter
    parser.add_argument("--prompt_template_name", type=str, default="template")

    parser.add_argument("--ckpt_dir", default=None)
    parser.add_argument("--model_class", type=str, default="Qwen4Rec", help="which model is using")

    return parser

def identifier_args(parser):
    parser.add_argument("--n_query", type=int, default=2)
    parser.add_argument("--n_cf", type=int, default=1)
    parser.add_argument("--n_sem", type=int, default=1)
    parser.add_argument("--tokenizer", type=str, default="AE")
    parser.add_argument("--alpha", type=float, default=0.5, help="coefficient of tokenizer loss during training")
    return parser

def wandb_args(parser):
    parser.add_argument("--wandb_project", type=str, default="")
    parser.add_argument("--wandb_run_name", type=str, default="")
    parser.add_argument("--wandb_watch", type=str, default="") # options: false | gradients | all
    parser.add_argument("--wandb_log_model", type=str, default="") # options: false | true
    return parser

def parse_AE_args(parser):
    parser.add_argument("--loss_type", type=str, default="mse", help="loss_type")
    parser.add_argument('--layers', type=int, nargs='+', default=[512,256,128], help='hidden sizes of every layer')
    parser.add_argument('--dropout_prob', type=float, default=0)
    parser.add_argument("--bn", type=bool, default=False, help="use bn or not")
    parser.add_argument("--sem_encoder", type=str, default="t5", help="encoder of semantic embedding, e.g., qwen, t5")
    return parser

def lora_args(parser):
    parser.add_argument("--lora_target_modules", type=str, nargs="+", default=["q_proj", "v_proj"],help="List of module names to apply LoRA")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.02)
    return parser