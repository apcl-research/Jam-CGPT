import time

#out_dir = 'out-owt-gpt2mini'
out_dir = 'out-jam52m' #'out-funcom_raw_scratch_1b_local'
eval_interval = 1000
eval_iters = 100
wandb_log = True # feel free to turn on
wandb_project = 'jam_cgpt_38m_raw'
wandb_run_name = 'jam_cgpt_38m' #+ str(time.time())

dataset = 'jam_jm52m'
init_from = 'scratch'

# only save checkpoints if the validation loss improves
always_save_checkpoint = False

n_layer = 4
n_head = 4
n_embd = 512

block_size = 256


batch_size = 32 #16
gradient_accumulation_steps = 4
max_iters = 268000 * 3

learning_rate = 3e-5
weight_decay = 1e-5
#decay_lr = False

