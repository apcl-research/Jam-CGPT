import time

#out_dir = 'out-owt-gpt2mini'
out_dir = 'out-jam-cgpt'
eval_interval = 100
eval_iters = 80
wandb_log = True
wandb_project = 'jam-cgpt'
wandb_run_name = 'jam-cgpt-model110m-data620k'

dataset = 'jam_cgpt_620k'
init_from = 'resume'

# only save checkpoints if the validation loss improves
always_save_checkpoint = True 

dropout = 0.2


# jam-cgpt 620k has 118,941,493 tokens

# model iters
# 38m parameters model has 757,000 iters
# 110m parameters model has 762,000 iters
# 350m parameters model has 272,000 iters

block_size = 256

batch_size = 4 #16
gradient_accumulation_steps = 32
#max_iters = 5600 # 172394 training samples

max_iters = 762000 + 3630 * 3

# finetune at constant LR
learning_rate = 3e-5
decay_lr = False
