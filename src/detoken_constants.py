import torch

# Path or parameters for data
DATA_DIR = 'sampled_data'
SP_DIR = f'{DATA_DIR}/sp'
SRC_DIR = 'src'
TRG_DIR = 'trg'
SRC_RAW_DATA_NAME = 'raw_data.src'
TRG_RAW_DATA_NAME = 'raw_data.trg'
TRAIN_NAME = 'clean_train.txt'
VALID_NAME = 'clean_dev.txt'
TEST_NAME = 'test.txt'

# Parameters for sentencepiece tokenizer
pad_id = 0
sos_id = 1
eos_id = 2
unk_id = 3
src_model_prefix = 'src_sp'
trg_model_prefix = 'trg_sp'
sp_vocab_size = 16000
character_coverage = 1.0
model_type = 'BPE'

# Parameters for Transformer & training
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
learning_rate = 5e-4
batch_size = 8
seq_len = 200
num_heads = 8
num_layers = 6
d_model = 1024
d_ff = 4096
d_k = d_model // num_heads
drop_out_rate = 0.1
num_epochs = 15
beam_size = 8
ckpt_dir = '/home2/khanhnd/transformer-mt/Machine_Translation_DATN/checkpoints_bt'
