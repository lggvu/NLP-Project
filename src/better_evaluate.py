from itertools import chain
from fairseq.models.transformer import TransformerModel
import time
from sacremoses import MosesTokenizer, MosesDetokenizer


mt = MosesTokenizer(lang='en')
md = MosesDetokenizer(lang='en')

def translate(sentence,model):
    # sentence = sentence.lower()
    # sentence = mt.tokenize(sentence, return_str=True)
    print(sentence)
    output = model.translate(sentence)
    return md.detokenize(output.split())

def read_file(path):
    data = []
    with open(path,'r') as f:
        for line in f:
            data.append(line.replace('\n',""))
    return data

def gen_output(path_in,path_out, model):
    source_path = path_in
    out_path = path_out
    source = read_file(source_path)
    predict = []
    time_start = time.time()
    with open(out_path,"w") as f:
        for sen in source:
            output = translate(sen, model)
            print(output)
            predict.append(output)
            time_now = time.time()
            print("time spend : ",time_now - time_start)
            time_start = time_now
            f.write(output+'\n')


model = TransformerModel.from_pretrained(
  '/home2/khanhnd/transformer-mt/fairseq/checkpoints_en_vi_parallel_backtranslation',
  checkpoint_file='checkpoint_last.pt',
  data_name_or_path="/home2/khanhnd/fairseq/data-bin/phomt-backtranslation_en_vi",
  bpe='subword_nmt',
  tokenize='moses',
  bpe_codes='/home2/khanhnd/fairseq/data-bin/phomt-backtranslation_en_vi/code'
)
model.cuda()
model.eval()
gen_output("/home2/khanhnd/fairseq/examples/translation/pho-mt-backtranslation/test.en","/home2/khanhnd/transformer-mt/output_txt.txt", model)