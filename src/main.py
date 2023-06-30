from tqdm import tqdm
from detoken_constants import *
from custom_data import *
from transformer import *
from data_structure import *
from torch import nn

import torch
import sys, os
import numpy as np
import argparse
import datetime
import copy
import heapq
import sentencepiece as spm
import json

import wandb

# from torch.utils.tensorboard import SummaryWriter

class Manager():
    def __init__(self, is_train=True, ckpt_name=None):
        # Load vocabs
        print("Loading vocabs...")
        wandb.init(project='lggvu-nlp')
        # Set the configuration values
        wandb.config.update({
            'num_epochs': num_epochs,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            # Add other hyperparameters here
        })

        self.src_i2w = {}
        self.trg_i2w = {}
        if os.path.exists("tokenized_data.json"):
            self.load_tokenized_data()
        else:
            # Load vocabs from original code
            print("JSON file not found. Loading vocabs from original code...")
            with open(f"{SP_DIR}/{src_model_prefix}.vocab") as f:
                lines = f.readlines()
            for i, line in enumerate(lines):
                word = line.strip().split('\t')[0]
                self.src_i2w[i] = word

            with open(f"{SP_DIR}/{trg_model_prefix}.vocab") as f:
                lines = f.readlines()
            for i, line in enumerate(lines):
                word = line.strip().split('\t')[0]
                self.trg_i2w[i] = word

            print(f"The size of src vocab is {len(self.src_i2w)} and that of trg vocab is {len(self.trg_i2w)}.")

        # Load Transformer model & Adam optimizer
        print("Loading Transformer model & Adam optimizer...")
        self.model = Transformer(src_vocab_size=16000, trg_vocab_size=16000).to(device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.best_loss = sys.float_info.max
        # print(self.model.state_dict().keys())

        # print(self.model.state_dict.keys())
        if ckpt_name is not None:
            print(f"{ckpt_dir}/{ckpt_name}")
            assert os.path.exists(f"{ckpt_dir}/{ckpt_name}"), f"There is no checkpoint named {ckpt_name}."

            print("Loading checkpoint...")
            ckpt = torch.load(f"{ckpt_dir}/{ckpt_name}")
            state_dict = ckpt["model"]

            # Remove "encoder.version" and "decoder.version" from the state_dict
            state_dict.pop("encoder.version", None)
            state_dict.pop("decoder.version", None)

            for key in list(state_dict.keys()):
                state_dict[key.replace("decoder.embed_positions._float_tensor", "decoder.embed_positions.embed_positions"). \
                    replace("encoder.embed_positions._float_tensor", "encoder.embed_positions.embed_positions")] = state_dict.pop(key)

            # print(self.model.state_dict().keys())

            self.model.load_state_dict(ckpt["model"])

            # self.model.load_state_dict(checkpoint)
            # self.optim.load_state_dict(checkpoint['optim_state_dict'])
            # self.best_loss = checkpoint['loss']
        else:
            print("Initializing the model...")
            for p in self.model.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

        if is_train:
            # Load loss function
            print("Loading loss function...")
            self.criterion = nn.NLLLoss()

            # Load dataloaders
            print("Loading dataloaders...")
            self.train_loader = get_data_loader(TRAIN_NAME)
            self.valid_loader = get_data_loader(VALID_NAME)

        print("Setting finished.")

    def train(self):
        print("Training starts.")
        # writer = SummaryWriter("logs")

        # Save tokenized data into JSON
        tokenized_data = {
            "src_i2w": self.src_i2w,
            "trg_i2w": self.trg_i2w
        }
        print("Saving tokenized data to json...")
        with open("tokenized_data.json", "w") as f:
            json.dump(tokenized_data, f)

        for epoch in tqdm(range(1, num_epochs + 1), desc="Epoch"):
            self.model.train()
            
            train_losses = []
            start_time = datetime.datetime.now()

            for i, batch in tqdm(enumerate(self.train_loader), desc="Batch", total=len(self.train_loader)):
                wandb.log({'batch': i})  # Log the batch index

                src_input, trg_input, trg_output = batch
                src_input, trg_input, trg_output = src_input.to(device), trg_input.to(device), trg_output.to(device)

                e_mask, d_mask = self.make_mask(src_input, trg_input)

                output = self.model(src_input, trg_input, e_mask, d_mask) # (B, L, vocab_size)

                trg_output_shape = trg_output.shape
                self.optim.zero_grad()
                loss = self.criterion(
                    output.view(-1, sp_vocab_size),
                    trg_output.view(trg_output_shape[0] * trg_output_shape[1])
                )

                loss.backward()
                self.optim.step()

                train_losses.append(loss.item())
                # writer.add_scalar("Train Loss", loss.item(), epoch*len(self.train_loader) + i)
                wandb.log({'train_loss': loss.item(), 'epoch': epoch, 'batch': i})

                del src_input, trg_input, trg_output, e_mask, d_mask, output
                torch.cuda.empty_cache()

                # Inside the loop, after updating the model parameters, save the checkpoint
                save_interval=100_000
                if i % save_interval == 0:
                    state_dict = {
                        'model_state_dict': self.model.state_dict(),
                        'optim_state_dict': self.optim.state_dict(),
                        'loss': self.best_loss
                    }
                    torch.save(state_dict, f"{ckpt_dir}/epoch_{epoch}_iter_{i}_ckpt.tar")
                    print(f"Checkpoint saved at {ckpt_dir}/epoch_{epoch}_iter_{i}_ckpt.tar")

            end_time = datetime.datetime.now()
            training_time = end_time - start_time
            seconds = training_time.seconds
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            seconds = seconds % 60

            mean_train_loss = np.mean(train_losses)
            # writer.add_scalar("Mean Train Loss", mean_train_loss, epoch)
            wandb.log({'mean_train_loss': mean_train_loss, 'epoch': epoch})


            print(f"#################### Epoch: {epoch} ####################")
            print(f"Train loss: {mean_train_loss} || One epoch training time: {hours}hrs {minutes}mins {seconds}secs")

            valid_loss, valid_time = self.validation()
            # writer.add_scalar("Valid Loss", valid_loss, epoch)
            # writer.add_scalar("Valid Time", valid_time, epoch)
            if valid_loss < self.best_loss:
                if not os.path.exists(ckpt_dir):
                    os.mkdir(ckpt_dir)
                    
                self.best_loss = valid_loss
                state_dict = {
                    'model_state_dict': self.model.state_dict(),
                    'optim_state_dict': self.optim.state_dict(),
                    'loss': self.best_loss
                }
                torch.save(state_dict, f"{ckpt_dir}/best_ckpt.tar")
                print(f"***** Current best checkpoint is saved. *****")


            print(f"Best valid loss: {self.best_loss}")
            # writer.add_scalar("Best Valid Loss", self.best_loss, epoch)
            wandb.log({'best_valid_loss': self.best_loss, 'epoch': epoch})

            print(f"Valid loss: {valid_loss} || One epoch training time: {valid_time}")

        print(f"Training finished!")
        # Close Writer
        # writer.close()
        wandb.finish()


    def load_tokenized_data(self):
        # Load tokenized data from JSON
        print("Loading tokenized data from JSON...")
        with open("tokenized_data.json", "r") as f:
            tokenized_data = json.load(f)
        self.src_i2w = tokenized_data["src_i2w"]
        self.trg_i2w = tokenized_data["trg_i2w"]
        
    def validation(self):
        print("Validation processing...")
        self.model.eval()
        
        valid_losses = []
        start_time = datetime.datetime.now()

        with torch.no_grad():
            for i, batch in tqdm(enumerate(self.valid_loader)):
                src_input, trg_input, trg_output = batch
                src_input, trg_input, trg_output = src_input.to(device), trg_input.to(device), trg_output.to(device)

                e_mask, d_mask = self.make_mask(src_input, trg_input)

                output = self.model(src_input, trg_input, e_mask, d_mask) # (B, L, vocab_size)

                trg_output_shape = trg_output.shape
                loss = self.criterion(
                    output.view(-1, sp_vocab_size),
                    trg_output.view(trg_output_shape[0] * trg_output_shape[1])
                )

                valid_losses.append(loss.item())

                del src_input, trg_input, trg_output, e_mask, d_mask, output
                torch.cuda.empty_cache()

        end_time = datetime.datetime.now()
        validation_time = end_time - start_time
        seconds = validation_time.seconds
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        
        mean_valid_loss = np.mean(valid_losses)
        
        return mean_valid_loss, f"{hours}hrs {minutes}mins {seconds}secs"

    def inference(self, input_sentence, method):
        print("Inference starts.")
        self.model.eval()

        print("Loading sentencepiece tokenizer...")
        src_sp = spm.SentencePieceProcessor()
        trg_sp = spm.SentencePieceProcessor()
        src_sp.Load(f"{SP_DIR}/{src_model_prefix}.model")
        trg_sp.Load(f"{SP_DIR}/{trg_model_prefix}.model")

        print("Preprocessing input sentence...")
        tokenized = src_sp.EncodeAsIds(input_sentence)
        src_data = torch.LongTensor(pad_or_truncate(tokenized)).unsqueeze(0).to(device) # (1, L)
        e_mask = (src_data != pad_id).unsqueeze(1).to(device) # (1, 1, L)

        start_time = datetime.datetime.now()

        print("Encoding input sentence...")
        src_data = self.model.encoder.embed_tokens(src_data)
        src_data = self.model.positional_encoder(src_data)
        e_output = self.model.encoder(src_data, e_mask) # (1, L, d_model)

        if method == 'greedy':
            print("Greedy decoding selected.")
            result = self.greedy_search(e_output, e_mask, trg_sp)
        elif method == 'beam':
            print("Beam search selected.")
            result = self.beam_search(e_output, e_mask, trg_sp)

        end_time = datetime.datetime.now()

        total_inference_time = end_time - start_time
        seconds = total_inference_time.seconds
        minutes = seconds // 60
        seconds = seconds % 60

        # print(f"Input: {input_sentence}")
        # print(f"Result: {result}")
        # print(f"Inference finished! || Total inference time: {minutes}mins {seconds}secs")
        print(result)
        
    def greedy_search(self, e_output, e_mask, trg_sp):
        last_words = torch.LongTensor([pad_id] * seq_len).to(device) # (L)
        last_words[0] = sos_id # (L)
        cur_len = 1

        for i in range(seq_len):
            d_mask = (last_words.unsqueeze(0) != pad_id).unsqueeze(1).to(device) # (1, 1, L)
            nopeak_mask = torch.ones([1, seq_len, seq_len], dtype=torch.bool).to(device)  # (1, L, L)
            nopeak_mask = torch.tril(nopeak_mask)  # (1, L, L) to triangular shape
            d_mask = d_mask & nopeak_mask  # (1, L, L) padding false

            trg_embedded = self.model.encoder.embed_tokens(last_words.unsqueeze(0))
            trg_positional_encoded = self.model.encoder.embed_positions(trg_embedded)
            decoder_output = self.model.decoder(
                trg_positional_encoded,
                e_output,
                e_mask,
                d_mask
            ) # (1, L, d_model)

            output = self.model.softmax(
                self.model.output_linear(decoder_output)
            ) # (1, L, trg_vocab_size)

            output = torch.argmax(output, dim=-1) # (1, L)
            last_word_id = output[0][i].item()
            
            if i < seq_len-1:
                last_words[i+1] = last_word_id
                cur_len += 1
            
            if last_word_id == eos_id:
                break

        if last_words[-1].item() == pad_id:
            decoded_output = last_words[1:cur_len].tolist()
        else:
            decoded_output = last_words[1:].tolist()
        decoded_output = trg_sp.decode_ids(decoded_output)
        
        return decoded_output
    
    def beam_search(self, e_output, e_mask, trg_sp):
        cur_queue = PriorityQueue()
        for k in range(beam_size):
            cur_queue.put(BeamNode(sos_id, -0.0, [sos_id]))
        
        finished_count = 0
        
        for pos in range(seq_len):
            new_queue = PriorityQueue()
            for k in range(beam_size):
                node = cur_queue.get()
                if node.is_finished:
                    new_queue.put(node)
                else:
                    trg_input = torch.LongTensor(node.decoded + [pad_id] * (seq_len - len(node.decoded))).to(device) # (L)
                    d_mask = (trg_input.unsqueeze(0) != pad_id).unsqueeze(1).to(device) # (1, 1, L)
                    nopeak_mask = torch.ones([1, seq_len, seq_len], dtype=torch.bool).to(device)
                    nopeak_mask = torch.tril(nopeak_mask) # (1, L, L) to triangular shape
                    d_mask = d_mask & nopeak_mask # (1, L, L) padding false
                    
                    trg_embedded = self.model.encoder.embed_tokens(trg_input.unsqueeze(0))
                    trg_positional_encoded = self.model.encoder.embed_positions(trg_embedded)
                    decoder_output = self.model.decoder(
                        trg_positional_encoded,
                        e_output,
                        e_mask,
                        d_mask
                    ) # (1, L, d_model)

                    output = self.model.softmax(
                        self.model.decoder.output_projection(decoder_output)
                    ) # (1, L, trg_vocab_size)
                    
                    output = torch.topk(output[0][pos], dim=-1, k=beam_size)
                    last_word_ids = output.indices.tolist() # (k)
                    last_word_prob = output.values.tolist() # (k)
                    
                    for i, idx in enumerate(last_word_ids):
                        new_node = BeamNode(idx, -(-node.prob + last_word_prob[i]), node.decoded + [idx])
                        if idx == eos_id:
                            new_node.prob = new_node.prob / float(len(new_node.decoded))
                            new_node.is_finished = True
                            finished_count += 1
                        new_queue.put(new_node)
            
            cur_queue = copy.deepcopy(new_queue)
            
            if finished_count == beam_size:
                break
        
        decoded_output = cur_queue.get().decoded
        
        if decoded_output[-1] == eos_id:
            decoded_output = decoded_output[1:-1]
        else:
            decoded_output = decoded_output[1:]
            
        return trg_sp.decode_ids(decoded_output)
        

    def make_mask(self, src_input, trg_input):
        e_mask = (src_input != pad_id).unsqueeze(1)  # (B, 1, L)
        d_mask = (trg_input != pad_id).unsqueeze(1)  # (B, 1, L)

        nopeak_mask = torch.ones([1, seq_len, seq_len], dtype=torch.bool)  # (1, L, L)
        nopeak_mask = torch.tril(nopeak_mask).to(device)  # (1, L, L) to triangular shape
        d_mask = d_mask & nopeak_mask  # (B, L, L) padding false

        return e_mask, d_mask


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True, help="train or inference?")
    parser.add_argument('--ckpt_name', required=False, help="best checkpoint file")
    parser.add_argument('--input', type=str, required=False, help="input sentence when inferencing")
    parser.add_argument('--decode', type=str, required=True, default="greedy", help="greedy or beam?")

    args = parser.parse_args()

    if args.mode == 'train':
        if args.ckpt_name is not None:
            manager = Manager(is_train=True, ckpt_name=args.ckpt_name)
        else:
            manager = Manager(is_train=True)

        manager.train()
    elif args.mode == 'inference':
        assert args.ckpt_name is not None, "Please specify the model file name you want to use."
        assert args.input is not None, "Please specify the input sentence to translate."
        assert args.decode == 'greedy' or args.decode =='beam', "Please specify correct decoding method, either 'greedy' or 'beam'."
       
        manager = Manager(is_train=False, ckpt_name=args.ckpt_name)
        manager.inference(args.input, args.decode)

    else:
        print("Please specify mode argument either with 'train' or 'inference'.")
