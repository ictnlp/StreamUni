
import pdb
import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    BatchFeature,
    Trainer,
    TrainingArguments,
)
import soundfile

ANSWER_SUFFIX = "<|end|><|endoftext|>"
_IGNORE_INDEX = -100
_TRAIN_SIZE = 50000
_EVAL_SIZE = 200

class CoTSTDataset(torch.utils.data.Dataset):
    def __init__(self, processor, data_dir, model=None):
        self.data = json.load(open(data_dir, 'r'))
        self.processor = processor
        self.model = model
    def filter_data(self):
        for i in range(len(self.data)):
            if self.data[i]['duration'] > 30:
                self.data.pop(i)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data = self.data[idx]
        user_message = {
            'role': 'user',
            'content': '<|audio_1|>' + data['instruction'],
        }
        
        prompt = self.processor.tokenizer.apply_chat_template(
            [user_message], tokenize=False, add_generation_prompt=True
        )
        
        audio, sr = soundfile.read(data["audio"])
        
        inputs = self.processor(text=prompt, audios=[(audio, sr)], return_tensors='pt')
        
        answer = f"{data['asr']} <sep> {data['translation']}{ANSWER_SUFFIX}"
        answer_ids = self.processor.tokenizer(answer, return_tensors='pt').input_ids
        if self.model.training:
            input_ids = torch.cat([inputs.input_ids, answer_ids], dim=1)
            labels = torch.full_like(input_ids, _IGNORE_INDEX)
            labels[:, -answer_ids.shape[1] :] = answer_ids
        else:
            input_ids = inputs.input_ids
            labels = answer_ids

        return {
            'input_ids': input_ids,
            'labels': labels,
            'input_audio_embeds': inputs.input_audio_embeds,
            'audio_embed_sizes': inputs.audio_embed_sizes,
        }

def pad_sequence(sequences, padding_side='right', padding_value=0):
    """
    Pad a list of sequences to the same length.
    sequences: list of tensors in [seq_len, *] shape
    """
    assert padding_side in ['right', 'left']
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    max_len = max(len(seq) for seq in sequences)
    batch_size = len(sequences)
    output = sequences[0].new_full((batch_size, max_len) + trailing_dims, padding_value)
    for i, seq in enumerate(sequences):
        length = seq.size(0)
        if padding_side == 'right':
            output.data[i, :length] = seq
        else:
            output.data[i, -length:] = seq
    return output

def cat_with_pad(tensors, dim, padding_value=0):
    """
    cat along dim, while pad to max for all other dims
    """
    ndim = tensors[0].dim()
    assert all(
        t.dim() == ndim for t in tensors[1:]
    ), 'All tensors must have the same number of dimensions'

    out_size = [max(t.shape[i] for t in tensors) for i in range(ndim)]
    out_size[dim] = sum(t.shape[dim] for t in tensors)
    output = tensors[0].new_full(out_size, padding_value)

    index = 0
    for t in tensors:
        slices = [slice(0, t.shape[d]) for d in range(ndim)]
        slices[dim] = slice(index, index + t.shape[dim])
        output[slices] = t
        index += t.shape[dim]

    return output

def covost_collate_fn(batch):
    input_ids_list = []
    labels_list = []
    input_audio_embeds_list = []
    audio_embed_sizes_list = []
    audio_attention_mask_list = []
    for inputs in batch:
        input_ids_list.append(inputs['input_ids'][0])
        labels_list.append(inputs['labels'][0])
        input_audio_embeds_list.append(inputs['input_audio_embeds'])
        audio_embed_sizes_list.append(inputs['audio_embed_sizes'])
        audio_attention_mask_list.append(
            inputs['input_audio_embeds'].new_full((inputs['input_audio_embeds'].size(1),), True, dtype=torch.bool)
        )

    try:
        input_ids = pad_sequence(input_ids_list, padding_side='left', padding_value=0)
        labels = pad_sequence(labels_list, padding_side='left', padding_value=0)
        audio_attention_mask = (
            pad_sequence(audio_attention_mask_list, padding_side='right', padding_value=False)
            if len(audio_attention_mask_list) > 1
            else None
        )
    except Exception as e:
        print(e)
        print(input_ids_list)
        print(labels_list)
        raise
    attention_mask = (input_ids != 0).long()
    input_audio_embeds = cat_with_pad(input_audio_embeds_list, dim=0)
    audio_embed_sizes = torch.cat(audio_embed_sizes_list)
    
    return BatchFeature(
        {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
            'input_audio_embeds': input_audio_embeds,
            'audio_embed_sizes': audio_embed_sizes,
            'audio_attention_mask': audio_attention_mask,
            'input_mode': 2,  # speech mode
        }
    )

def create_model(model_name_or_path, use_flash_attention=False):
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16 if use_flash_attention else torch.float32,
        _attn_implementation='flash_attention_2' if use_flash_attention else 'sdpa',
        trust_remote_code=True,
    )
    return model

def add_args(parser):
    parser.add_argument(
        '--model_name_or_path',
        type=str,
        default='microsoft/Phi-4-multimodal-instruct',
        help='Model name or path to load from',
    )
    parser.add_argument(
        "--voice_dir",
        type=str,
        default="CommonVoice/EN",
        help="Unzipped Common Voice Audio dataset directory",
    )
    parser.add_argument('--use_flash_attention', action='store_true', help='Use Flash Attention')
    parser.add_argument('--output_dir', type=str, default='./output/', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--batch_size_per_gpu', type=int, default=16, help='Batch size per GPU')
    parser.add_argument('--num_train_epochs', type=int, default=1, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=4.0e-5, help='Learning rate')
    parser.add_argument('--wd', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--no-tqdm', dest='tqdm', action='store_false', help='Disable tqdm')
    parser.add_argument('--deepspeed', type=str, help='Path to deepspeed config file')
    parser.add_argument('--local_rank', type=int, default=-1, help='Local rank for distributed training')
    return parser

def set_specific_params_requires_grad(model, keywords=('speech', 'audio')):
    """
    
    Args:
        model: 
        keywords: 
    
    Returns:
        int: 
    """
    enabled_count = 0
    total_count = 0
    
    for param in model.parameters():
        param.requires_grad = False
        total_count += 1
    
    for name, param in model.named_parameters():

        if any(keyword in name.lower() for keyword in keywords):
            param.requires_grad = True
            enabled_count += 1

    
    print(f"\nTotal Params: {total_count}")
    print(f"Enabled Params: {enabled_count}")
    print(f"Freezed Params: {total_count - enabled_count}")
    
    return enabled_count

def main():
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    args = parser.parse_args()
    print('args parsed')
    
    # initialize
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
    
    torch.cuda.manual_seed(42)

    if args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir, exist_ok=True)
    
    processor = AutoProcessor.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
    )
    
    model = create_model(
        args.model_name_or_path,
        use_flash_attention=args.use_flash_attention,
    )
    model.set_lora_adapter('speech')
    
    device = torch.device("cuda", args.local_rank) if args.local_rank != -1 else torch.device("cuda")
    model = model.to(device)
    
    if args.local_rank in [-1, 0]:
        print(f"Process rank: {args.local_rank}, device: {device}, n_gpu: {torch.cuda.device_count()}")
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")
    
    set_specific_params_requires_grad(model, keywords=('speech', 'audio'))
    model.model.embed_tokens_extend.audio_embed.requires_grad = False
    
    train_dataset = CoTSTDataset(
        processor,
        data_dir=args.voice_dir,
        model=model
    )

    if args.use_flash_attention:
        fp16 = False
        bf16 = True
    else:
        fp16 = True
        bf16 = False

    training_args = TrainingArguments(
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size_per_gpu,
        gradient_accumulation_steps=args.batch_size // args.batch_size_per_gpu,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        optim='adamw_torch',
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_epsilon=1e-7,
        learning_rate=args.learning_rate,
        weight_decay=args.wd,
        max_grad_norm=1.0,
        lr_scheduler_type='linear',
        warmup_steps=50,
        logging_steps=10,
        output_dir=args.output_dir,
        save_strategy='steps',
        save_total_limit=10,
        save_steps=100,
        bf16=bf16,
        fp16=fp16,
        remove_unused_columns=False,
        report_to='none',
        disable_tqdm=not args.tqdm,
        dataloader_num_workers=4,
        deepspeed=args.deepspeed,
        local_rank=args.local_rank,
    )
    
    out_path = Path(training_args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print('Begin Fine-tuning')

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=covost_collate_fn,
        train_dataset=train_dataset,
    )

    trainer.train()
    if args.local_rank == 0:
        trainer.save_model()
        processor.save_pretrained(training_args.output_dir)

    print('End Fine-tuning')

if __name__ == '__main__':
    main()
