import os
import requests
import torch
from PIL import Image
import soundfile
import pdb
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
import json
import math
import librosa
import fire

user_prompt = '<|user|>'
assistant_prompt = '<|assistant|>'
prompt_suffix = '<|end|>'

model_path = '/mnt/pfs-guan-ssai/nlu/guoshoutao/Phi-4-multimodal'

kwargs = {}
kwargs['torch_dtype'] = torch.bfloat16

def get_model_and_processor(model_path):
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    print(processor.tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype='auto',
        _attn_implementation='flash_attention_2',
    ).cuda()
    print("model.config._attn_implementation:", model.config._attn_implementation)

    generation_config = GenerationConfig.from_pretrained(model_path, 'generation_config.json')

    return model, processor, generation_config

def translate_with_cot(audio, sr, instruction, sp_prompt=None, max_tokens=1000):
    prompt = instruction
    if sp_prompt is not None:
        prompt = f'{user_prompt}<|audio_1|>{sp_prompt}'
    else:
        prompt = f'{user_prompt}<|audio_1|>{prompt}{prompt_suffix}{assistant_prompt}'

    print(f'>>> Prompt\n{prompt}')
    # audio = soundfile.read(audio_dir)
    inputs = processor(text=prompt, audios=[(audio, sr)], return_tensors='pt').to('cuda:0')
    generate_ids = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        generation_config=generation_config,
    )
    generate_ids = generate_ids[:, inputs['input_ids'].shape[1] :]
    response = processor.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    print(f'>>> Response\n{response}')

    translation = response.split('<sep>')[-1].strip()
    transcript = response.split('<sep>')[0].strip()
    return [translation, transcript]

def asr(audio, sr, instruction, model, processor, generation_config, max_new_tokens=1000):
    prompt = instruction
    prompt = f'{user_prompt}<|audio_1|>{prompt}{prompt_suffix}{assistant_prompt}'

    print(f'>>> Prompt\n{prompt}')
    # audio = soundfile.read(audio_dir)
    inputs = processor(text=prompt, audios=[(audio, sr)], return_tensors='pt').to('cuda:0')
    generate_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        generation_config=generation_config,
    )
    generate_ids = generate_ids[:, inputs['input_ids'].shape[1] :]
    response = processor.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    print(f'>>> Response\n{response}')

    transcript = response.split('<sep>')[0].strip()
    return transcript

def has_punctuation(sentence):
    final_des = False
    for i in range(len(sentence)):
        if sentence[i] in '.?!;:':
            final_des = True
            break
    
    if final_des == False:
        return False
    return sentence[i+1:].strip(' .?!;:') != ''

def meet_condition(history_asr, chunk_length):
    '''
    inputs:
        history_asr: asr queue
        chunk_length: number of chunk

    return:
        whether to trigger truncation
        truncation timing
    '''

    length = len(history_asr)
    # too short duration
    if length <= 2:
        return (False, -1)
    
    # too long duration
    if chunk_length > 30 :
        return (True, - 1)
    
    all_same = True

    # Long Silence
    i = max(0, length - 3)
    while i < length - 1:
        if history_asr[i] != history_asr[i+1]:
            all_same = False
            break
        i += 1
    
    if all_same:
        return (True, 2)
    else:
        # whether to just finish speaking a sentence
        for i in range(length-1):
            if history_asr[i] != history_asr[i+1] and (history_asr[i] in history_asr[i+1]) and has_punctuation(history_asr[i+1]) and has_punctuation(history_asr[i]) == False:
                j = i
                if j == 0:
                    return (True, len(history_asr) - 1)
                while j >= 0:
                    if  (history_asr[j] in history_asr[i+1]) and has_punctuation(history_asr[j]) == False:
                        return (True, len(history_asr) - 1 - j)
                    j -= 1
                return (True, len(history_asr) - 1)
    return (False, -1)

def all_blank_detection(history_asr):
    if len(history_asr) <= 2:
        return False

    all_same = True
    i = max(0, len(history_asr) - 3)
    while i < len(history_asr) - 1:
        if history_asr[i] != '':
            return False
        i += 1    
    return True
    

def decide_num(text, lang):
    if text == '':
        return 0
    if lang == 'zh':
        number = list(jieba.cut(text))
    else:
        number = text.split(' ')

    return len(number)

def stream_asr_st(audio_path, prompt, chunk_length, pool_size, waitk, model, processor, generation_config, lang_pair):
    # determine the chunk size, and load audio
    chunk_length = chunk_length / 1000.0
    audio, sr = librosa.load(audio_path)
    source_lang, target_lang = lang_pair.split('_')
    
    # the number chunks
    num_chunks = math.ceil(len(audio) / sr / chunk_length)
    
    # history_asr stores historical transcription, history_translation stores real-time translation
    history_asr = []
    history_translation = []
    
    simul_st = []
    chunk_begin = 0
    
    i = 0
    while i < num_chunks:
        # determine the current input audio chunk
        begin_time = int(i * chunk_length * sr)
        end_time = int((i + 1) * chunk_length * sr)
        
        chunk = audio[chunk_begin:end_time]
        
        # get real-time transcription
        tmp_asr = asr(chunk, sr, prompt, model, processor, generation_config, max_new_tokens=math.ceil(((end_time-chunk_begin) / sr) * 5))
        print(tmp_asr)

        # silence
        if tmp_asr == '':
            i += 1
            if all_blank_detection(history_asr):
                chunk_begin = end_time
            continue
        
        # maintain the queue
        history_asr.append(tmp_asr.lower())
        history_asr = history_asr[-pool_size:]
        
        # determine whether to trigger truncation
        meet_condition_result = meet_condition(history_asr, float(end_time-chunk_begin)/sr)

        if meet_condition_result[0] or i == num_chunks - 1:
            # should trigger truncation    
            reduce_length = len(history_asr) - meet_condition_result[1]
            if meet_condition_result[-1] == -1:
                reduce_length = 0
            current_chunk_end = end_time - int((reduce_length * chunk_length * sr))
            
            chunk = audio[chunk_begin:current_chunk_end]
            
            current_trans = ' '.join(simul_st)
            target_text = current_trans
            cot_asr = history_asr[-(1+reduce_length)]
            sp_prompt = f'{user_prompt}<|audio_1|>{prompt}{prompt_suffix}{assistant_prompt}'
            if target_text.strip() != '':
                new_prompt = sp_prompt[len('<|user|><|audio_1|>'):] + cot_asr + ' <sep> ' + target_text.strip()
            else:
                new_prompt = sp_prompt[len('<|user|><|audio_1|>'):] + cot_asr + ' <sep>'

            translation = translate_with_cot(chunk, sr, prompt, new_prompt)

            history_translation.append({
                'end_time': end_time,
                'translation': translation[0],
                'split': 'True'
            })
            print('Write: ' + current_trans + ' ' + translation[0])
            
            i -= reduce_length
            chunk_begin = current_chunk_end
            history_asr = []
            simul_st = []
            
        else:
            current_trans = ' '.join(simul_st)
            src_num = decide_num(tmp_asr, source_lang)
            tgt_num = decide_num(current_trans, target_lang)
            
            # generation policy, the number to be ouput at this step
            to_be_generate = ((src_num - tgt_num) - waitk) + 1
            
            if to_be_generate <= 0 or to_be_generate > 8:
                print('Read: ' + tmp_asr)
            else:
                sp_prompt = f'{user_prompt}<|audio_1|>{prompt}{prompt_suffix}{assistant_prompt}'
                
                target_text = current_trans
                if target_text.strip() != '':
                    new_prompt = sp_prompt[len('<|user|><|audio_1|>'):] + tmp_asr + ' <sep> ' + target_text.strip()
                else:
                    new_prompt = sp_prompt[len('<|user|><|audio_1|>'):] + tmp_asr + ' <sep>'
                
                cot_st = translate_with_cot(chunk, sr, prompt, new_prompt, max_tokens=to_be_generate*5)
                cot_st = cot_st[0].strip('.。 ')
                if cot_st == '':
                    print('Read: ' + tmp_asr)
                else:
                    target_text = ' '.join(cot_st.split(' ')[:to_be_generate])
                    simul_st.append(target_text)
                    print('Write: ' + ' '.join(simul_st))
                    history_translation.append({
                        'end_time': end_time,
                        'translation': target_text,
                        'split': 'False'
                    })
            
        i += 1
            
    return history_translation
    
def stream_cot_st(stream_st, prompt, chunk_length, pool_size, waitk, model, processor, generation_config, lang_pair):
    results = []
    # Simulate the manner in which each audio stream arrives as chunks
    for i in range(len(stream_st)):
        tmp_json = stream_st[i]
        audio_path = tmp_json['audio_path']
        tmp_json['inference_translation'] = stream_asr_st(audio_path, prompt, chunk_length, pool_size, waitk, model, processor, generation_config, lang_pair)
        results.append(tmp_json)
    return results
            
def entry_port(model_path, chunk_length, queue_size, wait_k, cot_instruction, infer_json, output_dir, lang_pair):
    """
    Entry point for streaming speech translation from English to German using a wait-k policy.
    
    This function performs streaming speech translation by processing audio in chunks,
    first transcribing the audio to text, then translating the transcription to translation.
    The results include both the original transcript and the translation separated
    by '<sep>'.
    
    Args:
        model_path (str): Path to the pre-trained model used for speech translation.
        chunk_length (int): Length of audio chunks (in ms) for streaming processing.
        queue_size (int): Size of the transcript pool for parallel processing.
        wait_k (int): Wait-k parameter that determines the delay policy in simultaneous translation.
                      The system waits for k source words before starting translation.
        cot_instruction (str): Chain-of-thought instruction for guiding the translation process.
        infer_json (str): Path to the JSON file containing inference audios.
        output_dir (str): Directory path where the translation results will be saved.
    
    Returns:
        None: Results are saved to a JSON file in the specified output directory.
    
    """
    prompt = 'Transcribe the audio to text, and then translate the audio to German. Use <sep> as a separator between the original transcript and the translation.'    
    json_dir = '/mnt/pfs-guan-ssai/nlu/guoshoutao/datasets/MuST-C/En-De/mustc_en_de_stream_test.json'
    output_dir = '/mnt/pfs-guan-ssai/nlu/guoshoutao/results/MuST-C/phi4/En-De/stream_st/new_vanilla_simul_' + str(chunk_length) + '_' + str(pool_size) + '_' + str(waitk) + '.json'
    
    model, processor, generation_config = get_model_and_processor(model_path)

    stream_st = json.load(open(infer_json, 'r', encoding='utf-8'))
    results = stream_cot_st(stream_st, cot_instruction, chunk_length, queue_size, wait_k, model, processor, generation_config, lang_pair)
    
    with open(output_dir, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    print(processor.tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype='auto',
        _attn_implementation='flash_attention_2',
    ).cuda()
    print("model.config._attn_implementation:", model.config._attn_implementation)

    generation_config = GenerationConfig.from_pretrained(model_path, 'generation_config.json')

    return model, processor, generation_config

if __name__ == '__main__':
    fire.Fire(entry_port)