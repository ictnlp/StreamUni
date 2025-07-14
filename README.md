


# StreamUni: Achieving Streaming Speech Translation with a Unified Large Speech-Language Model

[![paper](https://img.shields.io/badge/arXiv-2507.07803-b31b1b)]([https://arxiv.org/abs/2507.07803])
[![model](https://img.shields.io/badge/Huggingface-StreamUni_Phi4-brightgreen)]([https://huggingface.co/ICTNLP/StreamUni-Phi4])
[![model](https://img.shields.io/badge/Huggingface-StreamUni_data-brightred)]([https://huggingface.co/datasets/ICTNLP/StreamUni])

> **[Shoutao Guo](https://scholar.google.com/citations?user=XwHtPyAAAAAJ&hl=zh-CN), [Xiang Li](https://scholar.google.com.hk/citations?user=DMfYmIEAAAAJ&hl=zh-CN/), [Mengge Liu](https://scholar.google.com/citations?user=2WF8LjoAAAAJ&hl=zh-CN), [Wei Chen](https://ieeexplore.ieee.org/author/841945267640363) [Yang Feng*](https://people.ucas.edu.cn/~yangfeng?language=en)**
<p align="center">
  <img src="https://github.com/ictnlp/StreamUni/blob/main/model.png" alt="Image description" width="800">
</p>

StreamUni is a framework that efficiently enables unified Large Speech-Language Models to accomplish streaming speech translation in a cohesive manner. Experimental results demonstrate that StreamUni efficiently achieves state-of-the-art performance on streaming speech translation tasks across multiple directions.

Our method achieves the state-of-the-art performance on Streaming En-De task and Simultaneous En-Zh task.
<div style="display: flex;">
  <img src="https://github.com/ictnlp/StreamUni/blob/main/stream_ende.png" alt="图片1" style="width: 45%; margin-right: 15%;">
  <img src="https://github.com/ictnlp/StreamUni/blob/main/enzh.png" alt="图片2" style="width: 45%;">
</div>


## 🔥 Quick Start
### Requirements
- Install packages:

  ```bash
  pip install -r requirements.txt

  # For Stream Evaluation
  wget https://www-i6.informatik.rwth-aachen.de/web/Software/mwerSegmenter.tar.gz
  tar -zxvf your_file.tar.gz
  ```

### Fine-tuning Model
- You can fine-tune the Phi-4-Multimodal by running `**bash** [fintune/finetune.sh](https://github.com/ictnlp/StreamUni/blob/main)':
  ```bash
  MODEL_NAME=model_dir
  VOICE_DIR=train_json_dir
  OUTPUT_DIR=StreamUni_model
  
  BATCH_SIZE=32
  BATCH_SIZE_PER_GPU=2
  NUM_EPOCHS=1
  LEARNING_RATE=4e-5
  WEIGHT_DECAY=0.01
  
  deepspeed \
      --include localhost:0,1,2,3,4,5,6,7 \
      --master_port $MASTER_PORT \
      speech_finetune.py \
      --deepspeed zero2.json \
      --model_name_or_path $MODEL_NAME \
      --voice_dir $VOICE_DIR \
      --output_dir $OUTPUT_DIR \
      --batch_size $BATCH_SIZE \
      --batch_size_per_gpu $BATCH_SIZE_PER_GPU \
      --learning_rate $LEARNING_RATE \
      --wd $WEIGHT_DECAY \
      --use_flash_attention
  ```
  We provide an example **train_json_dir** in [fintune/train_example.json](https://github.com/ictnlp/StreamUni/blob/main/fintune/train_example.json)

### Inference and Evaluation


- You can run streaming speech translation inference by running `**bash** [inference/infer.sh](https://github.com/ictnlp/StreamUni/blob/main/inference/infer.sh)':
  ```bash
  MODEL_DIR="model_dir"
  CHUNK_LENGTH=640
  QUEUE_SIZE=3
  WAIT_K=5
  INSTRUCTION='Transcribe the audio to text, and then translate the audio to German. Use <sep> as a separator between the original transcript and the translation.'
  JSON_DIR="json_dir"
  OUTPUT_DIR="output_dir"
  LANG_PAIR="en_de"
  
  python stream_st_infer.py \
      --model_path "$MODEL_DIR" \
      --chunk_length $CHUNK_LENGTH \
      --queue_size $QUEUE_SIZE \
      --wait_k $WAIT_K \
      --cot_instruction "$INSTRUCTION" \
      --infer_json "$JSON_DIR" \
      --output_dir "$OUTPUT_DIR" \
      --lang_pair "$LANG_PAIR"
  ```
We provide an example **json_dir** in [inference/example_infer.json](https://github.com/ictnlp/StreamUni/blob/main/inference/example_infer.json)
After running inference scripts, we can obtain the output results, whose example is in [inference/example_infer.json](https://github.com/ictnlp/StreamUni/blob/main/inference/example_infer.json). Then we can evaluate the results.

- You can run evaluation to get the Stream LAAL, Stream SacreBLEU, Stream COMET, Document-level SareBLEU and Document-level COMET by running `**bash** [inference/eval.sh](https://github.com/ictnlp/StreamUni/blob/main/inference/eval.sh)':

  ```bash
    OUTPUT_DIR=output_dir
    OUTPUT_FILE=output_file
    SEG_SOURCE_FILE=seg_source_file
    SEG_TARGET_FILE=seg_target_file
    STREAM_SOURCE_FILE=stream_source_file
    STREAM_TARGET_FILE=stream_target_file
    
    python latency_cal.py --directory $OUTPUT_DIR --file_name $OUTPUT_FILE
    
    cd $OUTPUT_DIR
    
    echo "Stream BLEU: "
    sacrebleu $SEG_TARGET_FILE -i segment_translation.txt -m bleu -b -w 4 -lc
    
    echo "Stream COMET: "
    comet-score -s $SEG_SOURCE_FILE -t segment_translation.txt -r $SEG_TARGET_FILE --model comet-22/model.ckpt
    
    echo "Document BLEU: "
    sacrebleu $STREAM_TARGET_FILE -i stream_translation.txt -m bleu -b -w 4 -lc
    
    echo "Document COMET: "
    comet-score -s $STREAM_SOURCE_FILE -t stream_translation.txt -r $STREAM_TARGET_FILE --model comet-22/model.ckpt
  ```


## 🖋Citation

If you have any questions, please feel free to submit an issue or contact `guoshoutao22z@ict.ac.cn`.

If our work is useful for you, please cite as:

```
@misc{guo2025streamuniachievingstreamingspeech,
      title={StreamUni: Achieving Streaming Speech Translation with a Unified Large Speech-Language Model}, 
      author={Shoutao Guo and Xiang Li and Mengge Liu and Wei Chen and Yang Feng},
      year={2025},
      eprint={2507.07803},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2507.07803}, 
}
```
