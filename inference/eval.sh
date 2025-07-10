# 
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

echo "Stream BLEU: "
sacrebleu $STREAM_TARGET_FILE -i stream_translation.txt -m bleu -b -w 4 -lc

echo "Stream COMET: "
comet-score -s $STREAM_SOURCE_FILE -t stream_translation.txt -r $STREAM_TARGET_FILE --model comet-22/model.ckpt