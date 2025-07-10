import json
import pdb
import string
import librosa
import tempfile
import fire
from typing import Dict, List, Any, Tuple
import subprocess
import os
sample_rate = 22050
def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

class MwerSegmenter:
    """
    Executes the mWERSegmenter tool introduced in `"Evaluating Machine Translation Output
    with Automatic Sentence Segmentation" by Matusov et al. (2005)
    <https://aclanthology.org/2005.iwslt-1.19/>`_.

    The tool can be downloaded at:
    https://www-i6.informatik.rwth-aachen.de/web/Software/mwerSegmenter.tar.gz
    """
    def __init__(self, character_level=False):
        self.mwer_command = "mwerSegmenter"
        self.character_level = character_level
        
        mwerSegmenter_root = '/data/guoshoutao/segment_tools/mwerSegmenter/mwerSegmenter'
        
        self.mwer_command = mwerSegmenter_root + "/mwerSegmenter"
        
    def __call__(self, prediction: str, reference_sentences: List[str]) -> List[str]:
        """
        Segments the prediction based on the reference sentences using the edit distance algorithm.
        """
        tmp_pred = tempfile.NamedTemporaryFile(mode="w", delete=False)
        tmp_ref = tempfile.NamedTemporaryFile(mode="w", delete=False)
        # create a  temporary directory where mwerSegmenter writes the segments
        # so that multiple parallel runs of stream_laal do not conflict
        tmp_dir = tempfile.mkdtemp()
        if self.character_level:
            # If character-level evaluation, add spaces for resegmentation
            prediction = " ".join(prediction)
            reference_sentences = [" ".join(reference) for reference in reference_sentences]
        try:
            tmp_pred.write(prediction)
            tmp_ref.writelines(ref + '\n' for ref in reference_sentences)
            tmp_pred.flush()
            tmp_ref.flush()
            subprocess.run([
                self.mwer_command,
                "-mref",
                tmp_ref.name,
                "-hypfile",
                tmp_pred.name,
                "-usecase",
                "1"], cwd=tmp_dir)
            # mwerSegmenter writes into the __segments file in the temporary directory. 
            segments_file = os.path.join(tmp_dir, "__segments")
            with open(segments_file, "r") as f:
                segments = []
                for line in f.readlines():
                    if self.character_level:
                        # If character-level evaluation, remove only spaces between characters
                        line = re.sub(r'(.)\s', r'\1', line)
                    segments.append(line.strip())
                return segments
        finally:
            tmp_pred.close()
            tmp_ref.close()
            os.unlink(tmp_pred.name)
            os.unlink(tmp_ref.name)
            os.unlink(segments_file)
            os.rmdir(tmp_dir)
            
def merge_text(all_segments):
    text = ' '
    
    for seg in all_segments:
        if len(seg) <= 0:
            continue
        if (text[-1] in string.punctuation) and (seg[0] in string.punctuation) == False:
            text += ' ' + seg
        elif (text[-1] in string.punctuation) == False and (seg[0] in string.punctuation) == False:
            text += ' ' + seg
        else:
            text += seg
    return text.strip()


def process_time_and_translation(segment, curr_translation_list):
    total_translation_txt = ''
    elpased_time_list = []
    segment_time_list = []
    begin_time_list = []
    
    for i in range(len(segment['inference_translation'])):
        total_translation_txt += segment['inference_translation'][i]['translation'] + ' '
        elpased_time_list += len(segment['inference_translation'][i]['translation'].split(' ')) * [segment['inference_translation'][i]['end_time'] / float(sample_rate) * 1000]
    
    for i in range(len(segment['chunks'])):
        begin_time_list.append(segment['chunks'][i]['chunk_start_time'])
        
    total_translation_txt = total_translation_txt.strip().split(' ')
    
    begin_index = 0
    curr_index = 0
    prediction_text_list = []
    
    for i in range(len(curr_translation_list)):
        translation = curr_translation_list[i]
        
        if translation == '':
            segment_time_list.append([])
            prediction_text_list.append('')
            continue
        
        while translation != merge_text(total_translation_txt[begin_index: curr_index+1]) and curr_index < len(total_translation_txt):
            if curr_index != begin_index:
                elpased_time_list[curr_index] = elpased_time_list[curr_index] - begin_time_list[i]
            curr_index += 1
        
        try:
            elpased_time_list[curr_index] = elpased_time_list[curr_index] - begin_time_list[i]
        except IndexError:
            pdb.set_trace()
        elpased_time_list[begin_index] = elpased_time_list[begin_index] - begin_time_list[i]
        segment_time_list.append(elpased_time_list[begin_index: curr_index+1])
        prediction_text_list.append(merge_text(total_translation_txt[begin_index: curr_index+1]))
        # print(merge_text(total_translation_txt[begin_index: curr_index+1]))
        curr_index += 1
        begin_index = curr_index

    return prediction_text_list, elpased_time_list, segment_time_list

def calculate_latency(segment_time_list, reference_list):
    tmp_latency_items = []
    
    numerator = 0
    tao = 0
    for sample_time_list in segment_time_list:
        for i in range(len(sample_time_list)):
            if sample_time_list[-1] <= sample_time_list[i]:
                tao = i + 1
                break
        tmp_latency_items.append({
            'latency_items': sample_time_list,
            'tao': tao,
            'translation_length': len(sample_time_list),
            'reference_length': len(reference_list[numerator].split(' ')),
        })
        
        numerator += 1
    return tmp_latency_items

def main(directory, file_name):
    
    translation_file = json.load(open(directory + '/' + file_name, "r"))

    stream_source_file = open(directory + "/stream_source.txt", "w")
    stream_target_file = open(directory + "/stream_target.txt", "w")
    stream_translation_file = open(directory + "/stream_translation.txt", "w")
    segment_translation_file = open(directory + "/segment_translation.txt", "w")

    # max_meaning_duration_time 1s
    max_meaning_duration_time = 0.0
    sample_rate = 16000

    source_text_list = []
    segment_reference_text_list = []
    translation_text_list = []
    segment_translation_text_list = []

    mwer_segmenter = MwerSegmenter(character_level=False)

    # Get source, reference, translation
    for i in range(len(translation_file)):
        segment = translation_file[i]
        
        source_text = ''
        target_text = ''
        target_text_list = []

        target_avg_length = 0.0

        for j in range(len(segment['chunks'])):
            
            segment['chunks'][j]['seg_begin_time'] = (segment['chunks'][j]['seg_begin_time'] - segment['begin_time']) / 16000 * 1000.0
            segment['chunks'][j]['chunk_start_time'] = (segment['chunks'][j]['chunk_start_time'] - segment['begin_time']) / 16000 * 1000.0
            segment['chunks'][j]['chunk_end_time'] = (segment['chunks'][j]['chunk_end_time'] - segment['begin_time']) / 16000 * 1000.0
            
            source_text += segment['chunks'][j]['source_text'] + ' '
            target_text += segment['chunks'][j]['target_text'] + ' '
            target_text_list.append(segment['chunks'][j]['target_text'])

            target_avg_length += len(segment['chunks'][j]['target_text'].split(' '))
        
        stream_source_file.write(source_text.strip() + "\n")
        stream_target_file.write(target_text.strip() + "\n")
        source_text_list.append(source_text.strip())
        segment_reference_text_list.append(target_text_list)
        

    # Get translation
    for i in range(len(translation_file)):
        segment = translation_file[i]
        #print(i)
        translation_text = ''

        # y, sr = librosa.load(segment['audio_path'])
        # total_time = (len(y) / float(sr)) * 1000
        # total_time_list.append(total_time)
        curr_tran_seg_list = []

        for j in range(len(segment['inference_translation'])):
            curr_tran_seg_list += segment['inference_translation'][j]['translation'].split(' ')
        
        translation_text_list.append(merge_text(curr_tran_seg_list))
        stream_translation_file.write(translation_text_list[-1].strip() + "\n")


    # Segment translation
    for i in range(len(translation_text_list)):
        # curr_translation_list = mwer_segmenter(remove_punctuation(translation_text_list[i]), [remove_punctuation(ref) for ref in segment_reference_text_list[i]])
        curr_translation_list = mwer_segmenter(translation_text_list[i], [ref for ref in segment_reference_text_list[i]])
        
        segment_translation_text_list.append(curr_translation_list)
        for line in curr_translation_list:
            segment_translation_file.write(line.strip() + "\n")


    trans_avg_length_list = []
    elpased_time_list = []
    total_time_list = []
    
    latency_items = []
        
    # Latency Preparation
    for i in range(len(segment_translation_text_list)):
        curr_translation_list = segment_translation_text_list[i]
        segment = translation_file[i]
        try:
            prediction_text_list, elpased_time_list, segment_time_list = process_time_and_translation(segment, curr_translation_list)
        except Exception as e:
            continue
        
        latency_items += calculate_latency(segment_time_list, segment_reference_text_list[i])

    # Latency Calculation
    
    total_laal = 0.0
    numer = 0
    
    for i in range(len(latency_items)):
        laal = 0.0
        
        tao = latency_items[i]['tao']
        translation_length = latency_items[i]['translation_length']
        reference_length = latency_items[i]['reference_length']
        elpased_time_list = latency_items[i]['latency_items']
        if len(elpased_time_list) > 0:
            for j in range(tao):
                laal += elpased_time_list[j] - j * (elpased_time_list[-1] / max(translation_length, reference_length))
            laal /= tao
        else:
            laal = 0.0
        print(laal)
        if laal <= -10000:
            laal = 0.0
            numer += 1
        
        total_laal += laal / float(tao)

    total_laal /= len(latency_items)
    print("Total number of segments: ", len(latency_items))
    print("Total number of segments with negative latency: ", numer)
    print("LAAL: ", total_laal)
        

    stream_source_file.close()
    stream_target_file.close()
    stream_translation_file.close()
    segment_translation_file.close()
    
if __name__ == "__main__":
    fire.Fire(main)