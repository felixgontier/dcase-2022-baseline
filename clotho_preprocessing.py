import numpy as np
import csv
from tqdm import tqdm
from pathlib import Path
import torch
import librosa as lr
import yaml
import argparse
import string

def preprocess_dataset(config):
    with open('data_settings/{}.yaml'.format(config.cfg), 'r') as f:
        settings = yaml.safe_load(f)
    
    vggish_model = torch.hub.load('harritaylor/torchvggish', 'vggish')
    vggish_model.eval()
    vggish_model.postprocess = False
    vggish_model.embeddings[5] = torch.nn.Sequential() # Remove last ReLU
    
    splits = settings['data']['splits']
    for split in splits:
        print('Split '+split+'.')
        out_path = Path(settings['data']['root_path'], settings['data']['output_path'], split)
        out_path.mkdir(parents=True, exist_ok=True)
        
        data_path = Path(settings['data']['root_path'], settings['data']['input_path'], split)
        audio_list = [fname for fname in data_path.iterdir() if fname.suffix == '.wav']
        
        annotation_path = Path(settings['data']['root_path'], settings['data']['input_path'], 'clotho_captions_'+split+'.csv')
        if annotation_path.exists():
            example_list = []
            with open(annotation_path, 'r') as f:
                reader = csv.reader(f, delimiter=',')
                next(reader)
                for r in reader:
                    assert(len(r)==6) # Ensure captions and filename are parsed correctly
                    example_list.append(r)
        
            for ex in tqdm(example_list[:10]):
                file_name = ex[0]
                if data_path.joinpath(file_name) in audio_list:
                    # Get captions
                    captions = ex[1:]
                    
                    # Pre-process captions
                    # Remove punctuation, lowercase
                    captions = [c.translate(str.maketrans('', '', string.punctuation)).lower() for c in captions]
                    
                    # Compute VGGish embeddings
                    vggish_embeddings = vggish_model.forward(str(data_path.joinpath(file_name))).detach().numpy()
                    
                    # Output one npy file per reference caption
                    for i_cap, caption in enumerate(captions):
                        # Create recarray
                        temp_rec_array = np.rec.array(np.array(
                            (file_name, vggish_embeddings, caption),
                            dtype=[
                                ('file_name', 'U{}'.format(len(file_name))),
                                ('vggish_embeddings', np.dtype(object)),
                                ('caption', 'U{}'.format(len(caption))),
                            ]
                        ))
                        # Save recarray
                        np.save(str(out_path.joinpath('clotho_{}_{}.npy'.format(file_name, i_cap))), temp_rec_array)
                        
                else:
                    print('Missing audio file {} in split {}.'.format(file_name, split))
        else: # No captions (e.g. testing split)
            for ex in tqdm(audio_list):
                file_name = ex.name
                # Compute VGGish embeddings
                vggish_embeddings = vggish_model.forward(str(ex)).detach().numpy()
                
                # Create recarray
                temp_rec_array = np.rec.array(np.array(
                    (file_name, vggish_embeddings, None),
                    dtype=[
                        ('file_name', 'U{}'.format(len(file_name))),
                        ('vggish_embeddings', np.dtype(object)),
                        ('caption', np.dtype(object)),
                    ]
                ))
                
                # Save recarray
                np.save(str(out_path.joinpath('clotho_{}.npy'.format(file_name))), temp_rec_array)
                
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='default', help='YAML configuration file for dataset pre-processing')
    config = parser.parse_args()
    preprocess_dataset(config)
    
