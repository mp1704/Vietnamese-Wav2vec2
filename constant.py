# Define constant variables
import glob
import os
import soundfile as sf
import uuid
import pandas as pd
import librosa
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import re
import glob
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import datasets
from sklearn.model_selection import train_test_split
import torch
import os
import pandas as pd
import soundfile as sf
import numpy as np
import re
import json
from transformers import AutoProcessor, Wav2Vec2Processor, Wav2Vec2ProcessorWithLM, Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC, TrainingArguments, Trainer, EarlyStoppingCallback

OUTPUT_DIR = './data'
# change this dir
DATA_DIR = [
  '/content/drive/MyDrive/Colab Notebooks/audio'
]
BASE_WAV2VEC_MODEL = "nguyenvulebinh/wav2vec2-base-vietnamese-250h"
BASE_WAV2VEC_PROCESSOR = BASE_WAV2VEC_MODEL

MY_MODEL_PATH = '/wav2vec2_pmp' 