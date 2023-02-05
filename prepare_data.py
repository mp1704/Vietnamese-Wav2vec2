from constant import *
from split_audio import *
from clean_text import *
#Create vocab+tokenizer+processor
chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”\�]'

def remove_special_characters(batch):
    batch["text"] = re.sub(chars_to_ignore_regex, '', batch["text"]).lower()
    return batch

def extract_all_chars(batch):
  all_text = " ".join(batch["text"])
  vocab = list(set(all_text))
  return {"vocab": [vocab], "all_text": [all_text]}

def create_vocab(train_ds, test_ds, vocab_json):
  train_ds = train_ds.map(remove_special_characters)
  test_ds = test_ds.map(remove_special_characters)

  vocab_train = train_ds.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=train_ds.column_names)
  vocab_test = test_ds.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=test_ds.column_names)

  vocab_list = list(set(vocab_train["vocab"][0]) | set(vocab_test["vocab"][0]))
  vocab_dict = {v: k for k, v in enumerate(vocab_list)}

  vocab_dict["|"] = vocab_dict[" "]
  del vocab_dict[" "]
  vocab_dict["[UNK]"] = len(vocab_dict)
  vocab_dict["[PAD]"] = len(vocab_dict)

  with open(vocab_json, 'w') as vocab_file:
    json.dump(vocab_dict, vocab_file)

def create_tokenizer(model_path, train_ds, test_ds):
  try:
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(model_path, do_lower_case=True)
  except:
    vocab_json = "./tmp_vocab.json"
    create_vocab(train_ds, test_ds, vocab_json)

    tokenizer = Wav2Vec2CTCTokenizer(vocab_json, unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
  return tokenizer
def create_processor(model_path):
  try:
    processor = AutoProcessor.from_pretrained(model_path)
  except:
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
    tokenizer = create_tokenizer(model_path)
    processor = Wav2Vec2Processor(feature_extractor, tokenizer)
  return processor

if __name__ == '__main__':
  results = []
  for folder in DATA_DIR:
    result = split_folder(folder)
    results.extend(result)
  # clean
  df = pd.DataFrame(results)
  df = df[df["text"].str.find('???') == -1]
  df = df.apply(clear_text, axis=1)
  train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
  train_df.to_csv(os.path.join(OUTPUT_DIR, "train_df.csv"), index=False)
  test_df.to_csv(os.path.join(OUTPUT_DIR, "test_df.csv"), index=False)
  
  ## load train dataset from pre-step
  train_ds = datasets.load_dataset('csv', data_files=os.path.join(OUTPUT_DIR, "train_df.csv"), keep_in_memory=True, split='train')
  test_ds = datasets.load_dataset('csv', data_files=os.path.join(OUTPUT_DIR, "test_df.csv"), keep_in_memory=True, split='train')
  processor = create_processor(BASE_WAV2VEC_PROCESSOR)
  def speech_file_to_array_fn(batch):
    speech_array, sampling_rate = sf.read(batch["file"])
    batch["input_values"] = processor(speech_array, sampling_rate=sampling_rate).input_values[0]
    with processor.as_target_processor():
        batch["labels"] = processor(batch["text"]).input_ids
    return batch
  train_ds = train_ds.map(speech_file_to_array_fn, remove_columns=train_ds.column_names)
  test_ds = test_ds.map(speech_file_to_array_fn, remove_columns=test_ds.column_names) 
  train_ds.save_to_disk(os.path.join(OUTPUT_DIR, "hf_datastet", "train"))
  test_ds.save_to_disk(os.path.join(OUTPUT_DIR, "hf_datastet", "test"))