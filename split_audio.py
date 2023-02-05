# SHOULD USE .WAV FILE
from constant import *

def get_audio_file(folder, txt_file):
  exts = ['wav', 'm4a', 'mp3']

  name = os.path.basename(txt_file)[:-4]

  for ext in exts:
    audio_files = glob.glob(glob.escape(folder) + "/**/" + name + '.' + ext, recursive=True)
    if len(audio_files) > 0:
      return audio_files[0]

  raise Exception(f"audio file not found: {name}")
def split_audio(txt_file, audio_file, output_dir):
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  # Get audio file
  audio_data, sample_rate = librosa.load(audio_file, sr=16000)

  results = []
  with open(txt_file) as file:
    for line in tqdm(file):
      data = line.strip().split('\t')
      if len(data) != 3:
        continue

      start = int(float(data[0]) * sample_rate)
      stop = int(float(data[1]) * sample_rate)
      if stop - start > 10 * sample_rate:
        continue

      text = data[2]

      audio_part = audio_data[start:stop]
      audio_part = np.concatenate([np.zeros((int(0.5 * sample_rate),)), audio_part, np.zeros((int(0.5 * sample_rate),))]) # padding zero

      audio_path = os.path.join(output_dir, str(uuid.uuid4()) + ".wav")

      with open(audio_path, 'wb') as out_file:
        sf.write(out_file, audio_part, sample_rate)

      results.append({
        "file": audio_path,
        "text": text
      })
  return results
def split_folder(folder):
  print(f"Split folder {folder}")

  results = []
  # Get list file txt
  txt_files = glob.glob(glob.escape(folder) + "/**/*.txt", recursive=True)
  for txt_file in txt_files:
    try:
      audio_file = get_audio_file(folder, txt_file)
      print("Subtitle file: ", txt_file)
      print("Audio file: ", audio_file)
      print()

      result = split_audio(txt_file, audio_file, OUTPUT_DIR)
      results.extend(result)

      dest_txt = os.path.join('data/raw_data', os.path.basename(txt_file))
      os.system(f'cp "{txt_file}" "{dest_txt}"')
      dest_audio = os.path.join('data/raw_data', os.path.basename(audio_file))
      os.system(f'cp "{audio_file}" "{dest_audio}"')  
    except:
      continue

  return results