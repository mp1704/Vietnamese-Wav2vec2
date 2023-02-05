from constant import *
from train import *
from transformers import AutoModelForCTC, Wav2Vec2Processor

def map_to_result(batch):
    with torch.no_grad():
        input_values = torch.tensor(batch["input_values"], device="cuda").unsqueeze(0)
        logits = model(input_values).logits

    pred_ids = torch.argmax(logits, dim=-1)
    batch["pred_str"] = processor.batch_decode(pred_ids)[0]
    batch["text"] = processor.decode(batch["labels"], group_tokens=False)

    return batch

if __name__ == '__main__':
    # load model and processor
    model = AutoModelForCTC.from_pretrained(MY_MODEL_PATH)
    processor = AutoProcessor.from_pretrained(MY_MODEL_PATH)
    # load dataset for evaluation
    test_ds = datasets.load_from_disk(os.path.join(OUTPUT_DIR, "hf_datastet", "test"))
    
    model.to("cuda")
    results = test_ds.map(map_to_result, remove_columns=test_ds.column_names)

    print("Test WER: {:.3f}".format(wer_metric.compute(predictions=results["pred_str"], references=results["text"])))
    
