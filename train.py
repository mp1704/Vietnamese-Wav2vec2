import os
from argparse import ArgumentParser
from transformers import AutoProcessor, Wav2Vec2Processor, Wav2Vec2ProcessorWithLM, Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC, TrainingArguments, Trainer, EarlyStoppingCallback
from constant import *
from DataCollator import *
from prepare_data import *
from datasets import load_metric

wer_metric = load_metric("wer")

def get_compute_metrics_fnc(processor, wer_metric):
    def compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)

        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.batch_decode(pred_ids)
        # we do not want to group tokens when computing the metrics
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

        wer = wer_metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}
    return compute_metrics

if __name__ == "__main__":
    
    parser = ArgumentParser()
    
    # Arguments users used when running command lines
    parser.add_argument("--batch-size-per-device", default=16, type=int)
    parser.add_argument("--epochs", default=60, type=int)

    home_dir = os.getcwd()
    args = parser.parse_args()
    # print(args)

    train_ds= datasets.load_from_disk(os.path.join(OUTPUT_DIR, "hf_datastet", "train"))
    test_ds= datasets.load_from_disk(os.path.join(OUTPUT_DIR, "hf_datastet", "test"))
    
    processor = create_processor(BASE_WAV2VEC_PROCESSOR)
    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    training_args = TrainingArguments(
    output_dir='wav2vec2_base_vn',
    group_by_length=True,
    per_device_train_batch_size=args.batch_size_per_device,
    evaluation_strategy="steps",
    num_train_epochs=args.epochs,
    fp16=True,
    gradient_checkpointing=True, 
    save_steps=500,
    eval_steps=500,
    logging_steps=500,
    learning_rate=1e-4,
    weight_decay=0.005,
    warmup_steps=1000,
    save_total_limit=2,
    )

    model = Wav2Vec2ForCTC.from_pretrained(
        BASE_WAV2VEC_MODEL, 
        ctc_loss_reduction="mean", 
        pad_token_id=processor.tokenizer.pad_token_id,
    )
    model.freeze_feature_extractor()

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=get_compute_metrics_fnc(processor, wer_metric), # !!!
        train_dataset=train_ds,
        eval_dataset=test_ds,
        tokenizer=processor.feature_extractor,
    )

    trainer.train()

    # save
    trainer.save_model(MY_MODEL_PATH)
    processor.save_pretrained(save_directory  = MY_MODEL_PATH)
    

