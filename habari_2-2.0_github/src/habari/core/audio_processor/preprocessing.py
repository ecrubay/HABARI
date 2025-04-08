
# habari-implementation-2.py
# Additional Learning and Transfer Mechanism for HABARI 2.0

from transformers import Trainer, TrainingArguments, Wav2Vec2ForSequenceClassification
from datasets import load_dataset

class TransferLearningEcoSound(EcoSoundModel):
    def fine_tune(self, dataset_path, num_labels=10, epochs=5, learning_rate=3e-5):
        dataset = load_dataset("csv", data_files=dataset_path, split="train")
        dataset = dataset.map(lambda x: self.processor(x["audio"], sampling_rate=16000, return_tensors="pt", padding=True), batched=True)

        model = Wav2Vec2ForSequenceClassification.from_pretrained("facebook/wav2vec2-base", num_labels=num_labels)
        args = TrainingArguments(output_dir="./results", evaluation_strategy="epoch", learning_rate=learning_rate,
                                 per_device_train_batch_size=8, num_train_epochs=epochs, weight_decay=0.01)

        trainer = Trainer(model=model, args=args, train_dataset=dataset)
        trainer.train()
        model.save_pretrained("./eco_sound_model")

        return model
