# Qwen2-Fine-Tuning-

$ srun --pty -p gpu --gres=gpu:1 /bin/bash
Warning: A 'Archive' link exists but the target is not available on this node
$ flight start
$ conda activate sip_01
$ python
Python 3.8.19 (default, Mar 20 2024, 19:58:24)
[GCC 11.2.0] :: Anaconda, Inc. on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> from transformers import AutoModelForCausalLM, AutoTokenizer
>>> tokenizer = AutoTokenizer.from_pretrained("/______/_____/Qwen2-7B-Instruct")
>>> model = AutoModelForCausalLM.from_pretrained("____/_____/Qwen2-7B-Instruct", device_map="auto")
Loading checkpoint shards: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:10
>>> import torch
>>> from transformers import TrainingArguments, Trainer
>>> from datasets import load_dataset
>>> from peft import LoraConfig, get_peft_model
>>>
>>> def preprocess_function(examples):
...     inputs = [f"{prompt}\n" for prompt in examples["prompt"]]
...     targets = [f"{completion}\n" for completion in examples["completion"]]
...     model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
...     labels = tokenizer(targets, max_length=512, truncation=True, padding="max_length")
...     model_inputs["labels"] = labels["input_ids"]
...     return model_inputs
...
>>> dataset = load_dataset("json", data_files="/_____/custom_dataset.json")
>>> tokenized_dataset = dataset["train"].train_test_split(test_size=0.1)
>>> tokenized_dataset = tokenized_dataset.map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names)
Map: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 27/27 [00:00<00:00, 1
Map: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:00<00:00,
>>> lora_config = LoraConfig(
...     r=8,
...     lora_alpha=32,
...     target_modules=["q_proj", "v_proj"],
...     lora_dropout=0.05,
...     bias="none",
...     task_type="CAUSAL_LM"
... )
>>> model = get_peft_model(model, lora_config)
>>> training_args = TrainingArguments(
...     output_dir="/____/finetuned",
...     num_train_epochs=3,
...     per_device_train_batch_size=4,
...     per_device_eval_batch_size=4,
...     warmup_steps=500,
...     weight_decay=0.01,
...     logging_dir="./logs",
...     logging_steps=10,
...     evaluation_strategy="steps",
...     eval_steps=500,
...     save_strategy="steps",
...     save_steps=1000,
...     learning_rate=1e-4,
...     fp16=True,
...     gradient_checkpointing=True,
...     gradient_accumulation_steps=4,
... )
FutureWarning: `evaluation_strategy` is deprecated and will be rem46 of 🤗 Transformers. Use `eval_strategy` instead
  warnings.warn(
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
        - Avoid using `tokenizers` before the fork if possible
        - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
>>> trainer = Trainer(
...     model=model,
...     args=training_args,
...     train_dataset=tokenized_dataset["train"],
...     eval_dataset=tokenized_dataset["test"],
...     tokenizer=tokenizer,
... )
<stdin>:1: FutureWarning: `tokenizer` is deprecated and will be removed in version 5.0.0 for `Trainer.__init__`. Use `processing_class` instead.
Detected kernel version 4.18.0, which is below the recommended minimum of 5.5.0; this can cause the process to hang. It is recommended to upgrade the kernel to the mhigher.
>>> model.enable_input_require_grads()
>>> trainer.train()
  0%|                                                                                                                                                            | 0/`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...
{'train_runtime': 8.9225, 'train_samples_per_second': 9.078, 'train_steps_per_second': 0.336, 'train_loss': 14.142072041829428, 'epoch': 1.71}
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:08
TrainOutput(global_step=3, training_loss=14.142072041829428, metrics={'train_runtime': 8.9225, 'train_samples_per_second': 9.078, 'train_steps_per_second': 0.336, 't8573800448.0, 'train_loss': 14.142072041829428, 'epoch': 1.7142857142857144})
>>> trainer.save_model("/____/finetuned")
>>> ftokenizer = AutoTokenizer.from_pretrained("/____/finetuned")
>>> prompt = "Translate the following sentence into Bengali- I am looking for ticket"
>>> import gc
>>> del model
>>> gc.collect()
20
>>> torch.cuda.empty_cache()
>>> fmodel = AutoModelForCausalLM.from_pretrained("/____/finetuned")
Loading checkpoint shards: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:07

>>> inputs = ftokenizer(prompt, return_tensors="pt")
>>> generate_ids = fmodel.generate(inputs.input_ids, max_length=60)
>>> ftokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
'Translate the following sentence into Bengali- I am looking for ticket booking service. আমি টিকেট বোঝানোর সেবা খুঁজছি। (Ami tiket bojanô'
# Not bad.. this was not directly into training data as i combined two sentences.
>>> del fmodel
>>> gc.collect()
20
>>> torch.cuda.empty_cache()
>>> tokenizer = AutoTokenizer.from_pretrained("/______/Qwen2-7B-Instruct")
>>> model = AutoModelForCausalLM.from_pretrained("/_____/Qwen2-7B-Instruct")
Loading checkpoint shards: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:08<00:00,  2.11s/it]
>>> prompt = "Translate the following sentence into Bengali- I am looking for ticket"
>>> inputs = tokenizer(prompt, return_tensors="pt")
>>> generate_ids = model.generate(inputs.input_ids, max_length=60)
>>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
'Translate the following sentence into Bengali- I am looking for ticket to Calcutta.\n\nBengali Translation: আমি কলকাতা নিয়ে টিকেট খুঁজছি।\n\nExplanation:'
# Original model performed poorly.
