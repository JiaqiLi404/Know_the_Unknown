# Know the Unknown
Code for paper: **Know the Unknown: An Uncertainty-Sensitive Method for LLM Instruction Tuning**

## Datasets and Benchmark
Our datasets and benchmark base on [ASQA](https://github.com/google-research/language/tree/master/language/asqa) and [HotpotQA](https://github.com/hotpotqa/hotpot).
For generating our proposed datasets, you need to refer to the scripts in `datasets` directory, and download the original datasets from the above links.

We provide the processed data and benchmark for our experiments.
The data and benchmark are available at [Onedrive](https://i3h5-my.sharepoint.com/:f:/g/personal/admin_ljqpersonal_com/EtxN0JRPoi5PjscUoDoRql8BPAz2G7TZSCIGLiG4WCmlMg).
Our the generated outputs of mainstream LLMs on our benchmark are also available at [Onedrive](https://i3h5-my.sharepoint.com/:f:/g/personal/admin_ljqpersonal_com/EtxN0JRPoi5PjscUoDoRql8BPAz2G7TZSCIGLiG4WCmlMg).


## Pretrained Models
We provide the pretrained models of our fine-tuned models at [Onedrive](https://i3h5-my.sharepoint.com/:f:/g/personal/admin_ljqpersonal_com/Eobo71eMD-1Eo3sAgm-gtAMBu64hzFOw-7jdLy-IsiOtuQ).
Those modes ended with `merged` are the entire weights merged with the original weights, while the others are just Lora heads.

## Training
We fork the code from [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) to achieve the fine-tuning.
As our proposed method is a two-stage framework, you need to first fine-tune the LLMs on the `TrustworthyLLM_Cognition_Finetuning_Dataset`, 
then fine-tune it on the `TrustworthyLLM_PromptSensitive_Finetuning_Dataset`. 
Here is an example of the command for fine-tuning on the `TrustworthyLLM_Cognition_Finetuning_Dataset`:
```bash
python LLaMA-Factory/src/train_bash.py
--stage
sft
--do_train
--model_name_or_path
/mnt/f/Models/llama-2-7b-chat-hf
--create_new_adapter
--dataset
TrustworthyLLM_Cognition_Finetuning_Dataset
--template
llama2
--finetuning_type
lora
--lora_target
q_proj,v_proj
--output_dir
models/TrustworthyLLM_Cognition_Finetuning_Model
--overwrite_cache
--per_device_train_batch_size
4
--gradient_accumulation_steps
4
--lr_scheduler_type
cosine
--logging_steps
10
--save_steps
1000
--learning_rate
4e-5
--num_train_epochs
1.0
--plot_loss
--fp16
```


