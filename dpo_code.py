#pip install trl

from trl import DPOTrainer, DPOConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, default_data_collator
import torch
from datasets import Dataset, load_dataset
from huggingface_hub import login
import os
import pandas as pd
from peft import LoraConfig, get_peft_model, TaskType
from datasets import DatasetDict
import json

#notebook_login()
login(token =
os.environ["WANDB_DISABLED"] = "true" #disable wandb logging dont use

model_name = "meta-llama/Llama-3.2-1B-Instruct"
#Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

#dataset from hugging face (diseases and symptoms)
#using pandas
df = pd.read_csv("hf://datasets/QuyenAnhDE/Diseases_Symptoms/Diseases_Symptoms.csv")
ds = Dataset.from_pandas(df)
# Wrap the single dataset as a DatasetDict with a 'train' split

ds = DatasetDict({"train": ds})
ds["train"] = ds["train"].select(range(400))  # start w/ 100 samples

#load dpo data set
dpo_data = load_dataset("json", data_files={"train": "dpo_data.jsonl"})

if(tokenizer.pad_token is None):
   tokenizer.add_special_tokens({'pad_token': '[PAD]'})
   model.resize_token_embeddings(len(tokenizer))


#column_names = list(dpo_data["train"].features) #saves origonal column names prompt, chosen, rejected

def apply_dpo_template

dpo_data = dpo_data.map(
    apply_dpo_template,
   # remove_columns = column_names,
    lambda examples: {
        "prompt_input_ids": tokenizer(examples["prompt"], padding="max_length", truncation=True)["input_ids"],
        "prompt_attention_mask": tokenizer(examples["prompt"], padding="max_length", truncation=True)["attention_mask"],
        "chosen_input_ids": tokenizer(examples["chosen"], padding="max_length", truncation=True)["input_ids"],
        "chosen_attention_mask": tokenizer(examples["chosen"], padding="max_length", truncation=True)["attention_mask"],
        "rejected_input_ids": tokenizer(examples["rejected"], padding="max_length", truncation=True)["input_ids"],
        "rejected_attention_mask": tokenizer(examples["rejected"], padding="max_length", truncation=True)["attention_mask"],
    }, 
    desc = "Formatting comparisons with prompt template",
    batched = True
  )

'''dpo_data = dpo_data.rename_columns({
    "text_chosen": "chosen",
    "text_rejected": "rejected"
})'''
print(dpo_data.column_names)


#to reduce gpu and mem use, and make faster
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
)

peft_model_train = get_peft_model(model, lora_config)

#use cpu if not enough gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
peft_model_train.to(device) #move model to device
#output_dir = "./llama3.2_disease_model"
ref_model = AutoModelForCausalLM.from_pretrained(model_name).to(device) #baseline for comparison - frozen model. No lora

dpo_config = DPOConfig(
    output_dir = "./llama3.2_dpo_disease_model", #results folder will be created during process with steps saved
    per_device_train_batch_size = 1,
    gradient_accumulation_steps=4, #to have larger batch sizes, dont run out of mem so easily
    num_train_epochs = 3,
    logging_steps = 10,
    save_strategy= "no", #no saving during training, for now
    learning_rate = 1e-5,
    fp16 = True, # using GPU...
    #evaluation_strategy = "no" #no eval during training needed?? wouldn't be recognized by transformer module
    beta = 0.1
)

trainer = DPOTrainer(
  model = peft_model_train,
  ref_model = ref_model,
  #beta = 0.1,
  #args = training_args,
  args = dpo_config,
  train_dataset = dpo_data['train'],#tokenized_dataset['train'],
  #tokenizer = tokenizer,
  processing_class = tokenizer, #the tokenizer here
  data_collator = default_data_collator
)

trainer.train()
peft_model_train.eval() #eval mode

#dir will be created when training ends and has model and tokenizer stuff
peft_model_train.save_pretrained("./llama3.2_dpo_disease_model")
tokenizer.save_pretrained("./llama3.2_dpo_disease_model")




while True:
  user_input = input("You: ")
  if (user_input.lower() == "exit"): #keep chatting until user says exit
    break
  #this will be the chat prompt that gets prepended to the user input about symptoms so that this is the format it expects (start out formal)

  system_prompt = "<[SYS]> You are a helpful medical assistant. Respond with only the possible disease and treatments after being given the symptoms by the user. <[/SYS]>\n"
  user_prompt = f"<[USER]> Symptoms: {user_input} <[/USER]>\n"

  full_prompt = system_prompt + user_prompt + "<[ASSIST]>"
  inputs = tokenizer(full_prompt, return_tensors = "pt").to(device)

  with torch.no_grad():
     outputs = peft_model_train.generate( #OUTPUT FORMAT FIX
        **inputs,
        max_length = 1064, #max length of the generated response
        #early_stopping = True, #stop when we reach the end of the sentence
        #temperature = 0.1, #how random response is but should be 0 cuz need accuracy
        #top_k = 1, #pick highest probability token
        #top_p = 0.3, #include all prob??
        do_sample = True, # for accuracy over creativity
        temperature = 0.3, #how random response is but really should be cuz just matching to data
        top_k = 30, #top k sampling ???
        top_p = 0.95, #top p sampling ???
        pad_token_id = tokenizer.eos_token_id #pad token id to use
     )

  #chat prompt has recent exchanges and prompt to help generate response all in one string tho
  chat_response = tokenizer.decode(outputs[0], skip_special_tokens = True)
  #response has recent exchanges+ its response so need to strip the first part for just the answer
  #remove tags

  #answer = chat_response[len(full_system_prompt):].strip().split("\n")[0]

  if "<[ASSIST]>" in chat_response and "<[/ASSIST]>" in chat_response:
    answer = chat_response.split("<[ASSIST]>")[1].split("<[/ASSIST]>")[0].strip()
  else:
    answer = chat_response.strip()

  print("Chatbot: \n", answer)
