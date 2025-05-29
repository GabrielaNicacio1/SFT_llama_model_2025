from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, pipeline, default_data_collator #DataCollatorForLanguageModeling
import torch
#import numpy as np
#import evaluate
from datasets import load_dataset

import os
os.environ["WANDB_DISABLED"] = "true" #disable wandb logging to avoid errors

#try whole new model cuz other had too many issues
model_name = "meta-llama/Llama-3.2-1B"
#dataset from hugging face to SFT on (diseases and symptoms)
ds = load_dataset("QuyenAnhDE/Diseases_Symptoms") #400 rows

#Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

#pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

#try first this
#prompt = "List symptoms of the flu and how to treat it."
#to speed things up, limit output length
'''output = pipe(
     "List symptoms of the flu and how to treat it.",
    max_new_tokens=80,        # Limit output length
    do_sample=False          #faster on CPU
    temperature = 0.6
    )

print(output[0]['generated_text'])
'''

#tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True) #cant use flash attention and prob dont need cuz for speed --set to FALSE instead
#model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

if(tokenizer.pad_token is None):
   tokenizer.add_special_tokens({'pad_token': '[PAD]'})
   model.resize_token_embeddings(len(tokenizer))

#need to convert to format model can use in prompt-response
def format_ds(example):
  Symptoms = example["Symptoms"]
  Disease = example["Name"]
  Treatments = example["Treatments"]

  prompt = f"Find the disease and recommended treatments given the symptoms of a particular disease: {Symptoms}"
  response = f"You might have {example['Name']}. The treatments are: {example['Treatments']}"

  #learning part i think
  #tokenzize seperately, turns each string into list of token ids
  prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids #wont add special tokens
  response_ids = tokenizer(response, add_special_tokens=True).input_ids # to add special tokens like EOS if model expects it
  #so will look like an array of ints
  #concatenate the prompt and response ids now
  input_ids = prompt_ids + response_ids # full thing
  labels = [-100] * len(prompt_ids) + response_ids #-100 is used to ignore the prompt part during training, only train on response
  
  max_length = 512 # need to set for max input length (should be way less tho)
  #pad the input ids and labels to max length or else truncate if too long
  input_ids = input_ids[:max_length] + [tokenizer.pad_token_id] * max(0, max_length - len(input_ids))
  labels = labels[:max_length] + [-100] * max(0, max_length - len(labels)) #pad with -100 to ignore prompt part
  #attention mask tells which tokens are real and which are padding when it attends to them
  attention_mask = [1 if i != tokenizer.pad_token_id else 0 for i in input_ids] #1 for real tokens, 0 for padding
  #return dictionary with all these fields that TRAINER will use
  return {'input_ids': input_ids, 'attention_mask': attention_mask,
          'labels': labels}   

#dataset = ds.map(format_ds)

#for tokenizing dataset
#def tokenize_function(examples):
   # return tokenizer(examples["text"], padding = "max_length", truncation = True)
tokenized_dataset = ds.map(format_ds) #batched = True)

#create data collator
data_collator = default_data_collator#DataCollatorForLanguageModeling(tokenizer = tokenizer, mlm = False)


#import accuracy eval metric
#DONT NEED??????????????

#define training args
training_args = TrainingArguments(
  output_dir = "./results", #results folder will be created during process with steps saved
  per_device_train_batch_size = 2,
  num_train_epochs = 3,
  logging_steps = 10,
  save_steps = 50,
  save_total_limit = 1,
  #optim = "adamw_torch",
  learning_rate = 1e-5,
  fp16 = False, #not using GPU...
  #evaluation_strategy = "no" #no eval during training needed?? wouldn't be recognized by transformer module
)
#print("TrainingArguments works.")
trainer = Trainer(
  model = model,
  args = training_args,
  train_dataset = tokenized_dataset['train'],
  tokenizer = tokenizer,
  data_collator = default_data_collator
)

trainer.train()

#dir will be created when training ends and has model and tokenizer stuff
model.save_pretrained("./custom_finetuned_model")
tokenizer.save_pretrained("./custom_finetuned_model")

#load model..
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device) #move model to device
model.eval() #eval mode

while True:
  user_input = input("You: ")
  if (user_input.lower() == "exit"): #keep chatting until user says exit
    break
  #this will be the chat prompt that gets prepended to the user input about symptoms so that this is the format it expects (start out formal)
  chat_prompt = f"Find the disease and recommended treatments given the symptoms of a particular disease: {user_input}\n"
  inputs = tokenizer(chat_prompt, return_tensors = "pt").to(device)

  with torch.no_grad():
     outputs = model.generate(
        **inputs,
        max_length = 150, #max length of the generated response
        #early_stopping = True, #stop when we reach the end of the sentence
        temperature = 0.7, #how random response is but really should be cuz just matching to data
        top_k = 50, #top k sampling ???
        top_p = 0.95, #top p sampling ???
        pad_token_id = tokenizer.eos_token_id #pad token id to use
     )

  #chat prompt has recent exchanges and prompt to help generate response all in one string tho
  response = tokenizer.decode(outputs[0], skip_special_tokens = True)
  #response has recent exchanges+ its response so need to strip the first part for just the answer
  answer = response[len(chat_prompt):].strip().split("\n")[0]
  
  print("Chatbot: ", answer)
