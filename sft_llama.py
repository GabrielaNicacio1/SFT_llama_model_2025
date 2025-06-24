from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, default_data_collator
import torch
from datasets import Dataset
from huggingface_hub import login
import os
import pandas as pd
from peft import LoraConfig, get_peft_model, TaskType
from datasets import DatasetDict
import numpy as np
import evaluate

#notebook_login()
login(token = "")
os.environ["WANDB_DISABLED"] = "true" #disable wandb logging dont use

#try whole new model cuz other had too many issues
model_name = "meta-llama/Llama-3.2-1B-Instruct"
#Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

#dataset from hugging face to SFT on (diseases and symptoms)
#using pandas
df = pd.read_csv("hf://datasets/QuyenAnhDE/Diseases_Symptoms/Diseases_Symptoms.csv")
ds = Dataset.from_pandas(df)
# Wrap the single dataset as a DatasetDict with a 'train' split

ds = DatasetDict({"train": ds})
ds["train"] = ds["train"].select(range(400))  # start w/ 100 samples

if(tokenizer.pad_token is None):
   tokenizer.add_special_tokens({'pad_token': '[PAD]'})
   model.resize_token_embeddings(len(tokenizer))


symptom_to_answer = {}
for example in ds['train']:
    symptom_to_answer[example["Symptoms"]] = {
        "disease": example["Name"],
        "treatment": example["Treatments"]
    }


#need to convert to format model can use in prompt-response
def format_ds(example):
  Symptoms = example["Symptoms"]
  Disease = example["Name"]
  Treatments = example["Treatments"]

  #instructions given to the model
  system_prompt ="<[SYS]> You are a helpful medical assistant. Respond with only the possible disease and treatments after being given the symptoms by the user. <[/SYS]>\n"
  #knowledgable about diseases, their symptoms, and how to treat them. Given symptoms from the user, find only the disease and recommended treatments. Be simple and accurate. <[/SYS]>"

  #start end tags
  user_prompt = f"<[USER]> Symptoms: {Symptoms} <[/USER]>\n" #input from user
  assist = f"<[ASSIST]>Disease: {Disease}\nTreatments: {Treatments} <[/ASSIST]>\n" #desired output format
  full_system_prompt = system_prompt + user_prompt #+ assist   #put together all prompts


  #training on response only??
  #tokenzize seperately, turns each string into list of token ids
  prompt_ids = tokenizer(full_system_prompt, add_special_tokens=False).input_ids #wont add special tokens
  response_ids = tokenizer(assist, add_special_tokens=True).input_ids # to add special tokens like EOS if model expects it
  #so will look like an array of ints
  #concatenate the prompt and response ids now
  input_ids = prompt_ids + response_ids # full thing
  labels = [-100] * len(prompt_ids) + response_ids #-100 is used to ignore the prompt part during training, only train on response

  max_length = 200 # need to set for max input length (should be way less tho?)
  #pad the input ids and labels to max length or else truncate if too long
  input_ids = input_ids[:max_length] + [tokenizer.pad_token_id] * max(0, max_length - len(input_ids))
  labels = labels[:max_length] + [-100] * max(0, max_length - len(labels)) #pad with -100 to ignore prompt part
  #attention mask tells which tokens are real and which are padding when it attends to them
  attention_mask = [1 if i != tokenizer.pad_token_id else 0 for i in input_ids] #1 for real tokens, 0 for padding
  #return dictionary with all these fields that TRAINER will use
  return {'input_ids': input_ids, 'attention_mask': attention_mask,
          'labels': labels}




def clean_response(response):
    # Remove system and user prompts
    response = response.split("<[ASSIST]>")[-1]
    # Remove any remaining tags
    response = response.replace("<[SYS]>", "").replace("<[/SYS]>", "")
    response = response.replace("<[USER]>", "").replace("<[/USER]>", "")
    response = response.replace("<[ASSIST]>", "").replace("<[/ASSIST]>", "")
    # Trim whitespace and newlines
    return response.strip()



from difflib import SequenceMatcher

def is_similar(a, b, threshold=0.3):
    """Check if two strings are similar, using difflib."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio() >= threshold

def is_treatment_match(generated, expected, threshold=0.3):
    """Check if any generated treatment matches any expected treatment."""
    # Split treatments into lists, handling commas and parentheses
    gen_treatments = []
    for t in generated.split(','):
        t = t.strip().lower()
        # Remove anything in parentheses (optional)
        t = t.split('(')[0].strip()
        gen_treatments.append(t)
    exp_treatments = []
    for t in expected.split(','):
        t = t.strip().lower()
        exp_treatments.append(t)
    # Check if any generated treatment matches any expected treatment
    for g in gen_treatments:
        for e in exp_treatments:
            if is_similar(g, e, threshold):
                return True
    return False


#function to get accuracy
def run_interactive_session_with_accuracy(model, tokenizer, device, symptom_to_answer):
    collected_data = []
    print("Start entering symptoms. Type 'exit' to finish and see accuracy.\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        system_prompt = "<[SYS]> You are a helpful medical assistant. Respond with only the possible disease and treatments after being given the symptoms by the user. <[/SYS]>\n"
        user_prompt = f"<[USER]> Symptoms: {user_input} <[/USER]>\n"
        full_prompt = system_prompt + user_prompt + "<[ASSIST]>"

        inputs = tokenizer(full_prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=1064,
                do_sample=True,
                temperature=0.2,
                top_k=30,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id
            )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        '''if "<[ASSIST]>" in response and "<[/ASSIST]>" in response:
            answer = response.split("<[ASSIST]>")[1].split("<[/ASSIST]>")[0].strip()
        else:
            answer = response.strip()'''
        answer = clean_response(response)
        print("Chatbot:\n", answer)
        collected_data.append({"symptoms": user_input, "response": answer})

    disease_matches = 0
    treatment_matches = 0
    total = 0

    for entry in collected_data:
        user_input = entry["symptoms"]
        generated = entry["response"]
        if user_input not in symptom_to_answer:
            print(f"Symptoms '{user_input}' not found in dataset. Skipping.")
            continue
        expected = symptom_to_answer[user_input]

        # Extract disease and treatment from generated response
        generated_disease = ""
        generated_treatment = ""
        if "Possible Disease:" in generated:
            generated_disease = generated.split("Disease:")[1].split("\n")[0].strip()
        if "Possible Treatments:" in generated:
            generated_treatment = generated.split("Treatments:")[1].strip()("<")[0].strip()  # Remove <[/ASSIST]>


        # Compare with fuzzy matching
        disease_match = is_similar(generated_disease, expected["disease"])
        treatment_match = is_treatment_match(generated_treatment, expected["treatment"])

        # Update counters
        if disease_match: disease_matches += 1
        if treatment_match: treatment_matches += 1
        total += 1

        print(f"\nSymptoms: {user_input}")
        print(f"Expected: Disease: {expected['disease']}, Treatment: {expected['treatment']}")
        print(f"Generated: {generated}")
        print(f"Disease Match: {disease_match} | Treatment Match: {treatment_match}")

    print("\nEvaluation Results:")
    print(f"Total evaluated exchanges: {total}")
    print(f"Disease Accuracy: {disease_matches / total:.2%}")
    print(f"Treatment Accuracy: {treatment_matches / total:.2%}")

#dataset = ds.map(format_ds)
#for tokenizing dataset
tokenized_dataset = ds.map(format_ds) #batched = True)

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
#define training args
training_args = TrainingArguments(
  output_dir = "./llama3.2_disease_model", #results folder will be created during process with steps saved
  per_device_train_batch_size = 1,
  gradient_accumulation_steps=4, #to have larger batch sizes, dont run out of mem so easily
  num_train_epochs = 3,
  logging_dir = "./sft_logs",
  logging_steps = 10,
  save_strategy= "no", #no saving during training, for now
  learning_rate = 1e-5,
  fp16 = True, # using GPU...
  #evaluation_strategy = "no" #no eval during training needed?? wouldn't be recognized by transformer module
)
#print("TrainingArguments works.")
trainer = Trainer(
  model = peft_model_train,
  args = training_args,
  train_dataset = tokenized_dataset['train'],
  tokenizer = tokenizer,
  data_collator = default_data_collator,
  eval_dataset=tokenized_dataset,
  #compute_metrics=compute_metrics
)


trainer.train()
peft_model_train.eval() #eval mode


#dir will be created when training ends and has model and tokenizer stuff
peft_model_train.save_pretrained("./llama3.2_disease_model")
tokenizer.save_pretrained("./llama3.2_disease_model")


run_interactive_session_with_accuracy(
    peft_model_train,  # or peft_model_train if you use LoRA
    tokenizer,
    device,
    symptom_to_answer
)

'''
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
  '''

'''if "<[ASSIST]>" in chat_response:
    answer = chat_response.split("<[ASSIST]>")[-1].strip()
  else:
    answer = chat_response.strip()'''
