#pip install trl

from trl import DPOTrainer, DPOConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, default_data_collator, DataCollatorWithPadding, BitsAndBytesConfig
import torch
from datasets import Dataset, load_dataset
from huggingface_hub import login
import os
import pandas as pd
from peft import LoraConfig, get_peft_model, TaskType
from datasets import DatasetDict
import json
#print(inspect.signature(DPOTrainer.__init__))
#notebook_login()
login(token = "")
os.environ["WANDB_DISABLED"] = "true" #disable wandb logging dont use

model_name = "meta-llama/Llama-3.2-1B-Instruct"
#Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)#, padding_side="right")
model = AutoModelForCausalLM.from_pretrained(model_name)


#baseline for comparison - frozen model. No lora
ref_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    #load_in_8bit=True,
    device_map="auto",
    #quantization_config=bnb_config,
)

ref_model.eval()
for param in ref_model.parameters():
    param.requires_grad = False


#load dpo data set
dpo_data = load_dataset("json", data_files={"train": "dpo_data.jsonl"})


if(tokenizer.pad_token is None):
# tokenizer.add_special_tokens({'pad_token': '[PAD]'})
  tokenizer.pad_token = tokenizer.eos_token
  model.resize_token_embeddings(len(tokenizer))




def build_symptom_to_answer(dataset):
    mapping = {}
    for example in dataset:
        prompt = example["prompt"]
        # Add the prompt as-is
        mapping[prompt] = {
            "disease": example["chosen"].split("Disease:")[1].split(".")[0].strip(),
            "treatment": example["chosen"].split("Treatment:")[1].split(".")[0].strip()
        }
        # Also add the symptoms-only version
        symptoms = prompt.replace("Symptoms: ", "").strip()
        mapping[symptoms] = mapping[prompt]  # Same answer for both keys
    return mapping



def is_disease_match(generated, expected_entry):
    expected_disease = expected_entry["disease"].lower()
    synonyms = [s.lower() for s in expected_entry.get("synonyms", [])]
    generated_lower = generated.lower()
    return (generated_lower == expected_disease) or (generated_lower in synonyms)

def is_treatment_match(generated, expected):
    gen_treatments = [t.strip().lower() for t in generated.split(',')]
    exp_treatments = [t.strip().lower() for t in expected.split(',')]
    for g in gen_treatments:
        if g in exp_treatments:
            return True
    for e in exp_treatments:
        if e in generated.lower():
            return True
    return False

def clean_response(response):
    # Remove system and user prompts
    response = response.split("<[ASSIST]>")[-1]
    # Remove any remaining tags
    response = response.replace("<[SYS]>", "").replace("<[/SYS]>", "")
    response = response.replace("<[USER]>", "").replace("<[/USER]>", "")
    response = response.replace("<[ASSIST]>", "").replace("<[/ASSIST]>", "")
    # Trim whitespace and newlines
    return response.strip()




def extract_disease_and_treatment(text):
    # Try to find the first disease and treatment mentioned
    disease = ""
    treatment = ""
    # Simple approach: find the first disease and treatment after colons or in lists
    # This is a simple example and may need to be improved for your use case
    lines = text.split('\n')
    for line in lines:
        if "disease:" in line.lower() or "diseases:" in line.lower() or "possible disease:" in line.lower():
            disease = line.split(':')[-1].strip()
        if "treatment:" in line.lower() or "treatments:" in line.lower() or "possible treatment:" in line.lower():
            treatment = line.split(':')[-1].strip()
    # If not found, try to get the first disease and treatment from the text
    if not disease:
        # Very simple: get the first disease mentioned after a number or bullet
        for line in lines:
            if line.strip().startswith(('1.', '**', '-')) and ':' not in line:
                disease = line.split('.')[-1].split(':')[-1].strip()
                break
    if not treatment:
        # Very simple: get the first treatment mentioned after a disease
        for i, line in enumerate(lines):
            if disease and disease in line:
                if i+1 < len(lines):
                    treatment = lines[i+1].strip()
                    break
    return disease, treatment




def run_interactive_session_with_accuracy(model, tokenizer, device, symptom_to_answer):
    collected_data = []
    print("Start entering symptoms (as in the prompt). Type 'exit' to finish and see accuracy.\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        system_prompt = "<[SYS]> You are a helpful medical assistant. Respond with only the possible disease and treatments after being given the symptoms by the user, in this format: Disease: Treatment: <[/SYS]>\n"
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
        answer = clean_response(response)  # You can use the same clean_response function as before
        print("Chatbot:\n", answer)
        collected_data.append({"symptoms": user_input, "response": answer})

    disease_matches = 0
    treatment_matches = 0
    total = 0

    for entry in collected_data:
        user_input = entry["symptoms"]
        # Try to match with and without "Symptoms: "
        possible_keys = [user_input, "Symptoms: " + user_input]
        matched = False
        for key in possible_keys:
            if key in symptom_to_answer:
                expected = symptom_to_answer[key]
                matched = True
                break
        if not matched:
            print(f"Symptoms '{user_input}' not found in dataset. Skipping.")
            continue

        generated = entry["response"]
        generated_disease, generated_treatment = extract_disease_and_treatment(generated)

        # Compare with simple matching
        disease_match = is_disease_match(generated_disease, expected)
        treatment_match = is_treatment_match(generated_treatment, expected["treatment"])

        # Update counters
        if disease_match: disease_matches += 1
        if treatment_match: treatment_matches += 1
        total += 1

        print(f"\nSymptoms: {user_input}")
        print(f"Expected: Disease: {expected['disease']}, Treatment: {expected['treatment']}")
        print(f"Generated: Disease: {generated_disease}, Treatment: {generated_treatment}")
        print(f"Disease Match: {disease_match} | Treatment Match: {treatment_match}")

    if total == 0:
        print("No valid exchanges were evaluated. Make sure your input matches a prompt in the dataset.")
    else:
        print("\nEvaluation Results:")
        print(f"Total evaluated exchanges: {total}")
        print(f"Disease Accuracy: {disease_matches / total:.2%}")
        print(f"Treatment Accuracy: {treatment_matches / total:.2%}")




#to reduce gpu and mem use, and make faster
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], # Modules to apply LoRA to (common for transformer models)
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
)

peft_model_train = get_peft_model(model, lora_config)
#peft_model_train.print_trainable_parameters()
#use cpu if not enough gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
peft_model_train.to(device) #move model to device
#output_dir = "./llama3.2_disease_model"
#ref_model = AutoModelForCausalLM.from_pretrained(model_name).to(device) #baseline for comparison - frozen model. No lora

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
    beta = 0.1,
    padding_value = tokenizer.pad_token_id #padding value for collator
)

#data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
trainer = DPOTrainer(
  model = peft_model_train,
  ref_model = ref_model,
  #beta = 0.1,
  #args = training_args,
  args = dpo_config,
  train_dataset = dpo_data['train'],#tokenized_dataset['train'],
  #tokenizer = tokenizer,
  processing_class = tokenizer, #the tokenizer here
  #data_collator =default_data_collator
)

symptom_to_answer = build_symptom_to_answer(dpo_data["train"])

trainer.train()
peft_model_train.eval() #eval mode

#dir will be created when training ends and has model and tokenizer stuff
peft_model_train.save_pretrained("./llama3.2_dpo_disease_model")
tokenizer.save_pretrained("./llama3.2_dpo_disease_model")


run_interactive_session_with_accuracy(model, tokenizer, device, symptom_to_answer)


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

  print("Chatbot: \n", answer)'''
