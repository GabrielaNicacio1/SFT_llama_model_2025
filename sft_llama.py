
#find a dataset to download and use SFT to learn it

# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments, pipeline
import torch
import numpy as np
import evaluate


model_name = "MBZUAI/MobiLlama-05B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

#Sgenerator = pipeline("text-generation", model = model, tokenizer=tokenizer)
#dataset from hugging face to SFT on (diseases and symptoms)
from datasets import load_dataset

ds = load_dataset("QuyenAnhDE/Diseases_Symptoms") #400 rows

#need to convert to format model can use in prompt-response
def format_ds(example):
  symptoms = example["Symptoms"]
  disease = example["Name"]
  treatments = example["Treatments"]

  prompt = f"Find the disease and recommended treatments given the symptoms of a particular disease: {symptoms}"
  response = f"{disease}. {treatments}"

  return {"text": prompt + " " + response} #train model on that full text   

dataset = ds.map(format_ds)

"""model.to('cuda')
text = "I was walking towards the river when "
input_ids = tokenizer(text, return_tensors="pt").to('cuda').input_ids
outputs = model.generate(input_ids, max_length=1000, repetition_penalty=1.2, pad_token_id=tokenizer.eos_token_id)
print(tokenizer.batch_decode(outputs[:, input_ids.shape[1]:-1])[0].strip())"""



#for tokenizing dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding = "max_length", truncation = True)

if(tokenizer.pad_token is None):
   tokenizer.add_special_tokens({'pad_token': '[PAD]'})
   model.resize_token_embeddings(len(tokenizer))
  
tokenized_dataset = dataset.map(tokenize_function, batched = True)

#create data collator
data_collator = DataCollatorForLanguageModeling(tokenizer = tokenizer, mlm = False)


#import accuracy eval metric
#DONT NEED??????????????
'''accuracy = evaluate.load("accuracy")

#define eval func to pass to trainer later
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis = 1)

    return {"accuracy": accuracy.compute(predictions = predictions, references = labels)}
'''

#define training args
training_args = TrainingArguments(
  output_dir = "./results", #results folder will be created during process with steps saved
  per_device_train_batch_size = 2,
  num_train_epochs = 3,
  logging_steps = 10,
  save_steps = 50,
  save_total_limit = 1,
  #optim = "adamw_torch",
  learning_rate = 2e-5,
  fp16 = False, #not using GPU...
  evaluation_strategy = "no" #no eval during training needed??
)

trainer = Trainer(
  model = model,
  args = training_args,
  train_dataset = tokenized_dataset,
  tokenizer = tokenizer,
  data_collator = data_collator
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
  chat_prompt = "Find the disease and recommended treatments given the symptoms of a particular disease: {user_input}\n"
  inputs = tokenizer(chat_prompt, return_tensors = "pt").to(device)

  with torch.no_grad():
     outputs = model.generate(
        **inputs,
        max_length = 512, #max length of the generated response
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
