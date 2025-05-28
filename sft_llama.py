
#find a dataset to download and use SFT to learn it

# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorWithPadding)
import torch
import numpy as np
import evaluate


model_name = "MBZUAI/MobiLlama-05B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

#dataset from hugging face to SFT on (diseases and symptoms)
from datasets import load_dataset

ds = load_dataset("QuyenAnhDE/Diseases_Symptoms") #400 rows



"""model.to('cuda')
text = "I was walking towards the river when "
input_ids = tokenizer(text, return_tensors="pt").to('cuda').input_ids
outputs = model.generate(input_ids, max_length=1000, repetition_penalty=1.2, pad_token_id=tokenizer.eos_token_id)
print(tokenizer.batch_decode(outputs[:, input_ids.shape[1]:-1])[0].strip())"""



#for tokenizing dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding = "max_length", truncation = True)

tokenized_dataset = ds.map(tokenize_function, batched = True)

if(tokenizer.pad_token is None):
   tokenizer.add_special_tokens({'pad_token': '[PAD]'})
   model.resize_token_embeddings(len(tokenizer))
  
tokenized_dataset = dataset.map(tokenize_function, batched = True)

#create data collator
date_collator = DataCollatorWithPadding(tokenizer = tokenizer)


#import accuracy eval metric
accuracy = evaluate.load("accuracy")

#define eval func to pass to trainer later
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis = 1)

    return {"accuracy": accuracy.compute(predictions = predictions, references = labels)}




while True:
  user_input = input("You: ")
  if (user_input.lower() == "exit"): #keep chatting until user says exit
    break
  #add their response to array of history
  recent_exchanges.append(f"You: {user_input}")
  recent_exchanges = recent_exchanges[-3:] #keep most recent 3
  chat_prompt = "\n".join(recent_exchanges) + "\nChatbot:"
  #storing in array should look like this 
  #{You: Hello, Chatbot: Hi I'm a chatbot!, You: Bye}

  #chat prompt has recent exchanges and prompt to help generate response all in one string tho
  response = generator(chat_prompt, max_length = max_length, temperature = temperature, truncation = True)[0]['generated_text'] #set higher max length ??
  #response has recent exchanges+ its response so need to strip the first part for just the answer
  answer = response[len(chat_prompt):].strip().split("\n")[0]
  
  print("Chatbot: ", answer)
  recent_exchanges.append(f"Chatbot: {answer}")
