
#find a dataset to download and use SFT to learn it
#unsupervised fine tuning
import transformers
import torch

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]

outputs = pipeline(
    messages,
    max_new_tokens=256,
)
print(outputs[0]["generated_text"][-1])

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
