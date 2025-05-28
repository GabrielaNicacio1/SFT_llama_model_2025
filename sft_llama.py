
#find a dataset to download and use SFT to learn it
#supervised fine tuning

# Use a pipeline as a high-level helper
from transformers import pipeline

#dataset from hugging face to SFT on (diseases and symptoms)
from datasets import load_dataset

ds = load_dataset("QuyenAnhDE/Diseases_Symptoms")


pipe = pipeline("text-generation", model="MBZUAI/MobiLlama-05B", trust_remote_code=True)

# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("MBZUAI/MobiLlama-05B", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("MBZUAI/MobiLlama-05B", trust_remote_code=True)



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
