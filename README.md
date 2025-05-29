# SFT_llama_model_2025
Using SFT to learn new data set with a version of llama from Hugging Face.

Using a dataset on diseases and symptoms. 400 rows in a csv.
Currently, was not able to test it, it got stuck at "Training" 0%.




Difference between supervised fine tuning and unsupervised fine tuning:
Supervised FT is when a model is trained on a certain chosen dataset so that the model becomes good at a specific task, whereas Unsupervised FT is when the model is trained on a large set of unlabeled data, just learning what it can.


Supervised FT is a process the model is additionally trained on a chosen dataset of input-output pairs. For the model to learn, x is the input/prompt, and y is the output/response desired. Before getting y, the model predicts what it should be and then compares the correct output with its own, adjusting its parameters to get closer to the desired response.

Pros: 
- have more control over what we want the model to learn
- can be a quicker way to train it for a specifc task
- because it has the right answers, it learns quickly what is the wrong thing to say

Cons:
- computationally more expensive, as well as the data sets, which can take a long time to make
- Model can focus too much specific examples in the data set and doesn't have enough broad understanding of things


Unsuperivised learning is when the the model is trained on a large set of unlabeled data, somewhat without direction maybe. It still works on predicitng the next token, and learning by comparing it with what its supposed to be, noticing patterns, but it works on general and kind of random understanding, not for a specific task.

Pros:
- Taking advantage of large unlabeled data sets, which can be easily to obtain
- Model can learn a wide range of topics

Cons:
- Can't lead model to learn specific desired tasks
- Large set of data used, so can take longer computationally, needing more resources


Noticed that BOTH can reinforce biases from the data





