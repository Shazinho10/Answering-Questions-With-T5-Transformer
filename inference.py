import torch

def answer_generator(tokenizer, model, questions, context):
  input_ids = tokenizer(questions, context, return_tensors='pt').input_ids
  output = model.generate(input_ids)
  print('Answer: ', tokenizer.decode(output[0], skip_special_tokens=True))



