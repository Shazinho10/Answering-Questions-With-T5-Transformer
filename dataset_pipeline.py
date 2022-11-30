import torch
from torch.utils.data import Dataset
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class BioQADataset(Dataset):

  def __init__(self, data, tokenizer, 
               source_max_len=396, target_max_len=32):
    self.data = data
    self.tokenizer = tokenizer
    self.source_max_len = source_max_len
    self.target_max_len = target_max_len


  def __len__(self):
    return len(self.data)


  def __getitem__(self, idx):
    data_row = self.data.iloc[idx]

    source_encoding = self.tokenizer(data_row['questions'],
                                  data_row['context'],
                                  max_length=self.source_max_len,
                                  padding='max_length',
                                  truncation = 'only_second',
                                  return_attention_mask=True,
                                  add_special_tokens=True,
                                  return_tensors='pt')
    
    target_encoding = self.tokenizer(data_row['answers'],
                                max_length=self.target_max_len,
                                padding='max_length',
                                return_attention_mask=True,
                                truncation=True,
                                add_special_tokens=True,
                                return_tensors='pt')
    
    labels = target_encoding['input_ids']
    labels[labels == 0] = -100

    return {
        'question': data_row['questions'],
        'context': data_row['context'],
        'answer': data_row['answers'],
        'input_ids': source_encoding['input_ids'].flatten().to(device),
        'attention_mask': source_encoding['attention_mask'].flatten().to(device),
        'labels': labels.flatten().to(device)
    }

