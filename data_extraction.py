import json
import pandas as pd

def extract_data(json_path):
  with open(json_path) as f:
    data = json.load(f)

  paragraphs = [i for i in  data['data'][0]['paragraphs']]
  context = [paragraphs[i]['context'] for i in range(len(paragraphs))]
  qna = [paragraphs[i]['qas'] for i in range(len(paragraphs))]

  questions = [qna[i][0]['question'] for i in range(len(qna))]
  answers = [qna[i][0]['answers'][0]['text'] for i in range(len(qna))]

  df = pd.DataFrame(questions, columns=['questions'])
  df['context'] = context
  df['answers'] = answers

  return df




def extract_test_data(json_path):
  with open(json_path) as f:
    data = json.load(f)

  paragraphs = [i for i in  data['data'][0]['paragraphs']]
  context = [paragraphs[i]['context'] for i in range(len(paragraphs))]
  qna = [paragraphs[i]['qas'] for i in range(len(paragraphs))]

  questions = [qna[i][0]['question'] for i in range(len(qna))]
  #answers = [qna[i][0]['answers'][0]['text'] for i in range(len(qna))]

  df = pd.DataFrame(questions, columns=['questions'])
  df['context'] = context
  #df['answers'] = answers

  return df

