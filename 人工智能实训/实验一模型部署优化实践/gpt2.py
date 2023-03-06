import torch
from transformers import BertTokenizer, BertForQuestionAnswering

tokenizer = BertTokenizer.from_pretrained('H:\\bert-base-uncased')
model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
question = "What is the capital of France?"
text = "France is a country in Europe. Its capital is Paris."
inputs = tokenizer.encode_plus(question, text, add_special_tokens=True, return_tensors="pt")

input_ids = inputs["input_ids"].tolist()[0]
attention_mask = inputs["attention_mask"].tolist()[0]
start_scores, end_scores = model(input_ids=torch.tensor(input_ids), attention_mask=torch.tensor(attention_mask))
start_index = torch.argmax(start_scores)
end_index = torch.argmax(end_scores)
answer_tokens = input_ids[start_index:end_index+1]
answer_tokens = tokenizer.convert_ids_to_tokens(answer_tokens, skip_special_tokens=True)
answer = tokenizer.convert_tokens_to_string(answer_tokens)
