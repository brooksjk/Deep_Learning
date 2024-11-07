import os
import json
import re
import string
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizerFast, BertForQuestionAnswering, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
from accelerate import Accelerator
from collections import Counter
from typing import List, Tuple, Dict

# configs
ROOT_DIR = '/scratch/jkbrook/Deep_Learning/HW_3'
BATCH_SIZE = 8
EPOCHS = 5
LEARNING_RATE = 1e-5
MODEL_PATH = os.path.join(ROOT_DIR, 'squad_qa_model')

class SquadDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings['input_ids'])

# read and parse SQuAD files
def read_squad_file(path: str) -> Tuple[List[str], List[str], List[Dict[str, str]]]:
    with open(path, 'rb') as f:
        squad_dict = json.load(f)

    contexts, questions, answers = [], [], []

    for group in squad_dict['data']:
        for passage in group['paragraphs']:
            context = passage['context']
            for qa in passage['qas']:
                question = qa['question']
                access = 'plausible_answers' if 'plausible_answers' in qa else 'answers'
                for answer in qa[access]:
                    contexts.append(context)
                    questions.append(question)
                    answers.append(answer)
    return contexts, questions, answers

# adjust end indices to ensure they match the answer span in context
def adjust_answer_indices(answers: List[Dict], contexts: List[str]):
    for answer, context in zip(answers, contexts):
        expected_text = answer['text']
        start_idx = answer['answer_start']
        end_idx = start_idx + len(expected_text)
        
        if context[start_idx:end_idx] != expected_text:
            for n in [1, 2]:  # Try to adjust up to 2 characters
                if context[start_idx - n:end_idx - n] == expected_text:
                    answer['answer_start'] = start_idx - n
                    answer['answer_end'] = end_idx - n
                    break
        else:
            answer['answer_end'] = end_idx

# tokenize and compute start and end positions
def tokenize_with_positions(encodings, answers):

    start_positions = []
    end_positions = []
    
    for i in range(len(answers)):
        
        start_positions.append(encodings.char_to_token(i, answers[i]['answer_start']))
        end_positions.append(encodings.char_to_token(i, answers[i]['answer_end']))

        if start_positions[-1] is None:
            start_positions[-1] = tokenizer.model_max_length
        
        shift = 1
        
        while end_positions[-1] is None:
            end_positions[-1] = encodings.char_to_token(i, answers[i]['answer_end'] - shift)
            shift += 1
    # update our encodings object with the new token-based start/end positions
    encodings.update({'start_positions': start_positions, 'end_positions': end_positions})

# training loop
def train_model(model, train_loader, optimizer, scheduler, num_epochs):
    for epoch in range(num_epochs):
        epoch_loss = 0
        model.train()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        
        for batch in progress_bar:
            optimizer.zero_grad()
            input_ids, attention_mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)
            start_positions, end_positions = batch['start_positions'].to(device), batch['end_positions'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions)
            loss = outputs[0]
            epoch_loss += loss
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            progress_bar.set_postfix({"Loss": loss.item()})

def test_model(model, test_loader, num_epochs):
    
    accuracy, predictions, references = [], [], []
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        progress_bar = tqdm(test_loader,  desc=f"Epoch {epoch + 1}/{num_epochs}")
    
        for batch in progress_bar:
            with torch.no_grad():
    
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
    
                start_true = batch['start_positions'].to(device)
                end_true = batch['end_positions'].to(device)
    
                outputs = model(input_ids, attention_mask=attention_mask)
    
                start_pred = torch.argmax(outputs['start_logits'], dim=1)
                end_pred = torch.argmax(outputs['end_logits'], dim=1)
                
                accuracy.append(((start_pred == start_true).sum()/len(start_pred)).item())
                accuracy.append(((end_pred == end_true).sum()/len(end_pred)).item())
                
                for i in range(start_pred.shape[0]):
                    all_tokens = tokenizer.convert_ids_to_tokens(batch['input_ids'][i])
                    prediction = ' '.join(all_tokens[start_pred[i] : end_pred[i]+1])
                    ref = ' '.join(all_tokens[start_true[i] : end_true[i]+1])
                    pred_ids = tokenizer.convert_tokens_to_ids(prediction.split())
                    prediction = tokenizer.decode(pred_ids)
                    predictions.append(prediction)
                    references.append(ref)

    return accuracy, predictions, references

# evaluation functions
def normalize_answer(s):
    return ' '.join(re.sub(r'[^\w\s]|_', '', s.lower()).split())

def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)

def f1_score(prediction, ground_truth):
    pred_tokens, gt_tokens = normalize_answer(prediction).split(), normalize_answer(ground_truth).split()
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision, recall = num_same / len(pred_tokens), num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)

def get_predictions(outputs, inputs, tokenizer):
    start_preds = torch.argmax(outputs.start_logits, dim=1)
    end_preds = torch.argmax(outputs.end_logits, dim=1)
    
    predictions = []
    
    for i in range(len(start_preds)):
        
        input_ids = inputs['input_ids'][i]
        answer_tokens = input_ids[start_preds[i] : end_preds[i] + 1]
        answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)
        
        predictions.append(answer)
        
    return predictions

def get_references(inputs, start_true, end_true, tokenizer):
    references = []
    
    for i in range(len(start_true)):
        # Extract the true start and end positions for each sample
        true_start = start_true[i].item()
        true_end = end_true[i].item()
        
        # Get the token IDs for the reference answer
        reference_tokens = inputs['input_ids'][i][true_start : true_end + 1]
        reference = tokenizer.decode(reference_tokens, skip_special_tokens=True)
        
        references.append(reference)
        
    return references

def evaluate(predictions, references):
    f1, exact_match = 0, 0
    for pred, ref in zip(predictions, references):
        exact_match += exact_match_score(pred, ref)
        f1 += f1_score(pred, ref)
    return {
        'f1': f1 / len(predictions) * 100 if predictions else 0,
        'exact_match': exact_match / len(predictions) * 100 if predictions else 0
    }

def custom_collate_fn(batch):
    input_ids = torch.stack([torch.tensor(item['input_ids']) for item in batch])
    attention_mask = torch.stack([torch.tensor(item['attention_mask']) for item in batch])
    start_positions = torch.tensor([item['start_positions'] for item in batch])
    end_positions = torch.tensor([item['end_positions'] for item in batch])
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'start_positions': start_positions,
        'end_positions': end_positions
    }

# Main execution
if __name__ == '__main__':
    accelerator = Accelerator()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load data
    train_contexts, train_questions, train_answers = read_squad_file(os.path.join(ROOT_DIR, 'squad_files/spoken_train-v1.1.json'))
    test_contexts, test_questions, test_answers = read_squad_file(os.path.join(ROOT_DIR, 'squad_files/spoken_test-v1.1.json'))
    adjust_answer_indices(train_answers, train_contexts)
    adjust_answer_indices(test_answers, test_contexts)

    # tokenize and prepare data
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-multilingual-uncased')
    train_encodings = tokenizer(train_contexts, train_questions, truncation=True, padding=True)
    test_encodings = tokenizer(test_contexts, test_questions, truncation=True, padding=True)

    tokenize_with_positions(train_encodings, train_answers)
    tokenize_with_positions(test_encodings, test_answers)

    # load model, optimizer, scheduler
    model = BertForQuestionAnswering.from_pretrained('bert-base-multilingual-uncased').to(device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    train_loader = DataLoader(SquadDataset(train_encodings), batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.1 * EPOCHS * len(train_loader), num_training_steps=EPOCHS * len(train_loader))
    model, optimizer, train_loader, scheduler = accelerator.prepare(model, optimizer, train_loader, scheduler)

    # train model
    train_model(model, train_loader, optimizer, scheduler, EPOCHS)

    # save model
    model.save_pretrained(MODEL_PATH)
    tokenizer.save_pretrained(MODEL_PATH)

    model = BertForQuestionAnswering.from_pretrained(MODEL_PATH)
    model.to(device)

    # evaluate on test data
    model.eval()

    test_loader = DataLoader(SquadDataset(test_encodings), batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn)

    accuracy, predictions, references = test_model(model, test_loader, EPOCHS)

    # calculate evaluation metrics
    results = evaluate(predictions, references)
    print("Evaluation Results:", results)

    




