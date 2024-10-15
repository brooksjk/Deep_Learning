import sys
import json
import pickle
import torch
from torch.utils.data import DataLoader
from bleu_eval import BLEU
from training import test_data, test, Seq2SeqModel, Encoder, Decoder, Attention

# Load the trained model
model = torch.load('trained_model.h5', map_location=lambda storage, loc: storage)

# Load test data and create DataLoader
dataset = test_data('{}'.format(sys.argv[1]))
testing_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

# Load the index-to-word mapping
with open('i2w.pickle', 'rb') as handle:
    i2w = pickle.load(handle)

# Move the model to GPU if available
model = model.cuda()

# Get predictions using the test function
ss = test(testing_loader, model, i2w)

# Save the predictions to an output file
with open(sys.argv[2], 'w') as f:
    for id, s in ss:
        f.write('{},{}\n'.format(id, s))

# Load the reference captions for evaluation
test_file = json.load(open('/scratch/jkbrook/Deep_Learning/HW_2/MLDS_hw2_1_data/testing_label.json'))

# Load the generated captions from the output file
output = sys.argv[2]
result = {}

with open(output, 'r') as f:
    for line in f:
        line = line.rstrip()
        comma = line.index(',')
        test_id = line[:comma]
        caption = line[comma+1:]
        result[test_id] = caption

# Initialize BLEU scores list
bleu_scores = []

# Iterate over the test reference data
for item in test_file:
    test_id = item['id']
    
    # Ensure that the predicted caption exists for the test ID
    if test_id in result:
        predicted_caption = result[test_id]
        captions = [x.rstrip('.') for x in item['caption']]
        
        # Calculate BLEU score for the predicted caption and references
        bleu_score = BLEU(predicted_caption, captions, True)
        bleu_scores.append(bleu_score)
    else:
        print(f"Warning: No predicted caption found for ID {test_id}")

# Calculate the average BLEU score
if bleu_scores:
    avg_bleu = sum(bleu_scores) / len(bleu_scores)
else:
    avg_bleu = 0

print("Average BLEU score: " + str(avg_bleu))
