import sys
import json
import pickle
import torch
from torch.utils.data import DataLoader
from bleu_eval import BLEU
from training import test_data, test, Seq2SeqModel, Encoder, Decoder, Attention

model = torch.load('trained_model.h5', map_location=lambda storage, loc: storage)
#filepath = '/scratch/jkbrook/Deep_Learning/HW_2/MDLS_hw2_1_data/testing_data/feat'
dataset = test_data('{}'.format(sys.argv[1]))
testing_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

with open('i2w.pickle', 'rb') as handle:
    i2w = pickle.load(handle)

model = model.cuda()
ss = test(testing_loader, model, i2w)

with open(sys.argv[2], 'w') as f:
    for id, s in ss:
        f.write('{},{}\n'.format(id, s))

test_file = json.load(open('/scratch/jkbrook/Deep_Learning/HW_2/MDLS_hw2_1_data/testing_label.json'))
output = sys.argv[2]
result = {}

with open(output,'r') as f:
    
    for line in f:
        line = line.rstrip()
        comma = line.index(',')
        test_id = line[:comma]
        caption = line[comma+1:]
        result[test_id] = caption

bleu=[]

for item in test_file:
    score_per_video = []
    captions = [x.rstrip('.') for x in item['caption']]
    score_per_video.append(BLEU(result[item['id']],captions,True))
    bleu.append(score_per_video[0])

avg = sum(bleu) / len(bleu)

print("Average bleu score: " + str(avg))