import math
import operator
import sys
import json
from functools import reduce 

def count_ngram(candidate, references, n):
    clipped_count = 0
    count = 0
    r = 0
    c = 0
    for si in range(len(candidate)):
        ref_counts = []
        ref_lengths = []
        for reference in references:
            ref_sentence = reference[si]
            ngram_d = {}
            words = ref_sentence.strip().split()
            ref_lengths.append(len(words))
            limits = len(words) - n + 1
            
            for i in range(limits):
                ngram = ' '.join(words[i:i+n]).lower()
                ngram_d[ngram] = ngram_d.get(ngram, 0) + 1
            
            ref_counts.append(ngram_d)
        
        cand_sentence = candidate[si]
        cand_dict = {}
        words = cand_sentence.strip().split()
        limits = len(words) - n + 1
        
        for i in range(limits):
            ngram = ' '.join(words[i:i + n]).lower()
            cand_dict[ngram] = cand_dict.get(ngram, 0) + 1
            
        clipped_count += clip_count(cand_dict, ref_counts)
        count += limits
        r += best_length_match(ref_lengths, len(words))
        c += len(words)
        
    pr = clipped_count / count if clipped_count > 0 else 0
    bp = brevity_penalty(c, r)
    return pr, bp

def clip_count(cand_d, ref_ds):
    count = 0
    for m in cand_d.keys():
        m_w = cand_d[m]
        # Check if the n-gram exists in any reference captions
        m_max = max((ref.get(m, 0) for ref in ref_ds), default=0)
        count += min(m_w, m_max)
    return count

def best_length_match(ref_l, cand_l):
    return min(ref_l, key=lambda ref: abs(cand_l - ref))

def brevity_penalty(c, r):
    return 1 if c > r else math.exp(1 - (r / c))

def geometric_mean(precisions):
    return (reduce(operator.mul, precisions)) ** (1.0 / len(precisions))

def BLEU(s, t, flag=False):
    score = 0.  
    candidate = [s.strip()]
    references = [[t[i].strip()] for i in range(len(t))] if flag else [[t.strip()]]
    precisions = []
    pr, bp = count_ngram(candidate, references, 1)
    precisions.append(pr)
    score = geometric_mean(precisions) * bp
    return score

if __name__ == "__main__":
    test_json = "MLDS_hw2_1_data/testing_label.json"
    output = sys.argv[1]
    result = {}
    
    with open(output, 'r') as f:
        for line in f:
            line = line.rstrip()
            comma = line.index(',')
            test_id = line[:comma]
            caption = line[comma+1:]
            result[test_id] = caption

    bleu = []
    
    # Load test reference data
    with open(test_json, 'r') as f:
        test_data = json.load(f)

    # Calculate BLEU scores
    for item in test_data:
        captions = [x.rstrip('.') for x in item['caption']]
        score = BLEU(result[item['id']], captions, True)
        bleu.append(score)

    average_bleu = sum(bleu) / len(bleu) if bleu else 0
    print("Average BLEU score is " + str(average_bleu))
