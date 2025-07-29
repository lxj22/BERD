import json
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge
from collections import Counter
import math
import nltk
import os

# Download necessary NLTK data
# nltk.download('wordnet')
# nltk.download('punkt')

# Function to calculate BLEU score with different n-gram weights
def calculate_bleu(reference, hypothesis, n=4):
    """
    Calculate BLEU-1 to BLEU-n scores
    n: maximum n-gram to consider
    """
    if len(hypothesis) == 0:
        return [0] * n
    
    smoothie = SmoothingFunction().method1
    reference_tokens = nltk.word_tokenize(reference.lower())
    hypothesis_tokens = nltk.word_tokenize(hypothesis.lower())
    
    bleu_scores = []
    for i in range(1, n+1):
        # Create weights for different n-grams
        weights = tuple([1.0/i if j < i else 0 for j in range(n)])
        
        score = sentence_bleu([reference_tokens], hypothesis_tokens, 
                             weights=weights, smoothing_function=smoothie)
        bleu_scores.append(score)
    
    return bleu_scores

# Function to calculate METEOR score - fixed to use tokenized inputs
def calculate_meteor(reference, hypothesis):
    if len(hypothesis) == 0:
        return 0
    
    # Tokenize the strings
    reference_tokens = nltk.word_tokenize(reference.lower())
    hypothesis_tokens = nltk.word_tokenize(hypothesis.lower())
    
    # Call meteor_score with tokenized inputs
    try:
        return meteor_score([reference_tokens], hypothesis_tokens)
    except Exception as e:
        print(f"METEOR error: {e} for ref={reference}, hyp={hypothesis}")
        return 0

# Function to calculate ROUGE scores
def calculate_rouge(reference, hypothesis):
    if len(hypothesis) == 0:
        return {'rouge-1': {'f': 0}, 'rouge-2': {'f': 0}, 'rouge-l': {'f': 0}}
    
    rouge = Rouge()
    try:
        scores = rouge.get_scores(hypothesis, reference)[0]
        return scores
    except Exception as e:
        print(f"ROUGE error: {e} for ref={reference}, hyp={hypothesis}")
        return {'rouge-1': {'f': 0}, 'rouge-2': {'f': 0}, 'rouge-l': {'f': 0}}

# Function to calculate CIDEr score
def calculate_cider(references, hypotheses, n=4):
    def compute_cider_score(reference_ngrams, hypothesis_ngrams, document_frequency, ref_len, hyp_len):
        score = 0.0
        for n_gram in range(1, n+1):
            # If hypothesis is empty, CIDEr score is 0
            if len(hypothesis_ngrams[n_gram]) == 0:
                continue

            # Compute TF-IDF for the hypothesis
            hypothesis_count = Counter(hypothesis_ngrams[n_gram])
            denominator_hyp = max(1, float(hyp_len - n_gram + 1))  # Avoid division by zero
            hypothesis_count = {k: v / denominator_hyp for k, v in hypothesis_count.items()}
            
            # Compute TF-IDF for the reference
            reference_count = Counter(reference_ngrams[n_gram])
            denominator_ref = max(1, float(ref_len - n_gram + 1))  # Avoid division by zero
            reference_count = {k: v / denominator_ref for k, v in reference_count.items()}
            
            # Compute cosine similarity
            keys = list(set(list(reference_count.keys()) + list(hypothesis_count.keys())))
            
            # Calculate IDF values
            num_docs = len(references)
            idf_values = {}
            for key in keys:
                if key in document_frequency[n_gram]:
                    idf_values[key] = math.log(num_docs / (1.0 + document_frequency[n_gram][key]))
                else:
                    idf_values[key] = math.log(num_docs)
            
            # Compute vectors with TF-IDF weights
            vec_ref = np.zeros(len(keys))
            vec_hyp = np.zeros(len(keys))
            
            for i, key in enumerate(keys):
                if key in reference_count:
                    vec_ref[i] = reference_count[key] * idf_values[key]
                if key in hypothesis_count:
                    vec_hyp[i] = hypothesis_count[key] * idf_values[key]
            
            # Calculate cosine similarity
            norm_ref = np.linalg.norm(vec_ref)
            norm_hyp = np.linalg.norm(vec_hyp)
            
            if norm_ref > 0 and norm_hyp > 0:
                cos_sim = np.dot(vec_ref, vec_hyp) / (norm_ref * norm_hyp)
            else:
                cos_sim = 0
            
            score += cos_sim
        
        # Average over n-grams
        if n > 0:
            score /= n
        
        return score
    
    # Extract n-grams from all references and hypotheses
    def extract_ngrams(sentence, n):
        tokens = nltk.word_tokenize(sentence.lower())
        ngrams = {}
        for i in range(1, n+1):
            ngrams[i] = []
            for j in range(len(tokens) - i + 1):
                ngram = tuple(tokens[j:j+i])
                ngrams[i].append(ngram)
        return ngrams, len(tokens)
    
    # Calculate document frequency for all n-grams in references
    document_frequency = {i: {} for i in range(1, n+1)}
    
    reference_ngrams_list = []
    reference_lengths = []
    
    for reference in references:
        reference_ngrams, ref_len = extract_ngrams(reference, n)
        reference_ngrams_list.append(reference_ngrams)
        reference_lengths.append(ref_len)
        
        # Update document frequency
        for i in range(1, n+1):
            for ngram in set(reference_ngrams[i]):
                if ngram in document_frequency[i]:
                    document_frequency[i][ngram] += 1
                else:
                    document_frequency[i][ngram] = 1
    
    # Calculate CIDEr scores for each hypothesis
    cider_scores = []
    
    for i, hypothesis in enumerate(hypotheses):
        try:
            hypothesis_ngrams, hyp_len = extract_ngrams(hypothesis, n)
            score = compute_cider_score(reference_ngrams_list[i], hypothesis_ngrams, 
                                        document_frequency, reference_lengths[i], hyp_len)
            cider_scores.append(score)
        except Exception as e:
            print(f"CIDEr error: {e} for ref={references[i]}, hyp={hypothesis}")
            cider_scores.append(0)
    
    # Return average CIDEr score
    if len(cider_scores) > 0:
        return sum(cider_scores) / len(cider_scores)
    else:
        return 0

# Load data from JSON file
def load_data_from_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return []

# Main function to calculate all metrics
def calculate_metrics(data):
    predictions = [item['pred'] for item in data]
    ground_truths = [item['gt'] for item in data]
    
    print(f"Sample data (first 3 items):")
    for i in range(min(3, len(data))):
        print(f"GT: {ground_truths[i]}")
        print(f"Pred: {predictions[i]}")
        print("---")
    
    # Calculate BLEU-1 to BLEU-4 scores
    bleu_scores = [calculate_bleu(gt, pred) for gt, pred in zip(ground_truths, predictions)]
    
    # Average BLEU-1 to BLEU-4 scores
    avg_bleu1 = sum([scores[0] for scores in bleu_scores]) / len(bleu_scores) if bleu_scores else 0
    avg_bleu2 = sum([scores[1] for scores in bleu_scores]) / len(bleu_scores) if bleu_scores else 0
    avg_bleu3 = sum([scores[2] for scores in bleu_scores]) / len(bleu_scores) if bleu_scores else 0
    avg_bleu4 = sum([scores[3] for scores in bleu_scores]) / len(bleu_scores) if bleu_scores else 0
    
    print("BLEU scores calculated")
    
    # Calculate METEOR scores
    meteor_scores = [calculate_meteor(gt, pred) for gt, pred in zip(ground_truths, predictions)]
    avg_meteor = sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0
    
    print("METEOR scores calculated")
    
    # Calculate ROUGE scores
    rouge_scores = [calculate_rouge(gt, pred) for gt, pred in zip(ground_truths, predictions)]
    avg_rouge1 = sum(score['rouge-1']['f'] for score in rouge_scores) / len(rouge_scores) if rouge_scores else 0
    avg_rouge2 = sum(score['rouge-2']['f'] for score in rouge_scores) / len(rouge_scores) if rouge_scores else 0
    avg_rougeL = sum(score['rouge-l']['f'] for score in rouge_scores) / len(rouge_scores) if rouge_scores else 0
    
    print("ROUGE scores calculated")
    
    # Calculate CIDEr score
    avg_cider = calculate_cider(ground_truths, predictions)
    
    print("CIDEr score calculated")
    
    # Return all metrics
    metrics = {
        'BLEU-1': avg_bleu1,
        'BLEU-2': avg_bleu2,
        'BLEU-3': avg_bleu3,
        'BLEU-4': avg_bleu4,
        'METEOR': avg_meteor,
        'ROUGE-1': avg_rouge1,
        'ROUGE-2': avg_rouge2,
        'ROUGE-L': avg_rougeL,
        'CIDEr': avg_cider
    }
    
    return metrics

# Specify the JSON file path
file_path = "/data/xingjian_luo/project/zhongshanyi-dataset/eval_result_qwen2_5.json"

# Check if file exists
if not os.path.exists(file_path):
    print(f"Error: File not found at {file_path}")
else:
    # Load data from file
    data = load_data_from_file(file_path)
    
    # Process the data and calculate metrics
    if data:
        print(f"Loaded {len(data)} samples from {file_path}")
        metrics = calculate_metrics(data)
        
        # Print the results
        print("\nEvaluation Metrics:")
        print("-" * 30)
        for metric_name, score in metrics.items():
            print(f"{metric_name}: {score:.4f}")
    else:
        print("No data loaded. Please check the file format.")