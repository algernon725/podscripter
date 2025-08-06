#!/usr/bin/env python3
"""
Model comparison script for SentenceTransformer models used in punctuation restoration.
"""

import time
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def compare_models():
    """Compare different SentenceTransformer models for punctuation restoration."""
    
    models = {
        'all-MiniLM-L6-v2': 'Current model',
        'paraphrase-multilingual-MiniLM-L12-v2': 'Multilingual model',
        'all-mpnet-base-v2': 'High-quality model'
    }
    
    # Test sentences for comparison
    test_sentences = [
        "what time is the meeting tomorrow",
        "I need to prepare my presentation",
        "can you send me the agenda",
        "thank you for your help",
        "that was amazing",
        "I really appreciate it"
    ]
    
    # Question patterns for similarity testing
    question_patterns = [
        "What is this?",
        "Where are you?",
        "When will it happen?",
        "Why did you do that?",
        "How does it work?",
        "Who is there?",
        "Can you help me?",
        "Could you explain?"
    ]
    
    results = {}
    
    for model_name in models.keys():
        print(f"\n{'='*60}")
        print(f"Testing: {model_name}")
        print(f"Description: {models[model_name]}")
        print(f"{'='*60}")
        
        try:
            # Load model and measure loading time
            start_time = time.time()
            model = SentenceTransformer(model_name)
            load_time = time.time() - start_time
            
            print(f"Model loaded in: {load_time:.2f} seconds")
            
            # Get model info
            embedding_dim = model.get_sentence_embedding_dimension()
            print(f"Embedding dimension: {embedding_dim}")
            
            # Test encoding speed
            start_time = time.time()
            embeddings = model.encode(test_sentences)
            encode_time = time.time() - start_time
            
            print(f"Encoding {len(test_sentences)} sentences in: {encode_time:.2f} seconds")
            print(f"Average time per sentence: {encode_time/len(test_sentences):.4f} seconds")
            
            # Test question detection accuracy
            question_embeddings = model.encode(question_patterns)
            
            similarities = []
            for i, sentence in enumerate(test_sentences):
                sentence_emb = embeddings[i].reshape(1, -1)
                max_similarity = 0
                for j, pattern in enumerate(question_patterns):
                    pattern_emb = question_embeddings[j].reshape(1, -1)
                    similarity = cosine_similarity(sentence_emb, pattern_emb)[0][0]
                    max_similarity = max(max_similarity, similarity)
                similarities.append(max_similarity)
            
            # Calculate question detection accuracy
            expected_questions = [True, False, True, False, False, False]  # Based on test sentences
            detected_questions = [sim > 0.6 for sim in similarities]
            accuracy = sum(1 for exp, det in zip(expected_questions, detected_questions) if exp == det) / len(expected_questions)
            
            print(f"Question detection accuracy: {accuracy:.2%}")
            
            # Memory usage estimation (rough)
            model_size_mb = embedding_dim * 4 * 2 / (1024 * 1024)  # Rough estimation
            print(f"Estimated model size: ~{model_size_mb:.1f} MB")
            
            results[model_name] = {
                'load_time': load_time,
                'encode_time': encode_time,
                'embedding_dim': embedding_dim,
                'accuracy': accuracy,
                'model_size_mb': model_size_mb,
                'similarities': similarities
            }
            
        except Exception as e:
            print(f"Error loading {model_name}: {e}")
            results[model_name] = None
    
    # Print comparison summary
    print(f"\n{'='*80}")
    print("MODEL COMPARISON SUMMARY")
    print(f"{'='*80}")
    
    print(f"{'Model':<35} {'Load(s)':<8} {'Encode(s)':<10} {'Dim':<6} {'Accuracy':<10} {'Size(MB)':<8}")
    print("-" * 80)
    
    for model_name, result in results.items():
        if result:
            print(f"{model_name:<35} {result['load_time']:<8.2f} {result['encode_time']:<10.2f} "
                  f"{result['embedding_dim']:<6} {result['accuracy']:<10.1%} {result['model_size_mb']:<8.1f}")
    
    # Recommendations
    print(f"\n{'='*80}")
    print("RECOMMENDATIONS")
    print(f"{'='*80}")
    
    if all(results.values()):
        # Find best model for each criterion
        fastest_load = min(results.items(), key=lambda x: x[1]['load_time'])
        fastest_encode = min(results.items(), key=lambda x: x[1]['encode_time'])
        highest_accuracy = max(results.items(), key=lambda x: x[1]['accuracy'])
        smallest_size = min(results.items(), key=lambda x: x[1]['model_size_mb'])
        
        print(f"Fastest loading: {fastest_load[0]} ({fastest_load[1]['load_time']:.2f}s)")
        print(f"Fastest encoding: {fastest_encode[0]} ({fastest_encode[1]['encode_time']:.2f}s)")
        print(f"Highest accuracy: {highest_accuracy[0]} ({highest_accuracy[1]['accuracy']:.1%})")
        print(f"Smallest size: {smallest_size[0]} ({smallest_size[1]['model_size_mb']:.1f}MB)")
        
        print(f"\nFor your punctuation restoration task:")
        print(f"- If speed is priority: {fastest_encode[0]}")
        print(f"- If accuracy is priority: {highest_accuracy[0]}")
        print(f"- If multilingual support needed: paraphrase-multilingual-MiniLM-L12-v2")
        print(f"- If balanced approach: all-mpnet-base-v2")

def detailed_model_info():
    """Provide detailed information about each model."""
    
    print(f"\n{'='*80}")
    print("DETAILED MODEL INFORMATION")
    print(f"{'='*80}")
    
    model_details = {
        'all-MiniLM-L6-v2': {
            'description': 'Fast, lightweight model optimized for speed',
            'architecture': 'MiniLM (distilled BERT)',
            'layers': '6 transformer layers',
            'embedding_dim': 384,
            'pros': ['Very fast', 'Small memory footprint', 'Good for real-time applications'],
            'cons': ['Lower quality than larger models', 'May miss subtle semantic differences'],
            'best_for': 'Speed-critical applications, real-time processing'
        },
        'paraphrase-multilingual-MiniLM-L12-v2': {
            'description': 'Multilingual model with 12 layers',
            'architecture': 'MiniLM (distilled BERT)',
            'layers': '12 transformer layers',
            'embedding_dim': 384,
            'pros': ['Multilingual support', 'Better than L6 version', 'Good for mixed-language content'],
            'cons': ['Slower than L6', 'Still not highest quality'],
            'best_for': 'Multilingual applications, mixed-language transcriptions'
        },
        'all-mpnet-base-v2': {
            'description': 'High-quality model with excellent semantic understanding',
            'architecture': 'MPNet (Masked and Permuted Pre-training)',
            'layers': '12 transformer layers',
            'embedding_dim': 768,
            'pros': ['Highest quality embeddings', 'Excellent semantic understanding', 'Good for complex tasks'],
            'cons': ['Slowest of the three', 'Largest memory footprint', 'Higher computational cost'],
            'best_for': 'Quality-critical applications, complex semantic analysis'
        }
    }
    
    for model_name, details in model_details.items():
        print(f"\n{model_name}")
        print(f"Description: {details['description']}")
        print(f"Architecture: {details['architecture']}")
        print(f"Layers: {details['layers']}")
        print(f"Embedding dimension: {details['embedding_dim']}")
        print(f"Pros: {', '.join(details['pros'])}")
        print(f"Cons: {', '.join(details['cons'])}")
        print(f"Best for: {details['best_for']}")
        print("-" * 60)

if __name__ == "__main__":
    print("SentenceTransformer Model Comparison for Punctuation Restoration")
    print("=" * 80)
    
    detailed_model_info()
    compare_models() 