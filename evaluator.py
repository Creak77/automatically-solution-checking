import re
from sentence_transformers import SentenceTransformer, util
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk


class AnswerEvaluator:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def preprocess_answer(self, answer):
        answer = answer.lower()
        answer = re.sub(r'[^\w\s]', '', answer)
        stop_words = set(stopwords.words('english'))
        word_tokens = word_tokenize(answer)
        filtered_answer = ' '.join([word for word in word_tokens if word not in stop_words])
        return filtered_answer
    
    def exact_match(self, given_answer, ground_truth):
        return self.preprocess_answer(given_answer) == self.preprocess_answer(ground_truth)
    
    def get_sentence_embeddings(self, text):
        return self.model.encode(text, convert_to_tensor=True)
    
    def semantic_similarity(self, given_answer, ground_truth):
        given_answer_emb = self.get_sentence_embeddings(self.preprocess_answer(given_answer))
        ground_truth_emb = self.get_sentence_embeddings(self.preprocess_answer(ground_truth))
        similarity = util.pytorch_cos_sim(given_answer_emb, ground_truth_emb).item()
        return similarity
    
    def evaluate_answer(self, given_answer, ground_truth, threshold=0.7):
        # Exact match
        if self.exact_match(given_answer, ground_truth):
            return 1.0, "Exact Match"
        
        # Semantic similarity
        similarity = self.semantic_similarity(given_answer, ground_truth)
        if similarity >= threshold:
            return similarity, "Semantic Match"
        
        return similarity, "No Match"

if __name__ == "__main__":
    # Download stopwords if not already downloaded
    #nltk.download('punkt')
    #nltk.download('stopwords')
    evaluator = AnswerEvaluator()

    # Example usage
    question = "What is the capital of France?"
    ground_truth = "Paris"
    given_answer = "The capital of France is Paris."

    accuracy, match_type = evaluator.evaluate_answer(given_answer, ground_truth)
    print(f"Accuracy: {accuracy}, Match Type: {match_type}")

    question = "Explain the theory of relativity."
    ground_truth = "The theory of relativity, formulated by Albert Einstein, revolutionized the way we understand space, time, and gravity. It consists of two theories: special relativity and general relativity."
    given_answer = "Einstein's theory of relativity changed our understanding of space, time, and gravity. It includes special and general relativity."

    accuracy, match_type = evaluator.evaluate_answer(given_answer, ground_truth)
    print(f"Accuracy: {accuracy}, Match Type: {match_type}")

    ground_truth = 'a'
    given_answer = 'a'

    accuracy, match_type = evaluator.evaluate_answer(given_answer, ground_truth)
    print(f"Accuracy: {accuracy}, Match Type: {match_type}")