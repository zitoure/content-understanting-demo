"""
Assessment Evaluation Framework for Azure AI Content Understanding

This module provides comprehensive evaluation capabilities for comparing
extracted assessment results against ground truth data. It supports multiple
evaluation metrics and works with different assessment types (healthcare, golf, etc.).

Supported Evaluation Metrics:
1. Exact Match Accuracy
2. Token-level F1 Score  
3. BLEU Score (for text similarity)
4. Field Coverage (percentage of fields correctly extracted)
5. Semantic Similarity (using embeddings)
6. Custom Domain-Specific Metrics

Usage:
    evaluator = AssessmentEvaluator()
    scores = evaluator.evaluate_assessment(predicted, ground_truth, assessment_type)
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import math

# Optional imports for semantic similarity
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    SEMANTIC_SIMILARITY_AVAILABLE = True
except ImportError:
    SEMANTIC_SIMILARITY_AVAILABLE = False

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Supported evaluation metric types"""
    EXACT_MATCH = "exact_match"
    TOKEN_F1 = "token_f1"
    BLEU = "bleu"
    FIELD_COVERAGE = "field_coverage"
    SEMANTIC_SIMILARITY = "semantic_similarity"
    CUSTOM_HEALTHCARE = "custom_healthcare"
    CUSTOM_GOLF = "custom_golf"


@dataclass
class EvaluationResult:
    """Container for evaluation results"""
    metric_name: str
    score: float
    max_score: float
    details: Dict[str, Any]
    field_scores: Optional[Dict[str, float]] = None


@dataclass
class AssessmentEvaluation:
    """Complete evaluation results for an assessment"""
    assessment_type: str
    ground_truth_file: str
    predicted_result: Dict[str, Any]
    overall_score: float
    metric_results: List[EvaluationResult]
    field_analysis: Dict[str, Any]
    timestamp: str


class AssessmentEvaluator:
    """
    Comprehensive evaluation framework for assessment extraction results.
    
    This evaluator supports multiple metrics and can be extended for domain-specific
    evaluation needs. It handles different data types and provides detailed analysis.
    """
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        Initialize the evaluator with optional metric weights.
        
        Args:
            weights: Dictionary mapping metric names to weights for overall scoring
        """
        self.weights = weights or {
            MetricType.EXACT_MATCH.value: 0.2,
            MetricType.TOKEN_F1.value: 0.2,
            MetricType.FIELD_COVERAGE.value: 0.2,
            MetricType.BLEU.value: 0.2,
            MetricType.SEMANTIC_SIMILARITY.value: 0.2
        }
        
        # Initialize semantic similarity model if available
        self.semantic_model = None
        if SEMANTIC_SIMILARITY_AVAILABLE:
            try:
                # Use TF-IDF vectorizer for semantic similarity
                self.semantic_model = TfidfVectorizer(
                    stop_words='english',
                    ngram_range=(1, 2),  # Use unigrams and bigrams
                    max_features=5000,
                    lowercase=True,
                    token_pattern=r'\b[a-zA-Z][a-zA-Z0-9]*\b'  # Only words starting with letters
                )
                logger.info("Initialized TF-IDF semantic similarity model")
            except Exception as e:
                logger.warning(f"Failed to load semantic similarity model: {e}")
                self.semantic_model = None
        
    def evaluate_assessment(
        self,
        predicted: Dict[str, Any],
        ground_truth: Dict[str, Any],
        assessment_type: str,
        metrics: Optional[List[MetricType]] = None,
        ignore_missing_fields: bool = False
    ) -> AssessmentEvaluation:
        """
        Perform comprehensive evaluation of an assessment extraction.
        
        Args:
            predicted: Predicted assessment results from AI system
            ground_truth: Ground truth assessment data
            assessment_type: Type of assessment (healthcare, golf, etc.)
            metrics: List of metrics to evaluate (defaults to all)
            ignore_missing_fields: If True, missing fields will be removed from ground truth
            
        Returns:
            Complete evaluation results
        """
        if metrics is None:
            # For healthcare, prioritize semantic similarity
            if assessment_type == "healthcare":
                metrics = [MetricType.EXACT_MATCH, MetricType.SEMANTIC_SIMILARITY, 
                          MetricType.FIELD_COVERAGE, MetricType.TOKEN_F1]
            else:
                metrics = [MetricType.EXACT_MATCH, MetricType.TOKEN_F1, 
                          MetricType.FIELD_COVERAGE, MetricType.BLEU]
        
        # Extract assessment data from both predicted and ground truth
        pred_assessment = self._extract_assessment_data(predicted)
        gt_assessment = ground_truth.get("assessment", {})
        
        # Filter ground truth if ignoring missing fields
        if ignore_missing_fields:
            # Remove fields from ground truth that are not in predicted results
            missing_fields = set(gt_assessment.keys()) - set(pred_assessment.keys())
            gt_assessment_filtered = {k: v for k, v in gt_assessment.items() 
                                    if k not in missing_fields}
            gt_assessment = gt_assessment_filtered
        
        # Run each evaluation metric
        metric_results = []
        for metric in metrics:
            result = self._evaluate_metric(pred_assessment, gt_assessment, metric, assessment_type)
            metric_results.append(result)
        
        # Calculate overall weighted score
        overall_score = self._calculate_overall_score(metric_results)
        
        # Perform field-level analysis
        field_analysis = self._analyze_fields(pred_assessment, gt_assessment)
        if ignore_missing_fields:
            field_analysis["ignore_missing_fields"] = True
            field_analysis["filtered_fields"] = list(missing_fields) if ignore_missing_fields else []
        
        return AssessmentEvaluation(
            assessment_type=assessment_type,
            ground_truth_file="",  # Will be set by caller
            predicted_result=predicted,
            overall_score=overall_score,
            metric_results=metric_results,
            field_analysis=field_analysis,
            timestamp=datetime.now().isoformat()
        )
    
    def _extract_assessment_data(self, predicted: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract assessment data from predicted results.
        Handles different output formats from the assessment processing.
        """
        # Handle different possible structures
        if "assessment_results" in predicted:
            return predicted["assessment_results"]
        elif "extracted_fields" in predicted:
            return predicted["extracted_fields"]
        elif "assessment" in predicted:
            return predicted["assessment"]
        else:
            # Try to find assessment-like data in the structure
            for key, value in predicted.items():
                if isinstance(value, dict) and len(value) > 3:  # Likely assessment data
                    return value
            return predicted
    
    def _evaluate_metric(
        self,
        predicted: Dict[str, Any],
        ground_truth: Dict[str, Any],
        metric: MetricType,
        assessment_type: str
    ) -> EvaluationResult:
        """Evaluate a specific metric"""
        
        if metric == MetricType.EXACT_MATCH:
            return self._evaluate_exact_match(predicted, ground_truth)
        elif metric == MetricType.TOKEN_F1:
            return self._evaluate_token_f1(predicted, ground_truth)
        elif metric == MetricType.BLEU:
            return self._evaluate_bleu(predicted, ground_truth)
        elif metric == MetricType.FIELD_COVERAGE:
            return self._evaluate_field_coverage(predicted, ground_truth)
        elif metric == MetricType.SEMANTIC_SIMILARITY:
            return self._evaluate_semantic_similarity(predicted, ground_truth)
        elif metric == MetricType.CUSTOM_HEALTHCARE and assessment_type == "healthcare":
            return self._evaluate_healthcare_specific(predicted, ground_truth)
        elif metric == MetricType.CUSTOM_GOLF and assessment_type == "golf":
            return self._evaluate_golf_specific(predicted, ground_truth)
        else:
            return EvaluationResult(
                metric_name=metric.value,
                score=0.0,
                max_score=1.0,
                details={"error": f"Metric {metric.value} not implemented"}
            )
    
    def _evaluate_exact_match(self, predicted: Dict[str, Any], ground_truth: Dict[str, Any]) -> EvaluationResult:
        """Evaluate exact match accuracy for each field"""
        field_scores = {}
        total_fields = 0
        correct_fields = 0
        
        all_fields = set(predicted.keys()) | set(ground_truth.keys())
        
        for field in all_fields:
            total_fields += 1
            pred_val = predicted.get(field)
            gt_val = ground_truth.get(field)
            
            # Normalize values for comparison
            pred_normalized = self._normalize_value(pred_val)
            gt_normalized = self._normalize_value(gt_val)
            
            is_match = pred_normalized == gt_normalized
            field_scores[field] = 1.0 if is_match else 0.0
            
            if is_match:
                correct_fields += 1
        
        accuracy = correct_fields / total_fields if total_fields > 0 else 0.0
        
        return EvaluationResult(
            metric_name="exact_match",
            score=accuracy,
            max_score=1.0,
            details={
                "correct_fields": correct_fields,
                "total_fields": total_fields,
                "accuracy": accuracy
            },
            field_scores=field_scores
        )
    
    def _evaluate_token_f1(self, predicted: Dict[str, Any], ground_truth: Dict[str, Any]) -> EvaluationResult:
        """Evaluate token-level F1 score for text fields"""
        field_scores = {}
        all_f1_scores = []
        
        all_fields = set(predicted.keys()) | set(ground_truth.keys())
        
        for field in all_fields:
            pred_val = str(predicted.get(field, "")).lower()
            gt_val = str(ground_truth.get(field, "")).lower()
            
            # Tokenize
            pred_tokens = set(self._tokenize(pred_val))
            gt_tokens = set(self._tokenize(gt_val))
            
            # Calculate F1
            if len(gt_tokens) == 0 and len(pred_tokens) == 0:
                f1 = 1.0
            elif len(gt_tokens) == 0 or len(pred_tokens) == 0:
                f1 = 0.0
            else:
                intersection = pred_tokens & gt_tokens
                precision = len(intersection) / len(pred_tokens) if pred_tokens else 0
                recall = len(intersection) / len(gt_tokens) if gt_tokens else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            field_scores[field] = f1
            all_f1_scores.append(f1)
        
        macro_f1 = sum(all_f1_scores) / len(all_f1_scores) if all_f1_scores else 0.0
        
        return EvaluationResult(
            metric_name="token_f1",
            score=macro_f1,
            max_score=1.0,
            details={
                "macro_f1": macro_f1,
                "field_count": len(all_fields)
            },
            field_scores=field_scores
        )
    
    def _evaluate_bleu(self, predicted: Dict[str, Any], ground_truth: Dict[str, Any]) -> EvaluationResult:
        """Evaluate BLEU score for text similarity"""
        field_scores = {}
        all_bleu_scores = []
        
        text_fields = []
        all_fields = set(predicted.keys()) | set(ground_truth.keys())
        
        for field in all_fields:
            pred_val = str(predicted.get(field, ""))
            gt_val = str(ground_truth.get(field, ""))
            
            # Only evaluate BLEU for text fields (not numeric or boolean)
            if len(pred_val.split()) > 1 or len(gt_val.split()) > 1:
                text_fields.append(field)
                bleu_score = self._calculate_bleu(pred_val, gt_val)
                field_scores[field] = bleu_score
                all_bleu_scores.append(bleu_score)
            else:
                field_scores[field] = None  # Not applicable
        
        avg_bleu = sum(all_bleu_scores) / len(all_bleu_scores) if all_bleu_scores else 0.0
        
        return EvaluationResult(
            metric_name="bleu",
            score=avg_bleu,
            max_score=1.0,
            details={
                "average_bleu": avg_bleu,
                "text_fields_evaluated": len(text_fields),
                "text_fields": text_fields
            },
            field_scores=field_scores
        )
    
    def _evaluate_field_coverage(self, predicted: Dict[str, Any], ground_truth: Dict[str, Any]) -> EvaluationResult:
        """Evaluate field coverage - how many ground truth fields were extracted"""
        gt_fields = set(ground_truth.keys())
        pred_fields = set(predicted.keys())
        
        # Fields present in ground truth that were extracted
        extracted_gt_fields = gt_fields & pred_fields
        
        # Fields extracted that weren't in ground truth (over-extraction)
        extra_fields = pred_fields - gt_fields
        
        coverage = len(extracted_gt_fields) / len(gt_fields) if gt_fields else 1.0
        
        field_scores = {}
        for field in gt_fields:
            field_scores[field] = 1.0 if field in pred_fields else 0.0
        
        return EvaluationResult(
            metric_name="field_coverage",
            score=coverage,
            max_score=1.0,
            details={
                "coverage": coverage,
                "extracted_gt_fields": len(extracted_gt_fields),
                "total_gt_fields": len(gt_fields),
                "extra_fields": list(extra_fields),
                "missing_fields": list(gt_fields - pred_fields)
            },
            field_scores=field_scores
        )
    
    def _evaluate_semantic_similarity(self, predicted: Dict[str, Any], ground_truth: Dict[str, Any]) -> EvaluationResult:
        """Evaluate semantic similarity using TF-IDF and cosine similarity"""
        field_scores = {}
        all_similarity_scores = []
        
        if not self.semantic_model:
            # Fallback to simple text similarity if model not available
            return self._evaluate_fallback_similarity(predicted, ground_truth)
        
        all_fields = set(predicted.keys()) | set(ground_truth.keys())
        
        for field in all_fields:
            pred_val = str(predicted.get(field, "")).strip()
            gt_val = str(ground_truth.get(field, "")).strip()
            
            # Handle lists by converting to sentences
            if isinstance(predicted.get(field), list):
                pred_val = "; ".join([str(item) for item in predicted.get(field, [])])
            if isinstance(ground_truth.get(field), list):
                gt_val = "; ".join([str(item) for item in ground_truth.get(field, [])])
            
            # Skip empty or very short values
            if len(pred_val) < 2 and len(gt_val) < 2:
                similarity = 1.0 if pred_val == gt_val else 0.0
            elif len(pred_val) < 2 or len(gt_val) < 2:
                similarity = 0.0
            else:
                try:
                    # Create TF-IDF vectors for both texts
                    texts = [pred_val, gt_val]
                    
                    # Create a fresh vectorizer instance for this comparison
                    vectorizer = TfidfVectorizer(
                        stop_words='english',
                        ngram_range=(1, 2),
                        max_features=1000,  # Smaller for individual field comparison
                        lowercase=True,
                        token_pattern=r'\b[a-zA-Z][a-zA-Z0-9]*\b'
                    )
                    
                    # Fit and transform the texts
                    tfidf_matrix = vectorizer.fit_transform(texts)
                    
                    # Calculate cosine similarity
                    similarity_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
                    similarity = float(similarity_matrix[0][0])
                    
                    # Ensure similarity is between 0 and 1
                    similarity = max(0.0, min(1.0, similarity))
                    
                except Exception as e:
                    logger.warning(f"Error calculating semantic similarity for field {field}: {e}")
                    similarity = 0.0
            
            field_scores[field] = similarity
            all_similarity_scores.append(similarity)
        
        avg_similarity = sum(all_similarity_scores) / len(all_similarity_scores) if all_similarity_scores else 0.0
        
        return EvaluationResult(
            metric_name="semantic_similarity",
            score=avg_similarity,
            max_score=1.0,
            details={
                "average_similarity": avg_similarity,
                "field_count": len(all_fields),
                "model_used": "TF-IDF with cosine similarity" if self.semantic_model else "fallback"
            },
            field_scores=field_scores
        )
    
    def _evaluate_fallback_similarity(self, predicted: Dict[str, Any], ground_truth: Dict[str, Any]) -> EvaluationResult:
        """Fallback similarity evaluation when semantic model is not available"""
        field_scores = {}
        all_similarity_scores = []
        
        all_fields = set(predicted.keys()) | set(ground_truth.keys())
        
        for field in all_fields:
            pred_val = self._normalize_value(predicted.get(field))
            gt_val = self._normalize_value(ground_truth.get(field))
            
            # Use Jaccard similarity on word sets
            pred_words = set(self._tokenize(pred_val))
            gt_words = set(self._tokenize(gt_val))
            
            if len(pred_words) == 0 and len(gt_words) == 0:
                similarity = 1.0
            elif len(pred_words) == 0 or len(gt_words) == 0:
                similarity = 0.0
            else:
                intersection = pred_words & gt_words
                union = pred_words | gt_words
                similarity = len(intersection) / len(union) if union else 0.0
            
            field_scores[field] = similarity
            all_similarity_scores.append(similarity)
        
        avg_similarity = sum(all_similarity_scores) / len(all_similarity_scores) if all_similarity_scores else 0.0
        
        return EvaluationResult(
            metric_name="semantic_similarity",
            score=avg_similarity,
            max_score=1.0,
            details={
                "average_similarity": avg_similarity,
                "field_count": len(all_fields),
                "model_used": "jaccard_fallback"
            },
            field_scores=field_scores
        )
    
    def _evaluate_healthcare_specific(self, predicted: Dict[str, Any], ground_truth: Dict[str, Any]) -> EvaluationResult:
        """Healthcare-specific evaluation metrics using semantic similarity for critical fields"""
        
        # Critical healthcare fields that must be accurate
        critical_fields = [
            "pain_level", "medication_side_effects", "functional_status",
            "risk_assessment", "intervention_needed", "mood_description",
            "depression_indicators", "anxiety_level"
        ]
        
        critical_accuracy = 0.0
        critical_total = 0
        critical_scores = {}
        
        for field in critical_fields:
            if field in ground_truth:
                critical_total += 1
                pred_val = predicted.get(field)
                gt_val = ground_truth.get(field)
                
                # Use semantic similarity for text fields, exact match for categorical
                if field in ["pain_level", "risk_assessment", "anxiety_level", "intervention_needed"]:
                    # Categorical fields - use exact match
                    pred_normalized = self._normalize_value(pred_val)
                    gt_normalized = self._normalize_value(gt_val)
                    is_correct = pred_normalized == gt_normalized
                    score = 1.0 if is_correct else 0.0
                else:
                    # Text fields - use semantic similarity
                    score = self._calculate_field_semantic_similarity(pred_val, gt_val)
                    # Consider scores above 0.7 as "correct" for healthcare
                    is_correct = score >= 0.7
                
                critical_scores[field] = score
                if is_correct:
                    critical_accuracy += 1
        
        critical_accuracy = critical_accuracy / critical_total if critical_total > 0 else 0.0
        
        return EvaluationResult(
            metric_name="healthcare_critical",
            score=critical_accuracy,
            max_score=1.0,
            details={
                "critical_accuracy": critical_accuracy,
                "critical_fields_correct": sum(1 for score in critical_scores.values() if score >= 0.7),
                "critical_fields_total": critical_total,
                "critical_fields": critical_fields,
                "similarity_threshold": 0.7
            },
            field_scores=critical_scores
        )
    
    def _calculate_field_semantic_similarity(self, pred_val: Any, gt_val: Any) -> float:
        """Calculate semantic similarity for a single field"""
        if not self.semantic_model:
            # Fallback to normalized string comparison
            pred_norm = self._normalize_value(pred_val)
            gt_norm = self._normalize_value(gt_val)
            if pred_norm == gt_norm:
                return 1.0
            # Use Jaccard similarity as fallback
            pred_words = set(self._tokenize(pred_norm))
            gt_words = set(self._tokenize(gt_norm))
            if len(pred_words) == 0 and len(gt_words) == 0:
                return 1.0
            elif len(pred_words) == 0 or len(gt_words) == 0:
                return 0.0
            intersection = pred_words & gt_words
            union = pred_words | gt_words
            return len(intersection) / len(union) if union else 0.0
        
        # Convert to strings
        pred_str = str(pred_val) if pred_val is not None else ""
        gt_str = str(gt_val) if gt_val is not None else ""
        
        # Handle lists
        if isinstance(pred_val, list):
            pred_str = "; ".join([str(item) for item in pred_val])
        if isinstance(gt_val, list):
            gt_str = "; ".join([str(item) for item in gt_val])
        
        # Skip very short or empty strings
        if len(pred_str.strip()) < 2 and len(gt_str.strip()) < 2:
            return 1.0 if pred_str.strip() == gt_str.strip() else 0.0
        elif len(pred_str.strip()) < 2 or len(gt_str.strip()) < 2:
            return 0.0
        
        try:
            # Create TF-IDF vectors for both texts
            texts = [pred_str, gt_str]
            
            # Create a fresh vectorizer instance for this comparison
            vectorizer = TfidfVectorizer(
                stop_words='english',
                ngram_range=(1, 2),
                max_features=1000,  # Smaller for individual field comparison
                lowercase=True,
                token_pattern=r'\b[a-zA-Z][a-zA-Z0-9]*\b'
            )
            
            # Fit and transform the texts
            tfidf_matrix = vectorizer.fit_transform(texts)
            
            # Calculate cosine similarity
            similarity_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
            similarity = float(similarity_matrix[0][0])
            
            # Ensure similarity is between 0 and 1
            return max(0.0, min(1.0, similarity))
            
        except Exception as e:
            logger.warning(f"Error calculating semantic similarity: {e}")
            return 0.0
    
    def _evaluate_golf_specific(self, predicted: Dict[str, Any], ground_truth: Dict[str, Any]) -> EvaluationResult:
        """Golf-specific evaluation metrics"""
        
        # Important golf assessment fields
        key_fields = [
            "current_handicap", "years_playing", "strongest_club", 
            "weakest_area", "goals", "practice_time"
        ]
        
        key_accuracy = 0.0
        key_total = 0
        key_scores = {}
        
        for field in key_fields:
            if field in ground_truth:
                key_total += 1
                pred_val = self._normalize_value(predicted.get(field))
                gt_val = self._normalize_value(ground_truth.get(field))
                
                is_correct = pred_val == gt_val
                key_scores[field] = 1.0 if is_correct else 0.0
                if is_correct:
                    key_accuracy += 1
        
        key_accuracy = key_accuracy / key_total if key_total > 0 else 0.0
        
        return EvaluationResult(
            metric_name="golf_key_fields",
            score=key_accuracy,
            max_score=1.0,
            details={
                "key_accuracy": key_accuracy,
                "key_fields_correct": sum(key_scores.values()),
                "key_fields_total": key_total,
                "key_fields": key_fields
            },
            field_scores=key_scores
        )
    
    def _normalize_value(self, value: Any) -> str:
        """Normalize values for comparison"""
        if value is None:
            return ""
        elif isinstance(value, bool):
            return str(value).lower()
        elif isinstance(value, (int, float)):
            return str(value)
        elif isinstance(value, list):
            return " ".join(sorted([str(v).lower().strip() for v in value]))
        else:
            return str(value).lower().strip()
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for text"""
        return re.findall(r'\w+', text.lower())
    
    def _calculate_bleu(self, predicted: str, reference: str) -> float:
        """Calculate simplified BLEU score"""
        pred_tokens = self._tokenize(predicted)
        ref_tokens = self._tokenize(reference)
        
        if not ref_tokens:
            return 1.0 if not pred_tokens else 0.0
        if not pred_tokens:
            return 0.0
        
        # Simple unigram BLEU
        pred_counts = {}
        for token in pred_tokens:
            pred_counts[token] = pred_counts.get(token, 0) + 1
        
        ref_counts = {}
        for token in ref_tokens:
            ref_counts[token] = ref_counts.get(token, 0) + 1
        
        overlap = 0
        for token, count in pred_counts.items():
            overlap += min(count, ref_counts.get(token, 0))
        
        precision = overlap / len(pred_tokens) if pred_tokens else 0
        
        # Brevity penalty
        bp = min(1.0, len(pred_tokens) / len(ref_tokens)) if ref_tokens else 1.0
        
        return bp * precision
    
    def _calculate_overall_score(self, metric_results: List[EvaluationResult]) -> float:
        """Calculate weighted overall score"""
        total_score = 0.0
        total_weight = 0.0
        
        for result in metric_results:
            weight = self.weights.get(result.metric_name, 1.0)
            total_score += result.score * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def _analyze_fields(self, predicted: Dict[str, Any], ground_truth: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze field-level performance"""
        analysis = {
            "total_gt_fields": len(ground_truth),
            "total_pred_fields": len(predicted),
            "common_fields": len(set(predicted.keys()) & set(ground_truth.keys())),
            "missing_fields": list(set(ground_truth.keys()) - set(predicted.keys())),
            "extra_fields": list(set(predicted.keys()) - set(ground_truth.keys())),
            "field_details": {}
        }
        
        all_fields = set(predicted.keys()) | set(ground_truth.keys())
        for field in all_fields:
            analysis["field_details"][field] = {
                "in_ground_truth": field in ground_truth,
                "in_predicted": field in predicted,
                "gt_value": ground_truth.get(field),
                "pred_value": predicted.get(field),
                "exact_match": self._normalize_value(predicted.get(field)) == self._normalize_value(ground_truth.get(field))
            }
        
        return analysis


def load_ground_truth(ground_truth_path: Union[str, Path]) -> Dict[str, Any]:
    """Load ground truth data from JSON file"""
    with open(ground_truth_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def evaluate_assessment_file(
    predicted_file: Union[str, Path],
    ground_truth_file: Union[str, Path],
    assessment_type: str,
    evaluator: Optional[AssessmentEvaluator] = None,
    ignore_missing_fields: bool = False
) -> AssessmentEvaluation:
    """
    Evaluate an assessment file against ground truth.
    
    Args:
        predicted_file: Path to predicted assessment results
        ground_truth_file: Path to ground truth data
        assessment_type: Type of assessment (healthcare, golf, etc.)
        evaluator: Optional evaluator instance
        ignore_missing_fields: If True, missing fields won't be penalized
        
    Returns:
        Evaluation results
    """
    if evaluator is None:
        evaluator = AssessmentEvaluator()
    
    # Load data
    with open(predicted_file, 'r', encoding='utf-8') as f:
        predicted = json.load(f)
    
    ground_truth = load_ground_truth(ground_truth_file)
    
    # Evaluate
    evaluation = evaluator.evaluate_assessment(
        predicted, 
        ground_truth, 
        assessment_type,
        ignore_missing_fields=ignore_missing_fields
    )
    evaluation.ground_truth_file = str(ground_truth_file)
    
    return evaluation


def save_evaluation_report(evaluation: AssessmentEvaluation, output_path: Union[str, Path]):
    """Save detailed evaluation report to JSON file"""
    
    # Convert dataclass to dict for JSON serialization
    report = {
        "evaluation_summary": {
            "assessment_type": evaluation.assessment_type,
            "ground_truth_file": evaluation.ground_truth_file,
            "overall_score": evaluation.overall_score,
            "timestamp": evaluation.timestamp
        },
        "metric_results": [],
        "field_analysis": evaluation.field_analysis
    }
    
    for result in evaluation.metric_results:
        metric_data = {
            "metric_name": result.metric_name,
            "score": result.score,
            "max_score": result.max_score,
            "details": result.details
        }
        if result.field_scores:
            metric_data["field_scores"] = result.field_scores
        report["metric_results"].append(metric_data)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


def print_evaluation_summary(evaluation: AssessmentEvaluation):
    """Print a human-readable evaluation summary"""
    
    print(f"\n=== Assessment Evaluation Summary ===")
    print(f"Assessment Type: {evaluation.assessment_type}")
    print(f"Overall Score: {evaluation.overall_score:.3f}")
    print(f"Timestamp: {evaluation.timestamp}")
    
    # Check if ignore_missing_fields was used
    if evaluation.field_analysis.get("ignore_missing_fields"):
        print(f"Mode: Ignoring missing fields")
        if "filtered_fields" in evaluation.field_analysis:
            filtered = evaluation.field_analysis["filtered_fields"]
            if filtered:
                print(f"Filtered fields: {', '.join(filtered)}")
    
    print(f"\n--- Metric Scores ---")
    for result in evaluation.metric_results:
        print(f"{result.metric_name}: {result.score:.3f} / {result.max_score}")
        if "accuracy" in result.details:
            print(f"  Accuracy: {result.details['accuracy']:.3f}")
        if "coverage" in result.details:
            print(f"  Coverage: {result.details['coverage']:.3f}")
        if "average_similarity" in result.details:
            print(f"  Avg Similarity: {result.details['average_similarity']:.3f}")
            if "model_used" in result.details:
                print(f"  Model: {result.details['model_used']}")
        if "similarity_threshold" in result.details:
            print(f"  Similarity Threshold: {result.details['similarity_threshold']}")
            correct_count = result.details.get('critical_fields_correct', 0)
            total_count = result.details.get('critical_fields_total', 0)
            print(f"  Fields Above Threshold: {correct_count}/{total_count}")
    
    print(f"\n--- Field Analysis ---")
    analysis = evaluation.field_analysis
    print(f"Ground Truth Fields: {analysis['total_gt_fields']}")
    print(f"Predicted Fields: {analysis['total_pred_fields']}")
    print(f"Common Fields: {analysis['common_fields']}")
    
    if "missing_fields" in analysis and analysis['missing_fields']:
        print(f"Missing Fields: {', '.join(analysis['missing_fields'])}")
    
    if analysis['extra_fields']:
        print(f"Extra Fields: {', '.join(analysis['extra_fields'])}")
    
    print(f"\n--- Field-by-Field Results ---")
    for field, details in analysis['field_details'].items():
        status = "✓" if details['exact_match'] else "✗"
        print(f"{status} {field}")
        if not details['exact_match'] and details['in_ground_truth'] and details['in_predicted']:
            print(f"    GT: {details['gt_value']}")
            print(f"  Pred: {details['pred_value']}")


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate assessment extraction results")
    parser.add_argument("predicted_file", help="Path to predicted assessment JSON file")
    parser.add_argument("ground_truth_file", help="Path to ground truth JSON file")
    parser.add_argument("--assessment-type", default="healthcare", help="Assessment type")
    parser.add_argument("--output", help="Output path for detailed report")
    parser.add_argument("--ignore-missing-fields", action="store_true", 
                       help="Ignore missing fields in evaluation")
    
    args = parser.parse_args()
    
    evaluation = evaluate_assessment_file(
        args.predicted_file,
        args.ground_truth_file,
        args.assessment_type,
        ignore_missing_fields=args.ignore_missing_fields
    )
    
    print_evaluation_summary(evaluation)
    
    if args.output:
        save_evaluation_report(evaluation, args.output)
        print(f"\nDetailed report saved to: {args.output}")
