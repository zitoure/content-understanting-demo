"""
Example: How to Use the Assessment Evaluation Framework

This script demonstrates how to use the evaluation framework to compare
predicted assessment results against ground truth data.
"""

import json
from pathlib import Path
from evaluate.evaluation import AssessmentEvaluator, load_ground_truth, print_evaluation_summary

def example_healthcare_evaluation():
    """Example of evaluating healthcare assessment results"""
    
    # Load ground truth
    ground_truth = load_ground_truth("groundtruth/mental.json")
    
    # Example predicted results (this would come from your AI system)
    predicted_results = {
        "assessment_results": {
            "mood_description": "Anxious and persistent sadness",  # Exact match
            "anxiety_level": "High",  # Different from "Moderate" in ground truth
            "sleep_quality": "Fair",  # Exact match
            "depression_indicators": ["Persistent sadness", "Fatigue"],  # Missing "Loss of interest"
            "social_support": "Limited support, feels alone",  # Similar but different wording
            "medication_compliance": "Good",  # Exact match
            "risk_assessment": "Low",  # Exact match
            "intervention_needed": "yes",  # Exact match
            "additional_notes": "Patient needs more support"  # Different wording
        }
    }
    
    # Create evaluator
    evaluator = AssessmentEvaluator()
    
    # Run evaluation
    evaluation = evaluator.evaluate_assessment(
        predicted=predicted_results,
        ground_truth=ground_truth,
        assessment_type="healthcare"
    )
    
    # Print results
    print("=== Healthcare Assessment Evaluation ===")
    print_evaluation_summary(evaluation)
    
    return evaluation

def example_golf_evaluation():
    """Example of evaluating golf assessment results"""
    
    # Load ground truth
    ground_truth = load_ground_truth("groundtruth/golf.json")
    
    # Example predicted results
    predicted_results = {
        "assessment_results": {
            "years_playing": 7,  # Exact match
            "current_handicap": "9",  # Exact match
            "strongest_club": "driver",  # Exact match
            "weakest_area": "short game",  # Different from "chipping"
            "driving_accuracy": "Good",  # Exact match
            "mental_game": "Struggles with pressure",  # Different wording
            "goals": "Lower handicap to 5, compete locally",  # Similar content
            "practice_time": "6 hours weekly"  # Different format
        }
    }
    
    # Create evaluator with custom weights
    custom_weights = {
        "exact_match": 0.4,
        "token_f1": 0.3,
        "field_coverage": 0.2,
        "bleu": 0.1
    }
    evaluator = AssessmentEvaluator(weights=custom_weights)
    
    # Run evaluation
    evaluation = evaluator.evaluate_assessment(
        predicted=predicted_results,
        ground_truth=ground_truth,
        assessment_type="golf"
    )
    
    # Print results
    print("\n=== Golf Assessment Evaluation ===")
    print_evaluation_summary(evaluation)
    
    return evaluation

def compare_multiple_predictions():
    """Example of comparing multiple prediction approaches"""
    
    ground_truth = load_ground_truth("groundtruth/mental.json")
    
    # Different prediction approaches
    predictions = {
        "High Precision": {
            "assessment_results": {
                "mood_description": "Anxious and persistent sadness",
                "anxiety_level": "Moderate",
                "sleep_quality": "Fair",
                "medication_compliance": "Good",
                "risk_assessment": "Low"
            }
        },
        "High Recall": {
            "assessment_results": {
                "mood_description": "Anxious and sad",
                "anxiety_level": "Moderate to high",
                "sleep_quality": "Poor to fair",
                "depression_indicators": ["Sadness", "Loss of interest", "Fatigue", "Sleep issues"],
                "social_support": "Some support available",
                "medication_compliance": "Good",
                "risk_assessment": "Low",
                "intervention_needed": "yes",
                "additional_notes": "Patient expressing need for help",
                "extra_field": "Additional extracted information"
            }
        }
    }
    
    evaluator = AssessmentEvaluator()
    
    print("\n=== Comparing Multiple Prediction Approaches ===")
    
    results = {}
    for approach_name, predicted in predictions.items():
        evaluation = evaluator.evaluate_assessment(
            predicted=predicted,
            ground_truth=ground_truth,
            assessment_type="healthcare"
        )
        results[approach_name] = evaluation.overall_score
        
        print(f"\n--- {approach_name} Approach ---")
        print(f"Overall Score: {evaluation.overall_score:.3f}")
        
        for metric_result in evaluation.metric_results:
            print(f"{metric_result.metric_name}: {metric_result.score:.3f}")
    
    # Find best approach
    best_approach = max(results, key=results.get)
    print(f"\n🏆 Best Approach: {best_approach} (Score: {results[best_approach]:.3f})")

if __name__ == "__main__":
    print("🎯 Assessment Evaluation Framework Examples\n")
    
    # Run examples
    try:
        example_healthcare_evaluation()
        example_golf_evaluation()
        compare_multiple_predictions()
        
        print("\n✅ All examples completed successfully!")
        print("\nNext steps:")
        print("1. Run your actual demos to generate prediction files")
        print("2. Use evaluate_assessments.py to run full evaluations")
        print("3. Customize evaluation metrics for your specific needs")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Make sure you're running this from the demo directory with ground truth files available.")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
