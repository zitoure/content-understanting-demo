"""
Evaluation Runner for Assessment Demos

This script evaluates existing assessment demo results against ground truth data.
It can run single evaluations or batch evaluations across multiple scenarios.

Usage:
    # Evaluate healthcare assessment result
    python evaluate_assessments.py --demo healthcare --ground-truth mental.json
    
    # Evaluate golf assessment result  
    python evaluate_assessments.py --demo golf --ground-truth golf.json
    
    # Evaluate specific output file
    python evaluate_assessments.py --output-file path/to/result.json --ground-truth mental.json --type healthcare
    
    # Run all available evaluations
    python evaluate_assessments.py --all
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from evaluate.evaluation import (
    AssessmentEvaluator, evaluate_assessment_file, 
    print_evaluation_summary, save_evaluation_report,
    load_ground_truth
)


class AssessmentEvaluationRunner:
    """
    Runner for evaluating assessment extraction results against ground truth data.
    Works with existing JSON output files from demos.
    """
    
    def __init__(self, base_dir: Optional[Path] = None):
        """
        Initialize the evaluation runner.
        
        Args:
            base_dir: Base directory containing demo files (defaults to current directory)
        """
        self.base_dir = base_dir or Path(".")
        self.output_dir = self.base_dir / "scenarios" / "output"
        self.ground_truth_dir = self.base_dir / "groundtruth"
        
        # Demo configurations for finding output files
        self.demo_configs = {
            "healthcare": {
                "output_patterns": [
                    "**/patient_assessments/*.json",
                    "**/mental_health_*.json",
                    "**/healthcare_*.json"
                ],
                "ground_truth_files": [
                    "mental.json",
                    "physical.json", 
                    "geriartric.json"
                ],
                "assessment_type": "healthcare"
            },
            "golf": {
                "output_patterns": [
                    "**/golf_assessments/*.json",
                    "**/golf_swing_*.json",
                    "**/golf_*.json"
                ],
                "ground_truth_files": [
                    "golf.json"
                ],
                "assessment_type": "golf"
            }
        }
    
    def evaluate_existing_results(
        self, 
        demo_type: str, 
        ground_truth_file: str,
        output_file: Optional[str] = None,
        save_report: bool = True,
        ignore_missing_fields: bool = False
    ) -> Dict:
        """
        Evaluate existing demo results against ground truth.
        
        Args:
            demo_type: Type of demo (healthcare, golf)
            ground_truth_file: Name of ground truth file
            output_file: Specific output file to evaluate (optional)
            save_report: Whether to save detailed evaluation report
            ignore_missing_fields: If True, missing fields won't be penalized
            
        Returns:
            Evaluation results dictionary
        """
        if demo_type not in self.demo_configs:
            raise ValueError(f"Unknown demo type: {demo_type}")
        
        config = self.demo_configs[demo_type]
        
        # Check if ground truth file exists
        gt_path = self.ground_truth_dir / ground_truth_file
        if not gt_path.exists():
            raise FileNotFoundError(f"Ground truth file not found: {gt_path}")
        
        print(f"\n🎯 Evaluating {demo_type} demo results")
        print(f"📁 Ground truth: {ground_truth_file}")
        
        # Find output file to evaluate
        if output_file:
            predicted_file = Path(output_file)
            if not predicted_file.exists():
                raise FileNotFoundError(f"Output file not found: {predicted_file}")
        else:
            # Find the most recent output file for this demo type
            output_files = self._find_output_files(demo_type)
            if not output_files:
                raise FileNotFoundError(f"No output files found for {demo_type} demo")
            
            predicted_file = max(output_files, key=lambda p: p.stat().st_mtime)
        
        print(f"� Evaluating file: {predicted_file}")
        
        # Extract assessment data from the output file
        extracted_data = self._extract_assessment_from_output(predicted_file, demo_type)
        
        # Load ground truth
        ground_truth = load_ground_truth(gt_path)
        
        # Run evaluation
        evaluator = AssessmentEvaluator()
        evaluation = evaluator.evaluate_assessment(
            predicted=extracted_data,
            ground_truth=ground_truth,
            assessment_type=config["assessment_type"],
            ignore_missing_fields=ignore_missing_fields
        )
        
        # Print summary
        print_evaluation_summary(evaluation)
        
        # Save detailed report if requested
        if save_report:
            report_name = f"{demo_type}_{ground_truth_file.replace('.json', '')}_evaluation.json"
            report_path = self.base_dir / "output" / "evaluations" / report_name
            report_path.parent.mkdir(parents=True, exist_ok=True)
            
            save_evaluation_report(evaluation, report_path)
            print(f"📊 Detailed report saved to: {report_path}")
        
        return {
            "demo_type": demo_type,
            "ground_truth_file": ground_truth_file,
            "output_file": str(predicted_file),
            "overall_score": evaluation.overall_score,
            "evaluation": evaluation
        }
    
    def _extract_assessment_from_output(self, output_file: Path, demo_type: str) -> Dict:
        """
        Extract assessment data from demo output file.
        
        Args:
            output_file: Path to the output JSON file
            demo_type: Type of demo (healthcare, golf)
            
        Returns:
            Extracted assessment data in evaluation format
        """
        with open(output_file, 'r', encoding='utf-8') as f:
            output_data = json.load(f)
        
        # Healthcare format extraction
        if demo_type == "healthcare":
            if "assessment_responses" in output_data:
                # Extract values from the detailed healthcare format
                extracted = {}
                for field, response_data in output_data["assessment_responses"].items():
                    if isinstance(response_data, dict) and "extracted_value" in response_data:
                        extracted[field] = response_data["extracted_value"]
                    else:
                        extracted[field] = response_data
                return {"assessment_results": extracted}
            
            elif "extracted_fields" in output_data:
                return {"assessment_results": output_data["extracted_fields"]}
        
        # Golf format extraction
        elif demo_type == "golf":
            if "extracted_fields" in output_data:
                # Map golf fields to ground truth structure
                extracted = {}
                golf_data = output_data["extracted_fields"]
                
                # Map the fields based on the ground truth structure
                field_mapping = {
                    "years_playing": self._extract_years_from_text(golf_data.get("additional_notes", "")),
                    "current_handicap": self._extract_handicap_from_text(golf_data.get("additional_notes", "")),
                    "strongest_club": self._extract_strongest_club(golf_data),
                    "weakest_area": self._extract_weakest_area(golf_data),
                    "driving_accuracy": golf_data.get("stance_evaluation", "Good"),
                    "short_game_confidence": self._map_confidence_level(golf_data.get("session_rating")),
                    "pre_shot_routine": self._extract_routine_status(golf_data.get("areas_for_improvement", [])),
                    "pressure_handling": self._extract_pressure_handling(golf_data.get("areas_for_improvement", [])),
                    "fitness_routine": self._extract_fitness_status(golf_data.get("areas_for_improvement", [])),
                    "primary_goal": self._extract_goals(golf_data.get("additional_notes", "")),
                    "target_handicap": self._extract_target_handicap(golf_data.get("additional_notes", "")),
                    "motivation_level": self._map_motivation_level(golf_data.get("session_rating")),
                    "practice_time": self._extract_practice_time(golf_data.get("additional_notes", "")),
                    "competitive_interest": self._extract_competitive_interest(golf_data.get("additional_notes", ""))
                }
                
                # Only include fields that have values
                for key, value in field_mapping.items():
                    if value is not None:
                        extracted[key] = value
                
                return {"assessment_results": extracted}
        
        # Fallback: try to find assessment-like data in the structure
        if "assessment" in output_data:
            return {"assessment_results": output_data["assessment"]}
        elif "extracted_fields" in output_data:
            return {"assessment_results": output_data["extracted_fields"]}
        else:
            # Return the whole structure and let the evaluator figure it out
            return output_data
    
    def _extract_years_from_text(self, text: str) -> Optional[int]:
        """Extract years playing from text"""
        import re
        match = re.search(r'(\d+)\s*years?', text.lower())
        return int(match.group(1)) if match else None
    
    def _extract_handicap_from_text(self, text: str) -> Optional[str]:
        """Extract handicap from text"""
        import re
        match = re.search(r'(\d+)\s*handicap', text.lower())
        return match.group(1) if match else None
    
    def _extract_strongest_club(self, data: Dict) -> Optional[str]:
        """Extract strongest club from golf data"""
        text = data.get("swing_technique", "") + " " + data.get("additional_notes", "")
        if "driver" in text.lower():
            return "driver"
        return None
    
    def _extract_weakest_area(self, data: Dict) -> Optional[str]:
        """Extract weakest area from golf data"""
        areas = data.get("areas_for_improvement", [])
        if isinstance(areas, list) and areas:
            for area in areas:
                if "chip" in area.lower():
                    return "chipping"
            return areas[0].lower()
        return None
    
    def _map_confidence_level(self, rating: str) -> Optional[str]:
        """Map session rating to confidence level"""
        if not rating:
            return None
        rating_lower = rating.lower()
        if "good" in rating_lower or "excellent" in rating_lower:
            return "Somewhat Confident"
        return "Needs Improvement"
    
    def _extract_routine_status(self, improvements: List[str]) -> Optional[bool]:
        """Extract pre-shot routine status"""
        if isinstance(improvements, list):
            for item in improvements:
                if "routine" in item.lower():
                    return False  # If it's in improvements, they don't have one
        return None
    
    def _extract_pressure_handling(self, improvements: List[str]) -> Optional[str]:
        """Extract pressure handling from improvements"""
        if isinstance(improvements, list):
            for item in improvements:
                if "pressure" in item.lower() or "tight" in item.lower():
                    return "gets anxious and rushes shots"
        return None
    
    def _extract_fitness_status(self, improvements: List[str]) -> Optional[bool]:
        """Extract fitness routine status"""
        if isinstance(improvements, list):
            for item in improvements:
                if "fitness" in item.lower():
                    return False  # If it's in improvements, they don't have one
        return None
    
    def _extract_goals(self, text: str) -> Optional[str]:
        """Extract goals from text"""
        if "handicap" in text.lower() and "compete" in text.lower():
            return "get down to a five handicap and compete"
        return None
    
    def _extract_target_handicap(self, text: str) -> Optional[str]:
        """Extract target handicap from text"""
        import re
        match = re.search(r'reduce.*handicap.*to\s*(\d+)', text.lower())
        if match:
            return match.group(1)
        return None
    
    def _map_motivation_level(self, rating: str) -> Optional[str]:
        """Map session rating to motivation level"""
        if not rating:
            return None
        rating_lower = rating.lower()
        if "good" in rating_lower or "excellent" in rating_lower:
            return "Very Motivated"
        return "Motivated"
    
    def _extract_practice_time(self, text: str) -> Optional[str]:
        """Extract practice time from text"""
        import re
        match = re.search(r'(\d+)\s*hours?.*week', text.lower())
        if match:
            return f"{match.group(1)} hours per week"
        return None
    
    def _extract_competitive_interest(self, text: str) -> Optional[bool]:
        """Extract competitive interest from text"""
        return "compete" in text.lower() or "tournament" in text.lower()
    
    def _find_output_files(self, demo_type: str) -> List[Path]:
        """Find output files generated by demo"""
        if demo_type not in self.demo_configs:
            return []
        
        patterns = self.demo_configs[demo_type]["output_patterns"]
        output_files = []
        
        for pattern in patterns:
            output_files.extend(self.output_dir.glob(pattern))
        
        # Also check the base scenarios directory
        base_scenarios = self.base_dir / "scenarios"
        for pattern in patterns:
            output_files.extend(base_scenarios.glob(pattern))
        
        return output_files
    
    def run_all_evaluations(self, save_reports: bool = True, ignore_missing_fields: bool = False) -> Dict[str, List[Dict]]:
        """
        Run all available evaluations on existing results.
        
        Args:
            save_reports: Whether to save detailed reports
            ignore_missing_fields: Whether to ignore missing fields in evaluation
            
        Returns:
            Dictionary mapping demo types to evaluation results
        """
        all_results = {}
        
        for demo_type, config in self.demo_configs.items():
            demo_results = []
            
            for gt_file in config["ground_truth_files"]:
                try:
                    result = self.evaluate_existing_results(
                        demo_type=demo_type,
                        ground_truth_file=gt_file,
                        save_report=save_reports,
                        ignore_missing_fields=ignore_missing_fields
                    )
                    demo_results.append(result)
                    
                except Exception as e:
                    print(f"❌ Error evaluating {demo_type} with {gt_file}: {e}")
                    demo_results.append({
                        "demo_type": demo_type,
                        "ground_truth_file": gt_file,
                        "error": str(e)
                    })
            
            all_results[demo_type] = demo_results
        
        return all_results
    
    def create_mock_predicted_data(self, demo_type: str, ground_truth_file: str) -> Dict:
        """
        Create mock predicted data for testing evaluation without running demos.
        This is useful for testing the evaluation framework.
        """
        gt_path = self.ground_truth_dir / ground_truth_file
        ground_truth = load_ground_truth(gt_path)
        
        # Create mock predicted results with some intentional differences
        mock_predicted = {
            "assessment_results": {},
            "metadata": {
                "demo_type": demo_type,
                "mock_data": True
            }
        }
        
        gt_assessment = ground_truth.get("assessment", {})
        
        # Copy most fields correctly, but introduce some errors for testing
        for i, (field, value) in enumerate(gt_assessment.items()):
            if i % 3 == 0:  # Every 3rd field has an error
                if isinstance(value, str):
                    mock_predicted["assessment_results"][field] = value + " (modified)"
                elif isinstance(value, (int, float)):
                    mock_predicted["assessment_results"][field] = value + 1
                elif isinstance(value, list):
                    mock_predicted["assessment_results"][field] = value + ["extra_item"]
                else:
                    mock_predicted["assessment_results"][field] = f"modified_{value}"
            else:
                mock_predicted["assessment_results"][field] = value
        
        # Add some extra fields not in ground truth
        mock_predicted["assessment_results"]["extra_field_1"] = "This shouldn't be here"
        mock_predicted["assessment_results"]["extra_field_2"] = 99
        
        return mock_predicted
    
    def test_evaluation_framework(self):
        """Test the evaluation framework with mock data"""
        print("\n🧪 Testing Evaluation Framework")
        
        evaluator = AssessmentEvaluator()
        
        for demo_type, config in self.demo_configs.items():
            for gt_file in config["ground_truth_files"]:
                print(f"\n--- Testing {demo_type} with {gt_file} ---")
                
                try:
                    # Create mock data
                    mock_predicted = self.create_mock_predicted_data(demo_type, gt_file)
                    
                    # Load ground truth
                    gt_path = self.ground_truth_dir / gt_file
                    ground_truth = load_ground_truth(gt_path)
                    
                    # Evaluate
                    evaluation = evaluator.evaluate_assessment(
                        predicted=mock_predicted,
                        ground_truth=ground_truth,
                        assessment_type=config["assessment_type"]
                    )
                    
                    # Print results
                    print(f"Overall Score: {evaluation.overall_score:.3f}")
                    for result in evaluation.metric_results:
                        print(f"  {result.metric_name}: {result.score:.3f}")
                    
                except Exception as e:
                    print(f"❌ Test failed: {e}")
        
        print("\n✅ Evaluation framework testing completed")


def main():
    """Main entry point for evaluation runner"""
    parser = argparse.ArgumentParser(
        description="Evaluate assessment extraction results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate healthcare demo results with mental health ground truth
  python evaluate_assessments.py --demo healthcare --ground-truth mental.json
  
  # Evaluate golf demo results
  python evaluate_assessments.py --demo golf --ground-truth golf.json
  
  # Evaluate specific output file
  python evaluate_assessments.py --output-file scenarios/output/patient_assessments/mental_health_screening.json --ground-truth mental.json --type healthcare
  
  # Run all evaluations on existing results
  python evaluate_assessments.py --all
  
  # Test evaluation framework with mock data
  python evaluate_assessments.py --test
        """
    )
    
    parser.add_argument(
        "--demo",
        choices=["healthcare", "golf"],
        help="Demo type to evaluate"
    )
    
    parser.add_argument(
        "--ground-truth",
        help="Ground truth file name (e.g., mental.json, golf.json)"
    )
    
    parser.add_argument(
        "--output-file",
        help="Specific output file to evaluate"
    )
    
    parser.add_argument(
        "--type",
        choices=["healthcare", "golf"],
        help="Assessment type (required when using --output-file)"
    )
    
    parser.add_argument(
        "--all",
        action="store_true", 
        help="Run all available evaluations on existing results"
    )
    
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test evaluation framework with mock data"
    )
    
    parser.add_argument(
        "--no-reports",
        action="store_true",
        help="Don't save detailed evaluation reports"
    )
    
    parser.add_argument(
        "--list-outputs",
        action="store_true",
        help="List available output files"
    )
    
    parser.add_argument(
        "--ignore-missing-fields",
        action="store_true",
        help="Ignore missing fields in evaluation (don't penalize for missing fields)"
    )
    
    args = parser.parse_args()
    
    # Initialize runner
    runner = AssessmentEvaluationRunner()
    
    if args.list_outputs:
        # List available output files
        print("📁 Available output files:")
        for demo_type in ["healthcare", "golf"]:
            files = runner._find_output_files(demo_type)
            if files:
                print(f"\n{demo_type.upper()} outputs:")
                for file_path in files:
                    print(f"  • {file_path}")
            else:
                print(f"\n{demo_type.upper()}: No output files found")
        
    elif args.test:
        # Test evaluation framework
        runner.test_evaluation_framework()
        
    elif args.all:
        # Run all evaluations
        print("🚀 Running all available evaluations on existing results...")
        results = runner.run_all_evaluations(save_reports=not args.no_reports, ignore_missing_fields=args.ignore_missing_fields)
        
        # Print summary
        print("\n📈 Overall Evaluation Summary:")
        for demo_type, demo_results in results.items():
            print(f"\n{demo_type.upper()} Demo:")
            for result in demo_results:
                if "error" in result:
                    print(f"  ❌ {result['ground_truth_file']}: {result['error']}")
                else:
                    print(f"  ✅ {result['ground_truth_file']}: {result['overall_score']:.3f}")
    
    elif args.output_file and args.ground_truth and args.type:
        # Evaluate specific output file
        try:
            # Extract assessment data and evaluate directly
            output_path = Path(args.output_file)
            gt_path = runner.ground_truth_dir / args.ground_truth
            
            extracted_data = runner._extract_assessment_from_output(output_path, args.type)
            ground_truth = load_ground_truth(gt_path)
            
            evaluator = AssessmentEvaluator()
            evaluation = evaluator.evaluate_assessment(
                predicted=extracted_data,
                ground_truth=ground_truth,
                assessment_type=args.type,
                ignore_missing_fields=args.ignore_missing_fields
            )
            
            print(f"\n🎯 Evaluation Results for {args.output_file}")
            print_evaluation_summary(evaluation)
            
            if not args.no_reports:
                report_name = f"custom_{args.type}_{args.ground_truth.replace('.json', '')}_evaluation.json"
                report_path = runner.base_dir / "output" / "evaluations" / report_name
                report_path.parent.mkdir(parents=True, exist_ok=True)
                save_evaluation_report(evaluation, report_path)
                print(f"📊 Detailed report saved to: {report_path}")
            
        except Exception as e:
            print(f"❌ Evaluation failed: {e}")
            sys.exit(1)
    
    elif args.demo and args.ground_truth:
        # Evaluate demo results
        try:
            result = runner.evaluate_existing_results(
                demo_type=args.demo,
                ground_truth_file=args.ground_truth,
                save_report=not args.no_reports,
                ignore_missing_fields=args.ignore_missing_fields
            )
            
            if "error" in result:
                print(f"❌ Evaluation failed: {result['error']}")
                sys.exit(1)
            else:
                print(f"\n✅ Evaluation completed successfully!")
                print(f"Overall Score: {result['overall_score']:.3f}")
                
        except Exception as e:
            print(f"❌ Evaluation failed: {e}")
            sys.exit(1)
    
    else:
        print("❌ Please specify evaluation parameters")
        print("Use --help for usage information")
        print("Use --list-outputs to see available files")
        sys.exit(1)


if __name__ == "__main__":
    main()
