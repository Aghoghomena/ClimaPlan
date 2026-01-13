# eval_code.py
import os
import sys
import json
from typing import Dict, Any
from dotenv import load_dotenv
from langsmith import Client
from langsmith import traceable
from pydantic import BaseModel, Field
from langchain.messages import SystemMessage, HumanMessage

load_dotenv()

# Import from main - this will load main.py
# Make sure to import after main.py is fully loaded
import main

# Get the subgraph and models from main
analysis_subgraph = main.analysis_subgraph
WeatherAnalysis = main.WeatherAnalysis
llm = main.llm
get_baseline_code = main.get_baseline_code

# Initialize LangSmith client
client = Client()

# Evaluation schema
class CorrectnessEvalSchema(BaseModel):
    """Schema for correctness evaluation"""
    score: int = Field(description="An integer percentage score from 0 to 100 indicating the correctness of the code")
    comment: str = Field(description="A brief explanation of the score")


# Evaluator prompt
evaluator_prompt2 = "You are a python expert, a Principal Python Engineer and a code evaluator."

@traceable(name="evaluate_code_quality")
def evaluate_code_quality(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """Evaluate the quality of the code.
    LLM-as-judge evaluator for code quality & efficiency.
    Returns a score from 0 to 100 and a short comment"""
    
    try:
        structured_llm = llm.with_structured_output(CorrectnessEvalSchema)
        
        code = outputs.get("generated_code", "")
        if not code:
            return {"score": 0, "comment": "No code generated"}
        
        # Parse JSON string if needed
        if isinstance(code, str) and code.startswith('"'):
            try:
                code = json.loads(code)
            except:
                pass
        
        benchmark_code = reference_outputs.get("reference_test_cases") or reference_outputs.get("baseline_code", "")
        
        evaluator_prompt = f"""You are evaluating python code for the task to read csv and compute analysis to find anomaly.

Review the code for correctness, errors, quality, efficiency, safety, usability and best practices:

Generated code:
{code}

Expected baseline code:
{benchmark_code}

Provide a score from 0 to 100, where 100 means the code is of excellent quality and efficiency and 0 means the code is of very poor quality and efficiency.
Provide a brief explanation of the score."""
        
        messages = [
            SystemMessage(content=evaluator_prompt2),
            HumanMessage(content=evaluator_prompt)
        ]
        
        result = structured_llm.invoke(messages)
        return {"score": result.score, "comment": result.comment}
    except Exception as e:
        return {"score": 0, "comment": f"Evaluation error: {e}"}


# Target function for evaluation
def target(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Target function for evaluation - runs analysis_subgraph"""
    try:
        # Convert inputs to WeatherAnalysis state format
        state_input = WeatherAnalysis(
            hist_file_path=inputs.get("hist_file_path"),
            hist_weather=inputs.get("hist_weather"),
            messages=inputs.get("messages", []),
            execution_retries=0,
            max_execution_retries=3
        )
        
        # Invoke the compiled subgraph
        result = analysis_subgraph.invoke(state_input.model_dump())
        
        # Extract generated_code - handle JSON string if needed
        generated_code = result.get("generated_code", "")
        if isinstance(generated_code, str) and generated_code.startswith('"'):
            try:
                generated_code = json.loads(generated_code)
            except:
                pass
        
        # Return outputs that evaluators can access
        return {
            "generated_code": generated_code,
            "analysis": result.get("analysis"),
            "anomaly": result.get("anomaly"),
            "stats": result.get("stats"),
            "Error": result.get("Error")
        }
    except Exception as e:
        return {
            "Error": f"Error in target function: {e}",
            "generated_code": ""
        }


# Run evaluation function
def run_evaluation():
    """Run LangSmith evaluation on analysis_subgraph"""
    dataset_name = "weather-analysis-eval"
    
    try:
        # Try to get existing dataset
        dataset = client.read_dataset(dataset_name=dataset_name)
        print(f"Using existing dataset: {dataset_name}")
    except Exception:
        # Create new dataset
        dataset = client.create_dataset(
            dataset_name=dataset_name,
            description="Evaluation dataset for weather analysis subgraph"
        )
        print(f"Created new dataset: {dataset_name}")
        
        try:
            baseline = get_baseline_code()
        except Exception as e:
            print(f"Warning: Could not get baseline code: {e}")
            baseline = ""
        
        # Create examples with proper input format
        # Update with actual test file path
        test_file_path = "data/historical_weather_compare_test.csv"  # Update this
        
        client.create_examples(
            inputs=[
                {
                    "hist_file_path": test_file_path,
                    "hist_weather": None,
                    "messages": [],
                }
            ],
            outputs=[
                {
                    "baseline_code": baseline,
                    "reference_test_cases": baseline
                }
            ],
            dataset_name=dataset_name
        )
        print(f"Created example in dataset: {dataset_name}")
    
    # Run evaluation
    print(f"\nRunning evaluation on dataset: {dataset_name}")
    print("This may take a while...")
    
    try:
        results = client.evaluate(
            target,
            data=dataset_name,
            evaluators=[evaluate_code_quality],
            experiment_prefix="weather-analysis-subgraph2",
            max_concurrency=1,
        )
        
        print(f"\n✅ Evaluation completed!")
        print(f"Results: {results}")
        return results
    except Exception as e:
        print(f"\n Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    print("Starting evaluation...")
    run_evaluation()