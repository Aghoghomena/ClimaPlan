
import asyncio
import os
from dotenv import load_dotenv
from langsmith import Client
from langsmith import traceable
from typing import Dict, Any
import json
from langchain.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
import main
from config import llm
from utility import get_baseline_code
from prompts import get_graph_system_prompt, get_prompt_subgraph2
import random

# create LangSmith client
client = Client()
# Use a structured output to ensure LLM returns a score.  Also returns a comment to provide explanation of score.
class CorrectnessEvalSchema(BaseModel):
    """CLass to define the schema for correctness evaluation."""
    score: int = Field(description="An integer percentage score from 0 to 100 indicating the correctness of the code")
    comment: str = Field(description="An extensive explanation of the score")
    failingscore: int = Field(description="An integer percentage score from 0 to 100 indicating the likelyhood the code will fail")



# -----------------------------
# Evaluation
# -----------------------------
# Evaluate Subgraph2
evaluator_prompt = """"
You are evaluating python code for the generated for the prompt to read csv and compute analysis to find anomaly:
review the code for correctness, errors, quality, efficiency, safety, usablity and best practices of this {generated_code} generated for this {prompt}.
Reviewing against this benchmark code:
{expected_code}
Provide a score from 0 to 10, where 10 and would perform better than the benchmark means the code is of excellent quality and efficiency and 0 means the code is of very poor quality and efficiency.:
Score:
Provide the reasom for the score why you gave that score and a summary of the issues. Provide ways to fix the issues with the code and make it better,faster and optimised.
Comment:
"""
evaluator_prompt2 = "You are a python expert a Principal Python Engineer and a code evaluator."
@traceable(name="evaluate_code_quality")
def evaluate_code_quality(inputs: dict, outputs: dict, reference_outputs: dict ) -> dict:
    """Evaluate the quality of the code.s
    LLM-as-judge evaluator for code correctness, errors, quality, efficiency, safety, usablity and best practices of python.
    Returns a score from 0 to 10, an extensive explanation of the score"""
    print("evaluate_code_quality of the code genreated")
    print("input: at 45", outputs)

    structured_llm = llm.with_structured_output(CorrectnessEvalSchema)
    

    code = outputs.get("generated_code", "")
    benchmark_code = reference_outputs.get("baseline_code")
    theprompt= inputs.get("prompt")
    benchmark_code = reference_outputs["reference_test_cases"]
    messages = [
            SystemMessage(content=evaluator_prompt2),
            HumanMessage(content=evaluator_prompt.format(prompt=theprompt,
                expected_code=benchmark_code,
                generated_code=code
            ))
    ]
        
    # extract score and comment from response
    result = structured_llm.invoke(messages)
    print(f"Score: {result.score}, Comment: {result.comment}") 
    # return score and comment as a dict. LangSmith expects a dict return type with these keys.
    # You can also return just an integer score or boolean if you prefer.
    return { "score": result.score, "comment": result.comment }



#Create Dataset to test passes the prompt, basecode and the generatedcode
randomnum = random.randint(1, 100000)
dataset_name = f"eval_code_prompt{randomnum}"
dataset = client.create_dataset(dataset_name)
baseline = get_baseline_code().strip()

#read the code in baseline.py

# Create examples with proper input format
client.create_examples(
    inputs=[
        {
            "prompt": get_prompt_subgraph2(),  # Update with actual test file path
        }
    ],
    outputs=[
        {
            "baseline_code": baseline,
            "reference_test_cases": baseline  # For evaluator reference
        }
    ],
    dataset_name=dataset_name
)

#Runs the main graph created on main.py so i can get the generated_code
async def run_graph():
    """Target function for evaluation - runs main graph"""
    try:
        # Create the main graph
        compiled_graph = await main.create_graph()
        
        # Invoke the graph with standard input (just like in main())
        result = await compiled_graph.ainvoke({"messages": get_graph_system_prompt()})
        
        # Extract generated_code from the final state
        if hasattr(result, 'model_dump'):
            result = result.model_dump()
        
        generated_code = result.get("generated_code", "")
        if isinstance(generated_code, str) and generated_code.startswith('"'):
            try:
                generated_code = json.loads(generated_code)
            except:
                pass
        
        return {
            "generated_code": generated_code,
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "Error": f"Error in target function: {e}",
            "generated_code": ""
        }

# Sync wrapper for async target function and get the generated code from the graph 
def target(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Sync wrapper for async target function"""
    generated_code = asyncio.run(run_graph())

    # Return whatever you want evaluators to see
    return {
        "generated_code": generated_code
    }



# Run evaluation
results = client.evaluate(
    target,
    data=dataset_name,
    evaluators=[evaluate_code_quality],
    experiment_prefix="weather-analysis-main-graph",
    max_concurrency=1,
)

print(f"\n✅ Evaluation completed!")
print(f"Results: {results}")