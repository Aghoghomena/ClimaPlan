# eval.py
import asyncio
import json
import difflib
import random
from typing import Dict, Any, List

from pydantic import BaseModel, Field
from langsmith import Client, traceable
from langsmith.schemas import Run, Example

from langchain.messages import SystemMessage, HumanMessage

import index
from config import llm
from utility import get_baseline_code
from prompts import get_graph_system_prompt, get_prompt_subgraph2

from models import (

  RecommendationState,
  CorrectnessEvalSchema,
  ReflectionQualitySchema
)


# -----------------------------
# LangSmith client
# -----------------------------
client = Client()

def _iter_runs(root: Run):
    stack = [root]
    while stack:
        r = stack.pop()
        yield r
        for c in (getattr(r, "child_runs", None) or []):
            stack.append(c)

def eval_no_tool_or_llm_errors(run: Run, example: Example) -> Dict[str, Any]:
    """
    Top-tier reliability evaluator:
    Fails if ANY tool/LLM/child run has an error.
    """
    errors: List[str] = []

    for r in _iter_runs(run):
        run_type = getattr(r, "run_type", "") or ""
        name = getattr(r, "name", None) or "unnamed"

        # 1) Explicit run.error (most important)
        err = getattr(r, "error", None)
        if err:
            errors.append(f"{run_type}:{name} error={err}")

        # 2) Common error fields in outputs
        outs = r.outputs or {}
        if isinstance(outs, dict):
            for k in ("Error", "error", "exception", "traceback"):
                v = outs.get(k)
                if v:
                    errors.append(f"{run_type}:{name} outputs[{k}]={str(v)[:300]}")

            # 3) Provider-specific patterns (Groq tool use failures often appear in text)
            for k in ("output", "content", "message"):
                v = outs.get(k)
                if isinstance(v, str) and ("tool_use_failed" in v or "invalid_request_error" in v):
                    errors.append(f"{run_type}:{name} outputs[{k}] contains tool_use_failed")

    ok = len(errors) == 0
    return {
        "key": "no_tool_or_llm_errors",
        "score": 1.0 if ok else 0.0,
        "comment": "No tool/LLM errors detected" if ok else (" | ".join(errors)[:4000]),
    }
# -----------------------------
# Prompts
# -----------------------------
evaluator_prompt = """
You are evaluating Python code generated to read a CSV and compute analysis to find anomalies.

Review the generated code for correctness, errors, quality, efficiency, safety, usability, and best practices.

Generated code:
{generated_code}

It was generated for this prompt:
{prompt}

Benchmark code:
{expected_code}

Give a score from 0 to 100 (100 = excellent; better than benchmark; 0 = unusable).
Also give a failingscore from 0 to 100 (likelihood it fails).

Return structured output.
"""

evaluator_prompt2 = "You are a principal Python engineer and a strict code evaluator."


# -----------------------------
# Evaluators (LLM-as-judge)
# -----------------------------
@traceable(name="evaluate_code_quality")
def evaluate_code_quality(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """
    LLM-as-judge evaluator: code correctness/quality.
    NOTE: outputs is what your target() returned.
    """
    structured_llm = llm.with_structured_output(CorrectnessEvalSchema)

    generated_code = (outputs.get("generated_code") or "").strip()
    prompt = inputs.get("code_prompt", "").encode("utf-8").decode("unicode_escape")
    expected_code = (reference_outputs.get("baseline_code") or "").strip()

    messages = [
        SystemMessage(content=evaluator_prompt2),
        HumanMessage(content=evaluator_prompt.format(
            prompt=prompt,
            expected_code=expected_code,
            generated_code=generated_code,
        )),
    ]

    result = structured_llm.invoke(messages)
    
    # Include failingscore in the comment, and return only key + score + comment
    comment_with_failingscore = f"{result.comment}\n\nFailing Score (0-100, likelihood of failure): {result.failingscore}"
    
    return {
        "key": "code_quality",
        "score": int(result.score),
        "comment": comment_with_failingscore
    }


@traceable(name="evaluate_reflection_quality")
def evaluate_reflection_quality(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """
    Scores whether reflection output is a genuine improvement over the original recommendation.
    Requires your target() to return:
      - recommendation_summary (original)
      - reflection_summary (reflected/proposed improved)
      - today_weather (optional but helpful)
    """
    structured_llm = llm.with_structured_output(ReflectionQualitySchema)

    original = (outputs.get("recommendation_summary") or "").strip()
    reflected = (outputs.get("reflection_summary") or "").strip()
    today_weather = outputs.get("today_weather")

    if not original or not reflected:
        return {
            "key": "reflection_quality",
            "score": 0,
            "comment": "Missing recommendation_summary or reflection_summary in outputs; cannot judge reflection quality.",
        }

    prompt = f"""
You are evaluating a reflection step for a weather recommender.

Weather context (JSON, may be partial):
{json.dumps(today_weather, indent=2) if isinstance(today_weather, dict) else str(today_weather)}

Original recommendation summary:
{original}

Reflected (proposed improved) summary:
{reflected}

Score 0-100 based on:
- correctness & consistency with weather
- improved specificity/clarity vs original (not just rephrasing)
- conciseness
- safety (no risky advice)
- actionable usefulness

Return structured output: score, comment.
"""
    res = structured_llm.invoke([
        SystemMessage(content="You are a strict evaluator of weather recommendation quality."),
        HumanMessage(content=prompt),
    ])
    return {"key": "reflection_quality","score": int(res.score), "comment": res.comment}


# -----------------------------
# Dataset setup
# -----------------------------
randomnum = random.randint(1, 100000)
dataset_name = f"eval_code_prompt_{randomnum}"
client.create_dataset(dataset_name)

baseline = get_baseline_code().strip()

# IMPORTANT:
# - inputs are what your target(inputs) receives
# - reference_outputs is what evaluator receives as reference_outputs
client.create_examples(
    inputs=[{"code_prompt": get_prompt_subgraph2()}],
    outputs=[{
        "baseline_code": baseline,
        "reference_test_cases": baseline,  # keep if you still want it
    }],
    dataset_name=dataset_name,
)


# -----------------------------
# Target: run the graph and return all fields evaluators need
# -----------------------------
async def run_graph(inputs):
    compiled_graph = await index.create_graph()
        
        # Invoke the graph with standard input (just like in main())
    result = await compiled_graph.ainvoke({"messages": get_graph_system_prompt()})

    if hasattr(result, "model_dump"):
        result = result.model_dump()
    
    recommendation_state  = result["recommendation_state"]
    return {
        "generated_code": result.get("generated_code", ""),
        "recommendation_summary":" ".join(map(str, recommendation_state.original_recommendations)),
        "reflection_summary": " ".join(map(str, recommendation_state.reflected_recommendations)),
        "updated_summary": result.get("updated_summary") or result.get("final_output"),
        "approve_reflection": result.get("approve_reflection") or result.get("approve_update"),
        "today_weather": result.get("today_weather"),
        "Error": result.get("Error"),
    }

def target(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Sync wrapper for async target function"""
    out =  asyncio.run(run_graph(inputs))
    return out



# -----------------------------
# Run evaluation
# -----------------------------

# Run evaluation
results = client.evaluate(
    target,
    data=dataset_name,
    evaluators=[evaluate_code_quality,evaluate_reflection_quality, eval_no_tool_or_llm_errors],
    experiment_prefix="weather-analysis-main-graph",
    max_concurrency=1,
)

print(f"\n✅ Evaluation completed!")
print(f"Results: {results}")