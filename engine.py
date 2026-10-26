import pandas as pd
import time
from typing import List, Dict, Any
from wrapper import GroqWrapper, LLMJudge

class BatchEvaluationEngine:
    """Enterprise Batch Evaluation Engine with LLM-as-a-Judge and Cost Analytics."""

    def __init__(self, wrapper=None):
        self.wrapper = wrapper or GroqWrapper()
        self.judge = LLMJudge(self.wrapper)
        
        # Groq Llama3 8B Pricing Rates (per 1,000 tokens)
        self.prompt_cost_per_1k = 0.00005
        self.completion_cost_per_1k = 0.00008

    def calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculates total estimated cost in USD ($) based on token throughput."""
        prompt_cost = (prompt_tokens / 1000.0) * self.prompt_cost_per_1k
        completion_cost = (completion_tokens / 1000.0) * self.completion_cost_per_1k
        return round(prompt_cost + completion_cost, 6)

    def process_batch(self, input_file: str, output_file: str) -> List[Dict[str, Any]]:
        """Processes prompts in batch, scores responses using LLM-as-a-Judge, and tracks metrics."""
        df = pd.read_csv(input_file)
        results = []

        print(f"\n🚀 [Engine] Starting Enterprise Batch Evaluation for {len(df)} prompts...\n")

        for index, row in df.iterrows():
            prompt = row['prompt']
            start_time = time.time()
            
            try:
                # 1. Generate model response
                res = self.wrapper.generate({"prompt": prompt})
                latency = round(time.time() - start_time, 4)

                # 2. Perform LLM-as-a-Judge Evaluation
                judge_eval = self.judge.evaluate_response(prompt, res.text)

                # 3. Calculate Cost ($)
                cost = self.calculate_cost(res.prompt_tokens, res.completion_tokens)

                record = {
                    "prompt": prompt,
                    "response": res.text,
                    "latency": latency,
                    "prompt_tokens": res.prompt_tokens,
                    "completion_tokens": res.completion_tokens,
                    "total_tokens": res.total_tokens,
                    "cost_usd": cost,
                    "correctness_score": judge_eval.get("correctness_score", 3),
                    "tone": judge_eval.get("tone_analysis", "Neutral"),
                    "safety_pass": judge_eval.get("safety_pass", True),
                    "judge_reasoning": judge_eval.get("reasoning", ""),
                    "status": "SUCCESS"
                }

            except Exception as e:
                import traceback
                print(f"\n❌ EXCEPTION CAUGHT ON PROMPT {index + 1}:")
                traceback.print_exc()
                
                latency = round(time.time() - start_time, 4)
                record = {
                    "prompt": prompt,
                    "response": f"ERROR: {str(e)}",
                    "latency": latency,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "cost_usd": 0.0,
                    "correctness_score": 0,
                    "tone": "N/A",
                    "safety_pass": False,
                    "judge_reasoning": f"Error: {str(e)}",
                    "status": "FAILED"
                }

            results.append(record)
            print(f"✅ Finished Prompt {index + 1}/{len(df)} | Latency: {latency}s | Cost: ${record['cost_usd']} | Judge Score: {record['correctness_score']}/5")

        # Save to Output CSV
        out_df = pd.DataFrame(results)
        out_df.to_csv(output_file, index=False)
        print(f"\n🎉 [Engine] Evaluation Completed! Results saved to '{output_file}'.")

        return results