from openai import OpenAI
from .base_agent import BaseAgent

class EvaluationAgent(BaseAgent):
    
    def __init__(
        self, 
        openai_instance: OpenAI, 
        instructions: str, 
        evaluation_criteria: str, 
        worker_agent: BaseAgent, 
        max_interactions: int=5):
        
        super().__init__(openai_instance, instructions)
        self.evaluation_criteria = evaluation_criteria
        self.worker_agent = worker_agent
        self.max_interactions = max_interactions
        self.final_response_dict = dict()

    def evaluate(self, initial_input: str):
        # This method manages interactions between agents to achieve a solution.
        input_to_evaluate = initial_input

        for i in range(self.max_interactions):
            print(f"\n--- Interaction {i+1} ---")

            print(" Step 1: Worker agent generates a response to the input")
            print(f"Prompt:\n{input_to_evaluate}")
            response_from_worker = self.worker_agent.get_response_text(input_to_evaluate)
            print(f"Worker Agent Response:\n{response_from_worker}")

            print(" Step 2: Evaluator agent judges the response")
            eval_input = (
                f"Does the following answer: {response_from_worker}\n"
                f"Meet this criteria: {self.evaluation_criteria}"
                f"Respond Yes or No, and the reason why it does or doesn't meet the criteria."
            )
            evaluation = self.get_response_text(eval_input)
            print(f"Evaluator Agent Evaluation:\n{evaluation}")

            print(" Step 3: Check if evaluation is positive")
            if evaluation.lower().startswith("yes"):
                print("✅ Final solution accepted.")
                self.final_response_dict = {
                    "final_response": response_from_worker,
                    "evaluation": evaluation,
                    "num_iterations": i + 1
                }
                break
            else:
                print(" Step 4: Generate instructions to correct the response")
                input_to_fix_response = (
                    f"Provide instructions to fix an answer based on these reasons why it is incorrect: {evaluation}"
                )
                instructions_to_fix_answer = self.get_response_text(input_to_fix_response)
                print(f"Instructions to fix:\n{instructions_to_fix_answer}")

                print(" Step 5: Send feedback to worker agent for refinement")
                input_to_evaluate = (
                    f"The original prompt was: {initial_input}\n"
                    f"The response to that prompt was: {response_from_worker}\n"
                    f"It has been evaluated as incorrect.\n"
                    f"Make only these corrections, do not alter content validity: {instructions_to_fix_answer}"
                )
        return self.final_response_dict