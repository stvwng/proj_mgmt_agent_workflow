# agentic_workflow.py

from agents.workflow_agents.base_agents import ActionPlanningAgent, KnowledgeAugmentedPromptAgent, EvaluationAgent, RoutingAgent
from openai import OpenAI

openai_instance = OpenAI()

# Load product spec
file_path = "Product-Spec-Email-Router.txt"
try:
    with open(file_path, 'r') as file:
        product_spec = file.read()
    # print("File content:")
    # print(product_spec)
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")

# Action Planning Agent
knowledge_action_planning = (
    "Stories are defined from a product spec by identifying a "
    "persona, an action, and a desired outcome for each story. "
    "Each story represents a specific functionality of the product "
    "described in the specification. \n"
    "Features are defined by grouping related user stories. \n"
    "Tasks are defined for each story and represent the engineering "
    "work required to develop the product. \n"
    "A development Plan for a product contains all these components"
)
action_planning_agent = ActionPlanningAgent(openai_instance, knowledge_action_planning)

# Product Manager - Knowledge Augmented Prompt Agent
instructions_product_manager = "You are a Product Manager, you are responsible for defining the user stories for a product."
knowledge_product_manager = f"""
    Stories are defined by writing sentences with a persona, an action, and a desired outcome.
    The sentences always start with: As a 
    Write several stories for the product spec below, where the personas are the different users of the product.
    
    PRODUCT SPEC:
    {product_spec}
    """
product_manager_knowledge_agent = KnowledgeAugmentedPromptAgent(
                                    openai_instance=openai_instance, 
                                    knowledge=knowledge_product_manager, 
                                    instructions=instructions_product_manager,
                                    name="product_manager",
                                    description="Responsible for defining product personas and user stories only. Does not define features or tasks. Does not group stories",
                                    func=lambda: print(f"Replace with support function"))

# Product Manager - Evaluation Agent
instructions_product_manager_evaluation_agent = "You are an evaluation agent that checks the answers of other worker agents"
product_manager_evaluation_criteria = "The answer should be stories that follow the following structure: As a [type of user], I want [an action or feature] so that [benefit/value]."
product_manager_evaluation_agent = EvaluationAgent(
                                    openai_instance=openai_instance, 
                                    instructions=instructions_product_manager_evaluation_agent, 
                                    evaluation_criteria=product_manager_evaluation_criteria, 
                                    worker_agent=product_manager_knowledge_agent,
                                    )

# Program Manager - Knowledge Augmented Prompt Agent
instructions_program_manager = "You are a Program Manager, you are responsible for defining the features for a product."
knowledge_program_manager = "Features of a product are defined by organizing similar user stories into cohesive groups."
program_manager_knowledge_agent = KnowledgeAugmentedPromptAgent(
                                    openai_instance=openai_instance, 
                                    knowledge=knowledge_program_manager, 
                                    instructions=instructions_program_manager,
                                    name="program_manager",
                                    description="Responsible for defining the features for a product. Groups stories. Does not define development tasks for a product.",
                                    func=lambda: print(f"Replace with support function")) 
# Program Manager - Evaluation Agent
instructions_program_manager_evaluation_agent = "You are an evaluation agent that checks the answers of other worker agents."
program_manager_evaluation_criteria =  """The answer should be product features that follow the following structure:  
                                       Feature Name: A clear, concise title that identifies the capability
                                       Description: A brief explanation of what the feature does and its purpose
                                       Key Functionality: The specific capabilities or actions the feature provides
                                       User Benefit: How this feature creates value for the user"""
program_manager_evaluation_agent = EvaluationAgent(
                                    openai_instance=openai_instance, 
                                    instructions=instructions_program_manager_evaluation_agent, 
                                    evaluation_criteria=program_manager_evaluation_criteria, 
                                    worker_agent=program_manager_knowledge_agent)

# Development Engineer - Knowledge Augmented Prompt Agent
instructions_dev_engineer = "You are a Development Engineer, you are responsible for defining the development tasks for a product."
knowledge_dev_engineer = "Development tasks are defined by identifying what needs to be built to implement each user story."
dev_engineer_agent = KnowledgeAugmentedPromptAgent(
                                    openai_instance=openai_instance, 
                                    instructions=instructions_dev_engineer, 
                                    knowledge=knowledge_dev_engineer,
                                    name="dev_engineer",
                                    description="Responsible for defining the development tasks for a product.",
                                    func=lambda : print("Replace with support function")) #

# Development Engineer - Evaluation Agent
instructions_dev_engineer_eval = "You are an evaluation agent that checks the answers of other worker agents."
dev_engineer_evaluation_criteria = """The answer should be tasks following this exact structure: 
                     Task ID: A unique identifier for tracking purposes
                     Task Title: Brief description of the specific development work
                     Related User Story: Reference to the parent user story
                     Description: Detailed explanation of the technical work required
                     Acceptance Criteria: Specific requirements that must be met for completion
                     Estimated Effort: Time or complexity estimation
                     Dependencies: Any tasks that must be completed first"""
dev_engineer_evaluation_agent = EvaluationAgent(
                                    openai_instance=openai_instance, 
                                    instructions=instructions_dev_engineer_eval, 
                                    evaluation_criteria=dev_engineer_evaluation_criteria, 
                                    worker_agent=dev_engineer_agent)

# Routing Agent
def generic_support_function(
    worker_agent: KnowledgeAugmentedPromptAgent,
    eval_agent: EvaluationAgent,
    input: str # step from action plan
):
    worker_response = worker_agent.get_response_text(input)
    final_response_dict = eval_agent.evaluate(worker_response)
    return final_response_dict["final_response"]

workers = [product_manager_knowledge_agent, program_manager_knowledge_agent, dev_engineer_agent]
workers2evals = {
    product_manager_knowledge_agent: product_manager_evaluation_agent,
    program_manager_knowledge_agent: program_manager_evaluation_agent,
    dev_engineer_agent: dev_engineer_evaluation_agent
}
for worker in workers:
    worker.agent_dict["func"] = lambda input: generic_support_function(worker, workers2evals[worker], input)
worker_dicts = [worker.agent_dict for worker in workers]
routing_agent = RoutingAgent(
                    openai_instance=openai_instance,
                    agents=worker_dicts)

# Job function persona support functions
# TODO: 11 - Define the support functions for the routes of the routing agent (e.g., product_manager_support_function, program_manager_support_function, development_engineer_support_function).
# Each support function should:
#   1. Take the input query (e.g., a step from the action plan).
#   2. Get a response from the respective Knowledge Augmented Prompt Agent.
#   3. Have the response evaluated by the corresponding Evaluation Agent.
#   4. Return the final validated response.

# Run the workflow

print("\n*** Workflow execution started ***\n")
# Workflow Prompt
# ****
workflow_input = "What would the development tasks for this product be?"
# ****
print(f"Task to complete in this workflow, workflow prompt = {workflow_input}")

print("\nDefining workflow steps from the workflow prompt")
# TODO: 12 - Implement the workflow.
#   1. Use the 'action_planning_agent' to extract steps from the 'workflow_prompt'.
#   2. Initialize an empty list to store 'completed_steps'.
#   3. Loop through the extracted workflow steps:
#      a. For each step, use the 'routing_agent' to route the step to the appropriate support function.
#      b. Append the result to 'completed_steps'.
#      c. Print information about the step being executed and its result.
#   4. After the loop, print the final output of the workflow (the last completed step).

workflow_steps = action_planning_agent.extract_steps_from_input(workflow_input)
completed_steps = []

for step in workflow_steps:
    print(f"Executing Step Number {workflow_steps.index(step)}")
    step_result = routing_agent.route_prompt(step)
    print(f"Step {workflow_steps.index(step)} Result: {step_result} \n")
    completed_steps.append(step_result)
    
print("FINAL RESULT")
print(completed_steps[-1])