# proj_mgmt_agent_workflow
Repo for Project Management Agentic AI workflow

This is an AI agentic workflow that manages a technical project.

To run the workflow: python agentic_workflow.py

1. ActionPlanningAgent: This AI agent creates a plan for other agents to execute
2. RoutingAgent: This AI agent directs prompts to the agent best suited to respond to that prompt.
3. KnowledgeAugmentedPromptAgent: This AI agent responds to prompts; its knowledge base is augmented with instructions and knowledge provided by the user.
4. EvaluationAgent: This AI agent evaluates responses by other agents and provides feedback, enabling the other agents to improve their responses.
5. BaseAgent: This is the parent class for ActionPlanningAgent, KnowledgeAugmentedPromptAgent, and EvaluationAgent.
