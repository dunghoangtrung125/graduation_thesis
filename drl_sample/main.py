from q_learning_agent import q_learning_agent
from deep_q_learning_agent import deep_q_learning_agent

agent = deep_q_learning_agent(dueling=True)
agent.learning()