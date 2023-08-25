
import numpy as np

states = ["increase", "decrease", "stable"] #Markov states
transition_probs = np.array([[0.6, 0.3, 0.1], [0.4, 0.4, 0.2], [0.5, 0.3, 0.2]])

num_steps = 10 #time-steps for simulation
def MC_states(current_state):
  future_states = []
  for i in range(num_steps):
    probs = transition_probs[states.index(current_state)]
    new_state = np.random.choice(states, p = probs)
    future_states.append(new_state)
    current_state = new_state #Update current state
  return future_states

#output
MC_states("increase")
