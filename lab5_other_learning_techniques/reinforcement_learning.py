import numpy as np

# Initialize Q-table
Q = np.zeros((5, 2))        # 5 states, 2 actions
learning_rate = 0.1
discount_factor = 0.9
episodes = 100

for _ in range(episodes):
    state = np.random.randint(0, 5)
    action = np.random.randint(0, 2)
    next_state = (state + 1) % 5
    reward = 1 if next_state == 4 else 0
    Q[state, action] = Q[state, action] + learning_rate * (reward + discount_factor * np.max(Q[next_state]) - Q[state, action])

print("Q-Table:\n", Q)