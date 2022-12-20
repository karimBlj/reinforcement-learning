import numpy as np

def valueIteration(
	max_iterations : int,
	escompte       : float,
	transition     : np.array,
	reward         : np.array,
	nb_state       : int,
	nb_action      : int,
):
	Vk = np.zeros(nb_state)
	Vk1 = np.zeros(nb_state)
	for _ in range(max_iterations):
		for s in range(nb_state):
			L = np.zeros(nb_action)
			for a in range(nb_action):
				value = 0
				for s2 in range(nb_state):
					value += transition[s,a,s2]*Vk[s2]
				value *= escompte
				value += reward[s,a]
				L[a] = value
			Vk1[s] = np.max(L)
		Vk = np.copy(Vk1)
	return Vk1



def policyIteration(
	max_iterations : int,
	escompte       : int,
	transition     : np.array,
	reward         : np.array,
	nb_state       : int,
	nb_action      : int,
):
	Vk = np.zeros(nb_state)
	Vk1 = np.zeros(nb_state)
	policy = np.ones(nb_state)
	policy = policy.astype('int')
	while(not policy_stable):
		policy_stable = True
		for _ in range(max_iterations):
			for s in range(nb_state):
				L = np.zeros(nb_action)
				value = 0
				for s2 in range(nb_state):
					value += transition[s,policy[s],s2]*Vk[s2]
				value *= escompte
				value += reward[s,policy[s]]
				Vk1[s] = value
			Vk = np.copy(Vk1)
		for s in range(nb_state):
			old_policy = policy[s]
			tmp = np.zeros(nb_action)
			
			for a in range(nb_action):
				value = 0
				for s2 in range(nb_state):
					value += transition[s,policy[s],s2]*Vk[s2]
				value *= escompte
				value += reward[s,a]
				tmp[a] = value
			policy[s] = np.argmax(tmp)
			if(policy[s] != old_policy):
				policy_stable = False
	return Vk1


def epsGreedy(
	epsilon       : float,
	vector_values : np.array
):
	if np.random.rand() < epsilon:
		action = np.random.randint(len(vector_values))
		return action
	return np.argmax(vector_values)


def sarsa(
	n_episode    : int,
	gamma        : float,
	alpha        : float,
	epsilon      : float,
	n_states     : int,
	n_action     : int,
	reward_decay : float = 0.001,
):
	reward_per_episode = []
	vector_values = np.zeros((n_states, n_action))

	for _ in range(n_episode):
		observation = env.reset()
		action = epsGreedy(epsilon, vector_values[observation, :])
		sum_reward = 0
		for _ in range(3000):
			observationP, reward, done, _ = env.step(action)
			sum_reward += reward

			if (done and reward == 0):
				reward = 0
			elif(not done):
				reward = -reward_decay

			actionP = epsGreedy(epsilon, vector_values[observationP,:])
			tmp     = alpha * (reward + gamma *  vector_values[observationP, actionP] - vector_values[observation, action])
			vector_values[observation, action] = vector_values[observation , action ] + tmp

			action      = actionP
			observation = observationP
			sum_reward += reward

			if done:
				break
		reward_per_episode.append(sum_reward)
	return vector_values, reward_per_episode



def q_learning(
	n_episode : int,
	gamma     : float,
	alpha     : float,
	epsilon   : float,
	n_states  : int,
	n_action  : int,
	reward_decay : float = 0.001,
):
	reward_per_episode = []
	vector_values = np.zeros((n_states, n_action))

	for _ in range(n_episode):
		observation = env.reset()
		action = epsGreedy(epsilon, vector_values[observation, :])
		sum_reward = 0

		for _ in range(3000):
			observationP, reward, done, _ = env.step(action)
			sum_reward += reward

			if (done and reward == 0):
				reward = 0
			elif(not done):
				reward = -reward_decay

			actionP = epsGreedy(epsilon, vector_values[observationP,:])
			if done:
				tmp = alpha * (reward	- vector_values[observation, action])
			else:
				mx = np.max(vector_values[observationP,:])
				tmp = alpha * (reward + gamma * mx - vector_values[observation, action])
			
			vector_values[observation, action] = vector_values[observation, action] + tmp

			action      = actionP
			observation = observationP
			sum_reward += reward

			if done:
				break
		reward_per_episode.append(sum_reward)
	return vector_values, reward_per_episode