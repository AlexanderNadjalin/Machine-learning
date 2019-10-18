"""
Very simple Reinforced learning program for "Multi-armed bandit".
"""

import numpy as np
import matplotlib.pyplot as plt
from loguru import logger


def play_bandit(a, variance=1.0):
    """

    Draw a bandit arm.
    :param a: Action to be taken.
    :param variance: Variance.
    :return: Reward value.
    """
    return np.random.normal(a, scale=np.sqrt(variance))


def explore(q):
    """

    Explore action. Draw another arm than exploit action.
    :param q: Current estimate.
    :return: Action value.
    """
    return np.random.randint(len(q))


def exploit(q):
    """

    Exploit action (greedy action). Draw arm with highest q.
    :param q: Current estimate.
    :return: Action value.
    """
    greedy_actions, = np.where(q == np.max(q))
    return np.random.choice(greedy_actions)


def act(q, epsilon=0.1):
    """

    Decide when to explore or when to exploit. Exploit randomly.
    :param q: Current estimate.
    :param epsilon: Probability of explore action.
    :return: Value of either action.
    """
    if np.random.random() > epsilon:
        return exploit(q)
    else:
        return explore(q)


def update_q(old_estimate, target, k):
    """

    Update running estimate.
    :param old_estimate: Previous level.
    :param target: New information.
    :param k: Counter.
    :return: New estimate.
    """
    step_size = 1./(k + 1)
    error = target - old_estimate
    return old_estimate + step_size * error


def simulate_agent(n=20, T=2500, epsilon=0.01, variance=1.0):
    """

    Agent for taking actions either "explore" or "exploit".
    :param n: Maximum reward level.
    :param T: Iterations.
    :param epsilon: Probability of explore action.
    :param variance: Variance.
    :return: Estimation, actions taken, rewards.
    """
    q = np.zeros(n)
    actions = np.zeros(T)
    rewards = np.zeros(T)
    for t in range(T):
        a = act(q, epsilon)
        reward = play_bandit(a, variance)
        q[a] = update_q(q[a], reward, 1)
        actions[t] = a
        rewards[t] = reward
    return q, actions, rewards


def main():
    q, actions, rewards = simulate_agent()

    # Plot actions
    plt.plot(actions)
    plt.xlabel('$t$')
    plt.ylabel('$a_t$')
    plt.show()

    # Plot rewards
    plt.plot(rewards)
    plt.xlabel('$t$')
    plt.ylabel('$r_t$')
    plt.show()


if __name__ == '__main__':
    logger.info('Started.')
    main()
    logger.info('Ended.')
