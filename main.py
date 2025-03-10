import numpy as np
import gymnasium as gym
import random
import os
import matplotlib.pyplot as plt


random_seed = random.randint(0, 10000)
np.random.seed(random_seed)

env = gym.make("FrozenLake-v1", desc=None, map_name="4x4", is_slippery=True, render_mode=None)
env.reset(seed=random_seed)


learning_rate = 0.1  
gamma = 0.99  
epsilon = 1.0  
epsilon_decay = 0.9995  
epsilon_min = 0.05  
num_episodios = 15000  


q_table_file = "q_table.npy"
if os.path.exists(q_table_file):
    tablaQ = np.load(q_table_file)  
    print("Tabla Q cargada desde archivo.")
else:
    tablaQ = np.zeros((env.observation_space.n, env.action_space.n))
    print("Nueva tabla Q inicializada.")


def elegir_accion(state, epsilon):
    if np.random.rand() < epsilon:
        return env.action_space.sample()  
    else:
        return np.argmax(tablaQ[state, :])  


success_rates = []  
avg_rewards = []  

for episode in range(num_episodios):
    estado, _ = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        accion = elegir_accion(estado,epsilon)
        siguiente_estado, recompensa, done, _, _ = env.step(accion)

        
        if done:
            if recompensa == 1:
                recompensa = 10  
            else:
                recompensa = -1  

        
        tablaQ[estado, accion] = tablaQ[estado, accion] + learning_rate * (
            recompensa + gamma * np.max(tablaQ[siguiente_estado, :]) - tablaQ[estado, accion]
        )

        estado = siguiente_estado
        total_reward += recompensa

    
    
    epsilon = max(epsilon * epsilon_decay, epsilon_min)
    
    
    if episode % 1000 == 0:
        success_count = 0
        total_rewards = 0
        for _ in range(100):
            state, _ = env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                action = np.argmax(tablaQ[state, :])  
                state, reward, done, _, _ = env.step(action)
                episode_reward += reward
            
            total_rewards += episode_reward
            if episode_reward > 0:
                success_count += 1
        
        success_rate = (success_count / 100) * 100
        avg_reward = total_rewards / 100
        success_rates.append(success_rate)
        avg_rewards.append(avg_reward)
        print(f"Episodio {episode} - Éxito: {success_rate:.2f}% - Epsilon: {epsilon:.3f}")


np.save(q_table_file, tablaQ)
print("Tabla Q guardada.")


plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(0, num_episodios, 1000), success_rates)
plt.xlabel("Episodios")
plt.ylabel("Tasa de éxito (%)")
plt.title("Evolución de la tasa de éxito")

plt.subplot(1, 2, 2)
plt.plot(range(0, num_episodios, 1000), avg_rewards)
plt.xlabel("Episodios")
plt.ylabel("Recompensa media")
plt.title("Evolución de la recompensa media")
plt.show()


def evaluate_agent(num_tests=100):
    success_count = 0
    total_rewards = 0
    
    for _ in range(num_tests):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action = np.argmax(tablaQ[state, :])  
            state, reward, done, _, _ = env.step(action)
            episode_reward += reward
        
        total_rewards += episode_reward
        if episode_reward > 0:  
            success_count += 1
    
    success_rate = success_count / num_tests * 100
    avg_reward = total_rewards / num_tests
    
    print(f"Tasa de éxito final: {success_rate:.2f}%")
    print(f"Recompensa media final: {avg_reward:.2f}")


evaluate_agent()


def play_agent():
    env_render = gym.make("FrozenLake-v1", desc=None, map_name="4x4", is_slippery=True, render_mode="human")
    state, _ = env_render.reset()
    done = False
    
    while not done:
        action = np.argmax(tablaQ[state, :])
        state, _, done, _, _ = env_render.step(action)
    
    env_render.close()

print("\nMostrando cómo juega el agente...")
play_agent()

