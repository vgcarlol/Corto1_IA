import numpy as np
import gymnasium as gym

env = gym.make("FrozenLake-v1", is_slippery=True, render_mode=None)

# Parámetros del agente
taza_aprendizaje = 0.1  # Tasa de aprendizaje
gamma = 0.99  # Factor de descuento
epsilon = 1.0  # Probabilidad inicial de exploración
epsilon_decay = 0.995  # Factor de decaimiento de epsilon
epsilon_min = 0.01  # Valor mínimo de epsilon
num_episodios = 10000  # Número de episodios

# Inicializamos la tabla Q con ceros
tablaQ = np.zeros((env.observation_space.n, env.action_space.n))

def elegir_accion(state):
    if np.random.rand() < epsilon:
        return env.action_space.sample()  # Exploración
    else:
        return np.argmax(tablaQ[state, :])  # Explotación

# Bucle de entrenamiento
for episode in range(num_episodios):
    estado, _ = env.reset()
    done = False
    
    while not done:
        accion = elegir_accion(estado)
        siguiente_estado, recompenza, done, _, _ = env.step(accion)
        
        # Actualización de la tabla Q con ecuación de Q-learning
        tablaQ[estado, accion] = tablaQ[estado, accion] + taza_aprendizaje * (
            recompenza + gamma * np.max(tablaQ[siguiente_estado, :]) - tablaQ[estado, accion]
        )
        
        estado = siguiente_estado
    
    # Reducir epsilon para menos exploración con el tiempo
    epsilon = max(epsilon * epsilon_decay, epsilon_min)
    
    # Mostrar progreso cada 1000 episodios
    if episode % 1000 == 0:
        print(f"Episodio {episode} - Epsilon: {epsilon:.3f}")

print("Entrenamiento terminado.")


# Evaluación del agente
def evaluate_agent(num_tests=100):
    success_count = 0
    total_rewards = 0
    
    for _ in range(num_tests):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action = np.argmax(tablaQ[state, :])  # Elegir la mejor acción
            state, reward, done, _, _ = env.step(action)
            episode_reward += reward
        
        total_rewards += episode_reward
        if episode_reward > 0:  # Si llegó a la meta
            success_count += 1
    
    success_rate = success_count / num_tests * 100
    avg_reward = total_rewards / num_tests
    
    print(f"Tasa de éxito: {success_rate:.2f}%")
    print(f"Recompensa media por episodio: {avg_reward:.2f}")

# Ejecutar evaluación
evaluate_agent()