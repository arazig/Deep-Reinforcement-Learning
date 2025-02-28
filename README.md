# Rainbow: Combining Improvements in Deep Reinforcement Learning

## Overview  
This project implements and evaluates the **Rainbow** algorithm, a combination of six extensions to the Deep Q-Network (DQN) algorithm, as described in the paper *"Rainbow: Combining Improvements in Deep Reinforcement Learning"* by Matteo Hessel et al. The Rainbow algorithm integrates several advancements in deep reinforcement learning (DRL) to achieve state-of-the-art performance on the Atari 2600 benchmark. This repository provides a Python implementation of Rainbow, along with experiments and ablation studies to demonstrate the contributions of each component to overall performance.

## Authors  
   ▪️ **`Abdelilah YOUNSI`**
   ▪️ **`Amine RAZIG`**
   ▪️ **`Yassine OJ`**

## Table of Contents  
1. [Objective](#objective)  
2. [Rainbow Components](#rainbow-components)  
3. [Experiments and Results](#experiments-and-results)  
4. [How to Run the Project](#how-to-run-the-project)  
5. [Requirements](#requirements)  
6. [References](#references)  

## Objective  
The primary objective of this project is to:  
1. Implement the Rainbow algorithm, combining six key extensions to DQN
2. Evaluate the performance of Rainbow on `XXXXXXXXXXX`
3. Conduct ablation studies to analyze the contribution of each component to the overall performance.  

## Rainbow Components  
The Rainbow algorithm integrates the following six improvements to DQN:  

1. **Double Q-Learning**: Reduces overestimation bias by decoupling action selection and evaluation.  
2. **Prioritized Experience Replay**: Prioritizes transitions with high temporal difference (TD) error for more efficient learning.  
3. **Dueling Networks**: Separates the estimation of state value and action advantages.  
4. **Multi-Step Learning**: Uses multi-step returns to propagate rewards more efficiently.  
5. **Distributional RL**: Models the distribution of returns instead of the expected value.  
6. **Noisy Nets**: Introduces noise into the network parameters to encourage exploration.  

## Experiments and Results  
The project includes the following experiments:  
1. **Full Rainbow Implementation**: Training and evaluation on the `XXXXXXX` benchmark.  
2. **Ablation Studies**: Isolating and evaluating the contribution of the components to overall performance.  

### Key Results  
- **Atari 2600 Benchmark**: Rainbow achieves state-of-the-art performance in terms of both data efficiency and final performance.  
- **Ablation Studies**: Each component contributes significantly to the overall performance, with Prioritized Experience Replay and Distributional RL providing the most substantial improvements.  

## How to Run the Project  

### Step 1: Clone the Repository  
```bash  
git clone https://github.com/arazig/Deep-Reinforcement-Learning.git
cd Rainbow-DRL 
```  

> **:warning: Attention:** There are two folders in this directory, labeled (1) and (2). The experiments are separate, with each folder having its own `requirements.txt` file.


### Step 2: Install Dependencies  
```bash  
pip install -r requirements.txt  
```  

### Step 3: Run Experiments  

#### Full Rainbow Implementation
```bash
 .py files on folder (1)-Experiments
```  

#### Ablation Studies  
To run ablation studies, use the following Notebook:  
```bash  
Notebook_experiments.ipynb
```  
Replace `"no_....."` components with the component you want to isolate (e.g., `"prioritized_replay"`, `"dueling"`, etc.).  

### Step 4: Visualize Results  
Results and training curves are provided notebooks to visualize and analyze the results:  

## Requirements  ( cf requierements)
- **Python 3.8+**  
- Libraries:  
  - `numpy`  
  - `pandas`  
  - `matplotlib`  
  - `seaborn`  
  - `gym` (Atari environments)  
  - `torch` (PyTorch for neural networks)  
  - `tensorboard` (for logging)  
- **Optional**: GPU for accelerated training.
- 
## Atari Experiments
This folder contains our implementation of Rainbow DQN specifically for Atari environments, with a focus on Ms. Pacman. Our implementation is based on and adapted from the [Tianshou](https://github.com/thu-ml/tianshou) reinforcement learning library. Note that this implementation requires its own setup distinct from the main project.

### Environment Setup for Atari
- **Python Version**: This implementation requires **Python 3.11** specifically
- **Virtual Environment**: We recommend creating a dedicated virtual environment:
  ```bash
  python3.11 -m venv atari_env
  source atari_env/bin/activate  # On Windows: atari_env\Scripts\activate
  ```
- **Dependencies**: Install the required packages from the requirements.txt file in the atari folder:
  ```bash
  cd atari_experiments
  pip install -r requirements.txt
  ```

### Project Files
- `atari_network.py`: Neural network architecture for the Atari agents (adapted from Tianshou)
- `atari_rainbow.py`: Implementation of Rainbow algorithm adapted for Atari games
- `atari_wrapper.py`: Custom wrappers for processing Atari environments (frame stacking, reward clipping, etc.)
- `video.py`: Utilities for recording trained agent gameplay
- `vis.py`: Visualization tools for analyzing agent performance

### Running Atari Experiments
To train a Rainbow agent on Ms. Pacman:
```bash
python atari_rainbow.py 
```

While our primary focus is on Ms. Pacman, the implementation supports other Atari games as well:
- MsPacman
- Pong
- Breakout
- SpaceInvaders
- Seaquest
- And other Atari environments

### Visualizing Results
To visualize training progress:
```bash
python vis.py 
```

### Recording Agent Gameplay
To record videos of a trained agent:
```bash
python video.py 
```

### Implementation Notes
- Our implementation uses frame stacking (4 frames) as input to capture temporal information
- We use the standard Atari preprocessing: grayscale conversion, frame skipping, etc.
- The network architecture follows the original DQN design with modifications for Rainbow components
- Experiments were run using Python 3.11 in a virtual environment
- This implementation is specifically optimized for Atari environments with particular focus on Ms. Pacman
- The Ms. Pacman environment presents unique challenges due to its complex maze navigation, ghost avoidance, and pellet collection strategies

### Acknowledgements
We would like to acknowledge the [Tianshou](https://github.com/thu-ml/tianshou) library, which provided the foundation for our implementation. Tianshou is a flexible, modular, and well-documented reinforcement learning platform. Our code adapts and builds upon their excellent work.


## References  
- Matteo Hessel, Joseph Modayil, Hado van Hasselt, Tom Schaul, Georg Ostrovski, Will Dabney, Dan Horgan, Bilal Piot, Mohammad Azar, David Silver. *Rainbow: Combining Improvements in Deep Reinforcement Learning*. 2017.  
- Mnih, Volodymyr, et al. *Human-level control through deep reinforcement learning*. Nature, 2015.  
- Van Hasselt, Hado, Arthur Guez, and David Silver. *Deep Reinforcement Learning with Double Q-Learning*. AAAI, 2016.  
- Schaul, Tom, et al. *Prioritized Experience Replay*. ICLR, 2016.  
- Wang, Ziyu, et al. *Dueling Network Architectures for Deep Reinforcement Learning*. ICML, 2016.  
- Bellemare, Marc G., et al. *A Distributional Perspective on Reinforcement Learning*. ICML, 2017.  
- Fortunato, Meire, et al. *Noisy Networks for Exploration*. ICLR, 2018.  
