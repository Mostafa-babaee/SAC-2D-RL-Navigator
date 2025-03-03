# Reinforcement Learning-Based Autonomous Agent Navigation

A reinforcement learning (RL) framework using the **Soft Actor-Critic (SAC) algorithm** to train an autonomous agent for optimal movement in a custom **2D environment**.

## Features  

 Custom **2D physics-based environment** (PyBullet)  
 **Soft Actor-Critic (SAC)** Reinforcement learning  
 Optimized **movement strategies** through reward-based learning  
 **Training progress tracking** with performance metrics  

## Installation  

### Prerequisites  
Ensure you have the following installed:  

- **Python 3.7+**  
- **PyTorch**  
- **PyBullet**  
- **Matplotlib**  
- **NumPy**  
## Setup  

```md
Clone the repository and install dependencies:  

git clone https://github.com/mostafa-babaee/SAC-2D-RL-Navigator.git  
cd SAC-2D-RL-Navigator  
pip install -r requirements.txt  
python train_agent.py
``` 

## Methodology  

- **Environment:** A bounded **2D space** with a **robot agent** (red) and a **target** (blue).  
- **State & Actions:**  
  - **State:** `[x_robot, y_robot, x_target, y_target]`  
  - **Actions:** `[Δx, Δy]` (step size ≤ 0.2)
- **Reward Function:**  
  ```math
  R = λ (prev\_distance - current\_distance) + P

 ## Results  

 **Performance Improvements:**  
 *Increased rewards* over training episodes  
 *Faster navigation* (steps reduced from ~200 to **~25-75**)  
 *High success rate* (~100% after training)  

### **Training Progress Visualization**  

#### **Episode 1: Random Movement**  
![Episode 1](https://i.imgur.com/yqXwA8T.png)

#### **Episode 2000: Improved Navigation**  
![Episode 2000](https://i.imgur.com/ONi3sce.png)
#### **Episode 5000: Optimized Path**  
![Episode 5000](https://i.imgur.com/7rpNndE.png) 

### **Performance Metrics**  
#### **Reward Progression Over Episodes**  
![Training Progress](https://i.imgur.com/clw3HeB.png) 

#### **Steps Taken Per Episode**  
![Steps Reduction](https://i.imgur.com/oimAdB3.png)  

#### **Success Rate Over Time**  
![Success Rate](https://i.imgur.com/xzbIBNI.png)  



