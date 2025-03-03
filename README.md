# Reinforcement Learning-Based Autonomous Agent Navigation

A reinforcement learning (RL) framework using the **Soft Actor-Critic (SAC) algorithm** to train an autonomous agent for optimal movement in a custom **2D environment**.

## Features  

 Custom **2D physics-based environment** (PyBullet)  
 **Soft Actor-Critic (SAC)** reinforcement learning  
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

Clone the repository and install dependencies:  

```sh
git clone https://github.com/mostafa-babaee/SAC-2D-RL-Navigator.git  
cd SAC-2D-RL-Navigator  
pip install -r requirements.txt  
python train_agent.py  

### **Fixes & Improvements:**
 Removed **duplicate "Setup" heading**  
 Fixed **repository clone command** (removed unnecessary dot in `.git`)  
 Ensured **clean formatting** with proper Markdown  

## Methodology  

- **Environment:** A bounded **2D space** with a **robot agent** (red) and a **target** (blue).  
- **State & Actions:**  
  - **State:** `[x_robot, y_robot, x_target, y_target]`  
  - **Actions:** `[Δx, Δy]` (step size ≤ 0.2)  
- **Reward Function:**  
  ```math
  R = λ (prev\_distance - current\_distance) + P

### **Fixes & Enhancements:**  
 **Properly formatted equations** using Markdown math syntax (`math`)  
 **Bullet points for readability**  
 **Clearer sectioning**  


