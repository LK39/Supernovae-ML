# Supernova Classification with Reinforcement Learning

This project applies a **Reinforcement Learning (RL)** approach to classify supernovae into **Type I** and **Type II** using a Deep Q-Network (DQN) enhanced with techniques such as Dueling Networks, Upper Confidence Bound (UCB) exploration, and Experience Replay.

---

## What this project does

Modern sky surveys generate thousands of transient astronomical events, including supernovae, that require classification. Traditional spectroscopic follow-ups are time-intensive. Our model aims to **automate the classification** using three key photometric features:

- `Gi` — Host galaxy brightness
- `GVhel` — Heliocentric radial velocity
- `mag` — Apparent magnitude of the supernova

The model is trained using **reinforcement learning**, simulating an agent that receives rewards based on correct classification decisions. This allows the agent to optimize its decision-making strategy over time.

---

## Repository Structure

```
Supernovae/
├── FinalDataset/                          # Final data used in model traning
├── Type II/                               # 1st approach data
├── Type IIP/                              # 1st approach data
├── AddType.py                             # Script to tag dataset with type info
├── augmented_sn_typeI.py                  # Data augmentation for Type I
├── augmented_sn_typeII.py                 # Data augmentation for Type II
├── filtered_specific_type_I_....xlsx      # Used to extract final data
├── filtered_specific_type_II_....xlsx     # Used to extract final data
├── Sternberg Supernova Catalog.csv        # Raw dataset from Sternberg catalog
├── SupernovaeRL.py                        # Main RL training & evaluation script
├── TypeCounter.py                         # Utility script for dataset inspection
├── Supernovae2/
│   └── Type II/                           # 2nd approach datasets
├── Analysis.py                            # Script for data analysis / exploration
├── SNaX.CSV                               # External dataset used
├── Supernovae.py                          # Alternative modeling script (if any)
├── requirements.txt                       # Python dependencies
└── README.md                              # This file
```


---

## Installation

> Make sure you are using **Python 3.10** or **Python 3.9**, as `tf-agents` and `tensorflow` may not support Python 3.11+ properly.

1. Create a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate     # on macOS/Linux
venv\Scripts\activate        # on Windows
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## How to run the demo

The entire training + evaluation pipeline is handled through one script:

```bash
python SupernovaeRL.py
```

This will:

- Load and preprocess the dataset from the `/supernova_data` folder
- Train the DQN agent over 3000 episodes
- Display training graphs:
  - Accuracy over time
  - Loss curve
  - Rewards
  - Epsilon and learning rate dynamics

> The model prints out key statistics every 1000 episodes (e.g., accuracy, reward, loss).

NB: Make sure to match the base-path in `SupernovaeRL.py` (now defined as: base_path = r'C:\Users\pazol\Programms\Supernovae') to the correct path of where you have stored the repository. 

---

## Output Example

During execution, you'll see console updates like:

```
Episode 3000/3000
Current Training Accuracy: 44.25%
Current Validation Accuracy: 42.19%
Current Loss: 140.88
Current Reward: -19.0
Average Reward (last 10 episodes): 15.00
Learning Rate: 0.000000
Epsilon: 0.0110
--------------------------------------------------
```

And visualizations like this will pop up:

- Training vs. Validation Accuracy
- Reward Curves
- Moving Averages
- Learning Rate & Epsilon Decay

---

## Notes

- The model uses **single-epoch observations** due to data limitations.
- The dataset is synthetically augmented using regression-based noise injection to balance class distribution.
- You can plug in your own `.csv` files with the columns `Gi`, `GVhel`, `mag`, and `Label`.

---


## 📄 License

This project is for academic purposes and follows the [MIT License](LICENSE) for educational use.


