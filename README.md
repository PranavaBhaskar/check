# Multi-Agent Drug Discovery Platform

A modular, plug-and-play framework for molecular property prediction using Graph Neural Networks (GNNs) and multi-agent architectures.

## Features

### Models (Plug-and-Play)
- **GCN** - Graph Convolutional Network
- **GAT** - Graph Attention Network
- **GIN** - Graph Isomorphism Network
- **GraphSAGE** - Graph Sample and Aggregate
- **MPNN** - Message Passing Neural Network
- **AttentiveFP** - Attentive Fingerprint
- **AdvancedGNN** - Multi-pooling GNN with residual connections
- **EnsembleModel** - Combine multiple models

### Meta-Heuristic Optimization
- **Genetic Algorithm (GA)** - Evolution-inspired optimization
- **Adaptive GA** - Self-adjusting mutation/crossover rates
- **Particle Swarm Optimization (PSO)** - Swarm intelligence
- **Adaptive PSO** - Self-adjusting parameters
- **Differential Evolution (DE)** - Population-based optimization
- **Self-Adaptive DE (jDE)** - Self-adapting F and CR
- **Bayesian Optimization (BO)** - Gaussian Process surrogate
- **Tree Parzen Estimator (TPE)** - As used in Optuna/Hyperopt
- **Simulated Annealing (SA)** - Physics-inspired optimization
- **Adaptive SA** - Automatic temperature scheduling
- **Quantum-Inspired Annealing** - Quantum tunneling effects

### Specialized Agents
- **InhibitorAgent** - Enzyme inhibition prediction (e.g., HDAC)
- **SolubilityAgent** - Aqueous solubility (LogS) prediction
- **ToxicityAgent** - General toxicity prediction
- **hERGAgent** - Cardiac toxicity (hERG liability)
- **AgentOrchestrator** - Coordinate multiple agents
- **DrugDiscoveryPipeline** - End-to-end screening

## Installation

### Quick Setup (Windows)

```powershell
# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate

# Install PyTorch (CPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install PyTorch Geometric
pip install torch-geometric torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.0.0+cpu.html

# Install remaining dependencies
pip install -r requirements.txt
```

### Quick Setup (Linux/Mac)

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install PyTorch (CPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install PyTorch Geometric
pip install torch-geometric torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.0.0+cpu.html

# Install remaining dependencies
pip install -r requirements.txt
```

## Project Structure

```
GenAI-POD-Potential-POC/
├── models/                      # GNN Model Architectures
│   ├── __init__.py
│   └── gnn_models.py           # All model classes + ModelRegistry
│
├── optimizers/                  # Meta-Heuristic Algorithms
│   ├── __init__.py
│   ├── base_optimizer.py       # BaseOptimizer + SearchSpace
│   ├── genetic_algorithm.py    # GA + Adaptive GA
│   ├── particle_swarm.py       # PSO + Adaptive PSO
│   ├── differential_evolution.py # DE + jDE
│   ├── bayesian_optimization.py  # BO + TPE
│   ├── simulated_annealing.py  # SA + Adaptive SA + QA
│   └── optimizer_factory.py    # OptimizerFactory
│
├── agents/                      # Multi-Agent System
│   ├── __init__.py
│   ├── base_agent.py           # BaseAgent + AgentConfig
│   ├── classification_agent.py # ClassificationAgent
│   ├── regression_agent.py     # RegressionAgent
│   ├── specialized_agents.py   # Inhibitor, Solubility, hERG
│   └── agent_orchestrator.py   # Orchestrator + Pipeline
│
├── utils/                       # Utility Functions
│   ├── __init__.py
│   ├── molecular_utils.py      # SMILES to Graph conversion
│   ├── data_utils.py           # Data loading + splitting
│   ├── metrics.py              # Evaluation metrics
│   └── visualization.py        # Plotting utilities
│
├── Maddie-Data/                 # Dataset files
│   ├── HDAC_processed.csv
│   ├── herg_processed.csv
│   └── solubility_processed.csv
│
├── saved_models/                # Trained model weights
├── outputs/                     # Evaluation visualizations
│
├── app.py                       # Streamlit web interface
├── requirements.txt             # Python dependencies
├── setup.py                     # Package setup
└── Multi_Agent_Drug_Discovery (1).ipynb  # Main notebook
```

## Usage

### 1. Using Models (Plug-and-Play)

```python
from models import ModelRegistry, create_model

# List available models
print(ModelRegistry.list_models())
# ['GCN', 'GAT', 'GIN', 'GraphSAGE', 'MPNN', 'AttentiveFP', 'AdvancedGNN']

# Create a model
model = create_model('GAT', num_node_features=6, hidden_channels=64, num_layers=3)

# Or use the registry
model = ModelRegistry.create('GIN', num_node_features=6, dropout=0.3)
```

### 2. Using Optimizers

```python
from optimizers import OptimizerFactory, SearchSpace, run_optimization

# Define search space
space = SearchSpace()
space.add_log_float('learning_rate', 1e-5, 1e-2)
space.add_int('hidden_channels', 32, 256)
space.add_categorical('model_type', ['GCN', 'GAT', 'GIN'])

# Define objective function
def objective(config):
    model = create_model(config['model_type'], hidden_channels=config['hidden_channels'])
    # Train and evaluate
    return auc_score

# Run optimization
best_config, best_score, history = run_optimization(
    objective, space, optimizer='bo', n_iterations=50
)
```

### 3. Using Agents

```python
from agents import InhibitorAgent, SolubilityAgent, hERGAgent, AgentOrchestrator

# Create agents
inhibitor_agent = InhibitorAgent(target_name="HDAC")
solubility_agent = SolubilityAgent()
herg_agent = hERGAgent()

# Create orchestrator
orchestrator = AgentOrchestrator()
orchestrator.add_agent(inhibitor_agent, weight=2.0)
orchestrator.add_agent(solubility_agent, weight=1.0)
orchestrator.add_agent(herg_agent, weight=1.5)

# Predict
results = orchestrator.predict_single("CCO")
print(results)

# Rank candidates
candidates = orchestrator.rank_candidates(smiles_list)
```

### 4. Running the Web Interface

```bash
streamlit run app.py
```

### 5. Running the Notebook

Open `Multi_Agent_Drug_Discovery (1).ipynb` in Jupyter Lab or VS Code.

## Data Format

Input CSV files should have:
- `SMILES` column: Molecular structures in SMILES format
- `target` column: Activity values (0/1 for classification, continuous for regression)

## Citation

If you use this code, please cite:

```bibtex
@software{drug_discovery_multiagent,
  title={Multi-Agent Drug Discovery Platform},
  year={2024},
  url={https://github.com/your-org/drug-discovery-multiagent}
}
```

## License

MIT License
