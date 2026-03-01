"""
🧬 Multi-Agent Drug Discovery Platform
=====================================
A Streamlit-based interface for the multi-agent drug discovery system.

This application provides:
- SMILES molecule input and visualization
- Real-time predictions from multiple AI agents
- Chemical structure rendering
- Property calculations and drug-likeness assessment
- Interactive result exploration

Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool, global_add_pool
from pathlib import Path
import warnings
import io
import base64
from typing import Dict, List, Tuple, Optional, Any

warnings.filterwarnings('ignore')

# Import RDKit
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, Draw
    from rdkit.Chem.Draw import MolToImage
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    st.error("RDKit is required for this application. Please install it with: pip install rdkit")

# Page configuration
st.set_page_config(
    page_title="Multi-Agent Drug Discovery",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .agent-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .status-active {
        background-color: #28a745;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 5px;
        font-size: 0.8rem;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #007bff;
    }
    .prediction-good {
        background-color: #d4edda;
        border-color: #28a745;
    }
    .prediction-warning {
        background-color: #fff3cd;
        border-color: #ffc107;
    }
    .prediction-bad {
        background-color: #f8d7da;
        border-color: #dc3545;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# MODEL DEFINITIONS (Same as notebook)
# ==============================================================================

class MolecularGCN(torch.nn.Module):
    """Graph Convolutional Network for molecular property prediction."""
    def __init__(self, num_node_features: int, hidden_channels: int = 64, 
                 num_classes: int = 1, dropout: float = 0.3):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv3(x, edge_index)
        x = global_mean_pool(x, batch)
        return self.lin(x)


class MolecularGAT(torch.nn.Module):
    """Graph Attention Network for molecular property prediction."""
    def __init__(self, num_node_features: int, hidden_channels: int = 64,
                 num_classes: int = 1, heads: int = 4, dropout: float = 0.3):
        super().__init__()
        self.conv1 = GATConv(num_node_features, hidden_channels, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout)
        self.conv3 = GATConv(hidden_channels * heads, hidden_channels, heads=1, concat=False, dropout=dropout)
        self.lin = torch.nn.Linear(hidden_channels, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index, batch):
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv3(x, edge_index)
        x = global_mean_pool(x, batch)
        return self.lin(x)


class MolecularGNN_Advanced(torch.nn.Module):
    """Advanced GNN with residual connections and multiple pooling."""
    def __init__(self, num_node_features: int, hidden_channels: int = 64,
                 num_classes: int = 1, dropout: float = 0.3):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.bn3 = torch.nn.BatchNorm1d(hidden_channels)
        self.lin1 = torch.nn.Linear(hidden_channels * 3, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index, batch):
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.bn3(self.conv3(x, edge_index)))
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x_sum = global_add_pool(x, batch)
        x = torch.cat([x_mean, x_max, x_sum], dim=1)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lin2(x)


def create_model(model_type: str, num_node_features: int, task_type: str = 'classification') -> torch.nn.Module:
    """Factory function to create models."""
    num_classes = 1
    if model_type == 'GCN':
        return MolecularGCN(num_node_features, num_classes=num_classes)
    elif model_type == 'GAT':
        return MolecularGAT(num_node_features, num_classes=num_classes)
    elif model_type == 'AdvancedGNN':
        return MolecularGNN_Advanced(num_node_features, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# ==============================================================================
# MOLECULAR PROCESSING
# ==============================================================================

def smiles_to_graph(smiles: str) -> Optional[Data]:
    """Convert SMILES to PyTorch Geometric graph."""
    if not RDKIT_AVAILABLE:
        return None
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # Atom features
    atom_features = []
    for atom in mol.GetAtoms():
        features = [
            atom.GetAtomicNum(),
            atom.GetDegree(),
            atom.GetFormalCharge(),
            int(atom.GetHybridization()),
            int(atom.GetIsAromatic()),
            atom.GetTotalNumHs(),
            atom.GetNumRadicalElectrons(),
            int(atom.IsInRing()),
        ]
        atom_features.append(features)
    
    x = torch.tensor(atom_features, dtype=torch.float)
    
    # Edge indices
    edge_indices = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_indices.append([i, j])
        edge_indices.append([j, i])
    
    if len(edge_indices) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    
    return Data(x=x, edge_index=edge_index)


def calculate_molecular_properties(smiles: str) -> Optional[Dict[str, float]]:
    """Calculate molecular properties from SMILES."""
    if not RDKIT_AVAILABLE:
        return None
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    properties = {
        'Molecular Weight': Descriptors.MolWt(mol),
        'LogP': Descriptors.MolLogP(mol),
        'TPSA': Descriptors.TPSA(mol),
        'H-Bond Donors': Descriptors.NumHDonors(mol),
        'H-Bond Acceptors': Descriptors.NumHAcceptors(mol),
        'Rotatable Bonds': Descriptors.NumRotatableBonds(mol),
        'Heavy Atoms': Descriptors.HeavyAtomCount(mol),
        'Aromatic Rings': Descriptors.NumAromaticRings(mol),
        'Fraction sp3': Descriptors.FractionCSP3(mol),
        'QED': Descriptors.qed(mol),  # Drug-likeness score
    }
    
    # Lipinski Rule of 5
    lipinski_violations = 0
    if properties['Molecular Weight'] > 500:
        lipinski_violations += 1
    if properties['LogP'] > 5:
        lipinski_violations += 1
    if properties['H-Bond Donors'] > 5:
        lipinski_violations += 1
    if properties['H-Bond Acceptors'] > 10:
        lipinski_violations += 1
    
    properties['Lipinski Violations'] = lipinski_violations
    properties['Drug-like'] = lipinski_violations <= 1
    
    return properties


def mol_to_base64_img(smiles: str, size: Tuple[int, int] = (300, 300)) -> Optional[str]:
    """Convert SMILES to base64 encoded image."""
    if not RDKIT_AVAILABLE:
        return None
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    img = Draw.MolToImage(mol, size=size)
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str


# ==============================================================================
# PREDICTION AGENTS
# ==============================================================================

class PredictionAgent:
    """Base agent for making predictions."""
    
    def __init__(self, name: str, task_type: str, model_dir: Path):
        self.name = name
        self.task_type = task_type
        self.model_dir = model_dir
        self.models = {}
        self.loaded = False
        
    def load_models(self) -> bool:
        """Load saved models from disk."""
        if not self.model_dir.exists():
            return False
        
        model_files = list(self.model_dir.glob("*.pt"))
        if len(model_files) == 0:
            return False
        
        for model_file in model_files:
            model_name = model_file.stem
            try:
                checkpoint = torch.load(model_file, map_location='cpu')
                model_type = checkpoint.get('model_type', 'GCN')
                num_features = checkpoint.get('num_features', 8)
                
                model = create_model(model_type, num_features, self.task_type)
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()
                
                self.models[model_name] = {
                    'model': model,
                    'type': model_type
                }
            except Exception as e:
                st.warning(f"Could not load model {model_name}: {e}")
        
        self.loaded = len(self.models) > 0
        return self.loaded
    
    def predict(self, smiles: str) -> Dict[str, Any]:
        """Make prediction for a SMILES string."""
        if not self.loaded:
            return {'error': 'Models not loaded'}
        
        graph = smiles_to_graph(smiles)
        if graph is None:
            return {'error': 'Invalid SMILES'}
        
        graph.batch = torch.zeros(graph.x.size(0), dtype=torch.long)
        
        results = {}
        with torch.no_grad():
            for model_name, model_info in self.models.items():
                model = model_info['model']
                output = model(graph.x, graph.edge_index, graph.batch)
                
                if self.task_type == 'classification':
                    prob = torch.sigmoid(output).item()
                    pred = 1 if prob > 0.5 else 0
                    results[model_name] = {
                        'prediction': pred,
                        'probability': prob,
                        'confidence': abs(prob - 0.5) * 2
                    }
                else:
                    value = output.item()
                    results[model_name] = {
                        'prediction': value,
                        'unit': 'LogS' if 'solubility' in self.name.lower() else ''
                    }
        
        # Ensemble prediction (average)
        if self.task_type == 'classification':
            avg_prob = np.mean([r['probability'] for r in results.values()])
            results['Ensemble'] = {
                'prediction': 1 if avg_prob > 0.5 else 0,
                'probability': avg_prob,
                'confidence': abs(avg_prob - 0.5) * 2
            }
        else:
            avg_value = np.mean([r['prediction'] for r in results.values()])
            results['Ensemble'] = {
                'prediction': avg_value,
                'unit': results[list(results.keys())[0]].get('unit', '')
            }
        
        return results


class InhibitorAgent(PredictionAgent):
    """Agent for HDAC inhibition prediction."""
    
    def interpret_result(self, result: Dict) -> str:
        """Interpret the prediction result."""
        ensemble = result.get('Ensemble', {})
        prob = ensemble.get('probability', 0.5)
        
        if prob > 0.7:
            return "🟢 HIGH PROBABILITY of HDAC inhibition activity. This compound shows strong potential as an HDAC inhibitor candidate."
        elif prob > 0.5:
            return "🟡 MODERATE PROBABILITY of HDAC inhibition. May be worth further investigation with structural modifications."
        else:
            return "🔴 LOW PROBABILITY of HDAC inhibition. This compound is unlikely to be an effective HDAC inhibitor."


class SolubilityAgent(PredictionAgent):
    """Agent for solubility prediction."""
    
    def interpret_result(self, result: Dict) -> str:
        """Interpret the prediction result."""
        ensemble = result.get('Ensemble', {})
        logs = ensemble.get('prediction', 0)
        
        if logs > -2:
            return f"🟢 HIGH SOLUBILITY (LogS = {logs:.2f}). Excellent aqueous solubility - favorable for oral bioavailability."
        elif logs > -4:
            return f"🟡 MODERATE SOLUBILITY (LogS = {logs:.2f}). Acceptable solubility, but formulation strategies may be needed."
        else:
            return f"🔴 LOW SOLUBILITY (LogS = {logs:.2f}). Poor aqueous solubility - may require solubility enhancement techniques."


class ToxicityAgent(PredictionAgent):
    """Agent for hERG toxicity prediction."""
    
    def interpret_result(self, result: Dict) -> str:
        """Interpret the prediction result."""
        ensemble = result.get('Ensemble', {})
        prob = ensemble.get('probability', 0.5)
        
        if prob > 0.7:
            return "🔴 HIGH RISK of hERG channel inhibition. CAUTION: This compound may cause cardiac QT prolongation and arrhythmias."
        elif prob > 0.5:
            return "🟡 MODERATE RISK of hERG inhibition. Further experimental validation recommended before proceeding."
        else:
            return "🟢 LOW RISK of hERG inhibition. Compound appears to have favorable cardiac safety profile."


# ==============================================================================
# STREAMLIT APP
# ==============================================================================

def main():
    # Header
    st.markdown('<h1 class="main-header">🧬 Multi-Agent Drug Discovery Platform</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-powered prediction of drug properties using Graph Neural Networks</p>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("⚙️ Control Panel")
    
    # Initialize agents
    models_dir = Path("saved_models")
    
    @st.cache_resource
    def load_agents():
        # Directory names must match those created by the notebook
        inhibitor_agent = InhibitorAgent("HDAC Inhibitor", "classification", models_dir / "HDAC_Inhibitor_Agent")
        solubility_agent = SolubilityAgent("Solubility", "regression", models_dir / "Solubility_Agent")
        toxicity_agent = ToxicityAgent("hERG Toxicity", "classification", models_dir / "hERG_Toxicity_Agent")
        
        return inhibitor_agent, solubility_agent, toxicity_agent
    
    inhibitor_agent, solubility_agent, toxicity_agent = load_agents()
    
    # Check if models are available
    models_exist = models_dir.exists() and any(models_dir.rglob("*.pt"))
    
    # Agent Status
    st.sidebar.markdown("### 📊 Agent Status")
    
    agents_status = {
        "HDAC Inhibitor": inhibitor_agent.load_models() if models_exist else False,
        "Solubility": solubility_agent.load_models() if models_exist else False,
        "hERG Toxicity": toxicity_agent.load_models() if models_exist else False,
    }
    
    for agent_name, status in agents_status.items():
        status_emoji = "✅" if status else "⚠️"
        status_text = "Ready" if status else "Not trained"
        st.sidebar.markdown(f"{status_emoji} **{agent_name}**: {status_text}")
    
    if not any(agents_status.values()):
        st.sidebar.warning("⚠️ No trained models found. Run the notebook first to train models.")
    
    # Example molecules
    st.sidebar.markdown("### 🧪 Example Molecules")
    example_molecules = {
        "Aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
        "Caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "Ibuprofen": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
        "Paracetamol": "CC(=O)NC1=CC=C(C=C1)O",
        "Vorinostat (HDAC)": "ONC(=O)CCCCCCC(=O)Nc1ccccc1",
        "Metformin": "CN(C)C(=N)NC(=N)N",
    }
    
    selected_example = st.sidebar.selectbox("Select example:", [""] + list(example_molecules.keys()))
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 🔬 Molecule Input")
        
        # SMILES input
        if selected_example and selected_example in example_molecules:
            default_smiles = example_molecules[selected_example]
        else:
            default_smiles = ""
        
        smiles_input = st.text_input(
            "Enter SMILES string:",
            value=default_smiles,
            placeholder="e.g., CC(=O)OC1=CC=CC=C1C(=O)O"
        )
        
        predict_button = st.button("🚀 Run Predictions", type="primary", use_container_width=True)
    
    with col2:
        st.markdown("### 🖼️ Molecular Structure")
        
        if smiles_input and RDKIT_AVAILABLE:
            img_base64 = mol_to_base64_img(smiles_input, size=(300, 300))
            if img_base64:
                st.markdown(
                    f'<div style="text-align: center;"><img src="data:image/png;base64,{img_base64}" style="border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);"></div>',
                    unsafe_allow_html=True
                )
            else:
                st.warning("⚠️ Invalid SMILES string")
        elif smiles_input:
            st.info("Install RDKit to visualize molecular structures")
        else:
            st.info("Enter a SMILES string to visualize")
    
    # Properties section
    if smiles_input:
        st.markdown("---")
        st.markdown("### 📊 Molecular Properties")
        
        properties = calculate_molecular_properties(smiles_input)
        
        if properties:
            props_col1, props_col2, props_col3, props_col4 = st.columns(4)
            
            with props_col1:
                st.metric("Molecular Weight", f"{properties['Molecular Weight']:.1f}")
                st.metric("LogP", f"{properties['LogP']:.2f}")
                st.metric("QED Score", f"{properties['QED']:.3f}")
            
            with props_col2:
                st.metric("H-Bond Donors", int(properties['H-Bond Donors']))
                st.metric("H-Bond Acceptors", int(properties['H-Bond Acceptors']))
                st.metric("Rotatable Bonds", int(properties['Rotatable Bonds']))
            
            with props_col3:
                st.metric("TPSA", f"{properties['TPSA']:.1f}")
                st.metric("Heavy Atoms", int(properties['Heavy Atoms']))
                st.metric("Aromatic Rings", int(properties['Aromatic Rings']))
            
            with props_col4:
                st.metric("Lipinski Violations", int(properties['Lipinski Violations']))
                drug_like = "✅ Yes" if properties['Drug-like'] else "❌ No"
                st.metric("Drug-like", drug_like)
                st.metric("Fraction sp³", f"{properties['Fraction sp3']:.2f}")
        else:
            st.warning("Could not calculate properties. Check SMILES validity.")
    
    # Predictions section
    if predict_button and smiles_input:
        st.markdown("---")
        st.markdown("### 🎯 AI Predictions")
        
        progress = st.progress(0)
        status_text = st.empty()
        
        # Run predictions
        results_container = st.container()
        
        with results_container:
            pred_col1, pred_col2, pred_col3 = st.columns(3)
            
            # HDAC Inhibitor
            status_text.text("🔄 Running HDAC Inhibitor prediction...")
            progress.progress(33)
            
            with pred_col1:
                st.markdown("#### 🧬 HDAC Inhibitor")
                if agents_status["HDAC Inhibitor"]:
                    hdac_result = inhibitor_agent.predict(smiles_input)
                    if 'error' not in hdac_result:
                        ensemble = hdac_result.get('Ensemble', {})
                        prob = ensemble.get('probability', 0)
                        pred = "Active" if ensemble.get('prediction', 0) == 1 else "Inactive"
                        
                        color = "#28a745" if prob > 0.7 else "#ffc107" if prob > 0.5 else "#dc3545"
                        st.markdown(f"""
                        <div style="background: {color}20; padding: 1rem; border-radius: 10px; border-left: 4px solid {color};">
                            <h3 style="color: {color}; margin: 0;">{pred}</h3>
                            <p style="margin: 0.5rem 0 0 0;">Probability: {prob:.1%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.info(inhibitor_agent.interpret_result(hdac_result))
                    else:
                        st.error(hdac_result['error'])
                else:
                    st.warning("Model not trained")
            
            # Solubility
            status_text.text("🔄 Running Solubility prediction...")
            progress.progress(66)
            
            with pred_col2:
                st.markdown("#### 💧 Solubility")
                if agents_status["Solubility"]:
                    sol_result = solubility_agent.predict(smiles_input)
                    if 'error' not in sol_result:
                        ensemble = sol_result.get('Ensemble', {})
                        logs = ensemble.get('prediction', 0)
                        
                        color = "#28a745" if logs > -2 else "#ffc107" if logs > -4 else "#dc3545"
                        solubility_class = "High" if logs > -2 else "Moderate" if logs > -4 else "Low"
                        
                        st.markdown(f"""
                        <div style="background: {color}20; padding: 1rem; border-radius: 10px; border-left: 4px solid {color};">
                            <h3 style="color: {color}; margin: 0;">{solubility_class}</h3>
                            <p style="margin: 0.5rem 0 0 0;">LogS: {logs:.2f}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.info(solubility_agent.interpret_result(sol_result))
                    else:
                        st.error(sol_result['error'])
                else:
                    st.warning("Model not trained")
            
            # hERG Toxicity
            status_text.text("🔄 Running hERG Toxicity prediction...")
            progress.progress(100)
            
            with pred_col3:
                st.markdown("#### ⚡ hERG Toxicity")
                if agents_status["hERG Toxicity"]:
                    herg_result = toxicity_agent.predict(smiles_input)
                    if 'error' not in herg_result:
                        ensemble = herg_result.get('Ensemble', {})
                        prob = ensemble.get('probability', 0)
                        pred = "Toxic" if ensemble.get('prediction', 0) == 1 else "Safe"
                        
                        # For toxicity, high prob = bad
                        color = "#dc3545" if prob > 0.7 else "#ffc107" if prob > 0.5 else "#28a745"
                        
                        st.markdown(f"""
                        <div style="background: {color}20; padding: 1rem; border-radius: 10px; border-left: 4px solid {color};">
                            <h3 style="color: {color}; margin: 0;">{pred}</h3>
                            <p style="margin: 0.5rem 0 0 0;">Risk Probability: {prob:.1%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.info(toxicity_agent.interpret_result(herg_result))
                    else:
                        st.error(herg_result['error'])
                else:
                    st.warning("Model not trained")
        
        status_text.text("✅ All predictions complete!")
        
        # Overall assessment
        st.markdown("---")
        st.markdown("### 📋 Overall Drug Candidate Assessment")
        
        if properties:
            assessment_points = []
            score = 0
            max_score = 5
            
            # Drug-likeness
            if properties['Drug-like']:
                assessment_points.append("✅ Passes Lipinski Rule of 5")
                score += 1
            else:
                assessment_points.append(f"⚠️ {int(properties['Lipinski Violations'])} Lipinski violations")
            
            # QED
            if properties['QED'] > 0.5:
                assessment_points.append(f"✅ Good drug-likeness score (QED={properties['QED']:.2f})")
                score += 1
            else:
                assessment_points.append(f"⚠️ Low drug-likeness score (QED={properties['QED']:.2f})")
            
            # TPSA
            if 20 < properties['TPSA'] < 140:
                assessment_points.append(f"✅ Good TPSA for oral absorption ({properties['TPSA']:.1f})")
                score += 1
            else:
                assessment_points.append(f"⚠️ TPSA outside optimal range ({properties['TPSA']:.1f})")
            
            # Predicted properties
            if agents_status["Solubility"] and 'error' not in sol_result:
                logs = sol_result.get('Ensemble', {}).get('prediction', -4)
                if logs > -4:
                    assessment_points.append(f"✅ Acceptable predicted solubility")
                    score += 1
                else:
                    assessment_points.append(f"⚠️ Poor predicted solubility")
            
            if agents_status["hERG Toxicity"] and 'error' not in herg_result:
                prob = herg_result.get('Ensemble', {}).get('probability', 0.5)
                if prob < 0.5:
                    assessment_points.append(f"✅ Low predicted cardiac toxicity risk")
                    score += 1
                else:
                    assessment_points.append(f"⚠️ Elevated cardiac toxicity risk")
            
            # Display assessment
            overall_color = "#28a745" if score >= 4 else "#ffc107" if score >= 2 else "#dc3545"
            overall_status = "PROMISING" if score >= 4 else "MODERATE" if score >= 2 else "CONCERNING"
            
            st.markdown(f"""
            <div style="background: {overall_color}20; padding: 1.5rem; border-radius: 10px; border: 2px solid {overall_color};">
                <h2 style="color: {overall_color}; margin: 0; text-align: center;">{overall_status} DRUG CANDIDATE</h2>
                <p style="text-align: center; margin: 0.5rem 0 0 0; font-size: 1.2rem;">Score: {score}/{max_score}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("**Assessment Points:**")
            for point in assessment_points:
                st.markdown(f"- {point}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        <p>🧬 Multi-Agent Drug Discovery Platform | Powered by Graph Neural Networks</p>
        <p>⚠️ Predictions are for research purposes only and should be validated experimentally.</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
