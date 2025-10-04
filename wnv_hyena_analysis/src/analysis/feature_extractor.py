"""
Feature extraction for WNV genome sequences
Traditional bioinformatics features and HyenaDNA-inspired deep features
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
import itertools
from collections import Counter
import logging

try:
    import torch
    import torch.nn as nn
    from transformers import AutoTokenizer, AutoModel
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

from src.utils.logger import setup_logger, ProgressLogger


class TraditionalFeatureExtractor:
    """Extract traditional bioinformatics features from DNA sequences"""
    
    def __init__(self, config: dict, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or setup_logger("feature_extractor")
        
        # Feature configuration
        self.kmer_sizes = config.get('features', {}).get('traditional', {}).get('kmer_size', [3, 4])
        self.include_composition = config.get('features', {}).get('traditional', {}).get('nucleotide_composition', True)
        self.include_physicochemical = config.get('features', {}).get('traditional', {}).get('physicochemical', True)
        self.include_codon = config.get('features', {}).get('traditional', {}).get('codon_usage', True)
    
    def extract_features(self, sequences: List[str]) -> np.ndarray:
        """
        Extract traditional bioinformatics features from sequences
        
        Args:
            sequences: List of DNA sequences
            
        Returns:
            Feature matrix of shape (n_sequences, n_features)
        """
        self.logger.info(f"Extracting traditional features from {len(sequences)} sequences")
        
        features_list = []
        progress = ProgressLogger(self.logger, len(sequences), "Extracting traditional features")
        
        for sequence in sequences:
            seq_features = self._extract_sequence_features(sequence)
            features_list.append(seq_features)
            progress.update()
        
        progress.finish()
        
        features = np.array(features_list)
        self.logger.info(f"Extracted traditional features with shape: {features.shape}")
        
        return features
    
    def _extract_sequence_features(self, sequence: str) -> np.ndarray:
        """Extract all traditional features for a single sequence"""
        features = []
        
        # Nucleotide composition
        if self.include_composition:
            features.extend(self._nucleotide_composition(sequence))
        
        # K-mer frequencies
        for k in self.kmer_sizes:
            features.extend(self._kmer_frequencies(sequence, k))
        
        # Physicochemical properties
        if self.include_physicochemical:
            features.extend(self._physicochemical_properties(sequence))
        
        # Codon usage - always include to maintain consistent dimensions
        if self.include_codon:
            features.extend(self._codon_usage(sequence))
        
        return np.array(features)
    
    def _nucleotide_composition(self, sequence: str) -> List[float]:
        """Calculate nucleotide composition frequencies"""
        seq = sequence.upper()
        length = len(seq)
        
        if length == 0:
            return [0.0] * 5
        
        composition = [
            seq.count('A') / length,
            seq.count('T') / length,
            seq.count('G') / length,
            seq.count('C') / length,
            seq.count('N') / length
        ]
        
        return composition
    
    def _kmer_frequencies(self, sequence: str, k: int) -> List[float]:
        """Calculate k-mer frequencies"""
        seq = sequence.upper().replace('N', '')  # Remove ambiguous bases
        
        # Generate all possible k-mers
        bases = ['A', 'T', 'G', 'C']
        kmers = [''.join(p) for p in itertools.product(bases, repeat=k)]
        
        # Count k-mers
        kmer_counts = Counter()
        for i in range(len(seq) - k + 1):
            kmer = seq[i:i+k]
            if all(base in bases for base in kmer):
                kmer_counts[kmer] += 1
        
        # Convert to frequencies
        total_kmers = sum(kmer_counts.values())
        if total_kmers == 0:
            return [0.0] * len(kmers)
        
        frequencies = [kmer_counts[kmer] / total_kmers for kmer in kmers]
        return frequencies
    
    def _physicochemical_properties(self, sequence: str) -> List[float]:
        """Calculate physicochemical properties"""
        seq = sequence.upper()
        length = len(seq)
        
        if length == 0:
            return [0.0] * 6
        
        # Base counts
        a_count = seq.count('A')
        t_count = seq.count('T')
        g_count = seq.count('G')
        c_count = seq.count('C')
        
        # GC/AT content and skew
        gc_content = (g_count + c_count) / length
        at_content = (a_count + t_count) / length
        
        gc_skew = (g_count - c_count) / (g_count + c_count) if (g_count + c_count) > 0 else 0
        at_skew = (a_count - t_count) / (a_count + t_count) if (a_count + t_count) > 0 else 0
        
        # Purine/Pyrimidine content
        purine_content = (a_count + g_count) / length
        pyrimidine_content = (t_count + c_count) / length
        
        return [gc_content, at_content, gc_skew, at_skew, purine_content, pyrimidine_content]
    
    def _codon_usage(self, sequence: str) -> List[float]:
        """Calculate codon usage frequencies"""
        seq = sequence.upper()
        
        # All possible codons
        bases = ['A', 'T', 'G', 'C']
        all_codons = [''.join(p) for p in itertools.product(bases, repeat=3)]
        
        # Extract codons (handle sequences not multiple of 3)
        codons = []
        for i in range(0, len(seq) - 2, 3):
            codon = seq[i:i+3]
            if len(codon) == 3 and all(base in 'ATGC' for base in codon):
                codons.append(codon)
        
        if not codons:
            return [0.0] * 64  # Return zeros if no valid codons
        
        # Count codon frequencies
        codon_counts = Counter(codons)
        total_codons = len(codons)
        
        frequencies = [codon_counts[codon] / total_codons for codon in all_codons]
        return frequencies
    
    def get_feature_names(self) -> List[str]:
        """Get feature names for interpretation"""
        names = []
        
        if self.include_composition:
            names.extend(['A_freq', 'T_freq', 'G_freq', 'C_freq', 'N_freq'])
        
        for k in self.kmer_sizes:
            bases = ['A', 'T', 'G', 'C']
            kmers = [''.join(p) for p in itertools.product(bases, repeat=k)]
            names.extend([f'{kmer}_freq' for kmer in kmers])
        
        if self.include_physicochemical:
            names.extend(['GC_content', 'AT_content', 'GC_skew', 'AT_skew', 'Purine_content', 'Pyrimidine_content'])
        
        if self.include_codon:
            bases = ['A', 'T', 'G', 'C']
            codons = [''.join(p) for p in itertools.product(bases, repeat=3)]
            names.extend([f'{codon}_codon' for codon in codons])
        
        return names


class HyenaDNAFeatureExtractor:
    """Extract deep learning features using HyenaDNA or HyenaDNA-inspired methods"""
    
    def __init__(self, config: dict, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or setup_logger("hyenadna_extractor")
        
        # Configuration
        self.embedding_dim = config.get('features', {}).get('deep_learning', {}).get('embedding_dim', 256)
        self.max_length = config.get('features', {}).get('deep_learning', {}).get('max_seq_length', 15000)
        self.batch_size = config.get('features', {}).get('deep_learning', {}).get('batch_size', 16)
        self.use_gpu = config.get('features', {}).get('deep_learning', {}).get('use_gpu', True) and torch.cuda.is_available()
        
        self.device = torch.device('cuda' if self.use_gpu else 'cpu')
        self.logger.info(f"Using device: {self.device}")
        
        # Try to load real HyenaDNA model, fallback to custom implementation
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """Load HyenaDNA model or create custom implementation"""
        if not PYTORCH_AVAILABLE:
            self.logger.warning("PyTorch not available, skipping deep features")
            return
        
        try:
            # Try to load real HyenaDNA model
            model_name = self.config.get('features', {}).get('deep_learning', {}).get('model_name', 'hyenadna-medium-450k-seqlen')
            self.logger.info(f"Attempting to load HyenaDNA model: {model_name}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(f"LongSafari/{model_name}", trust_remote_code=True)
            self.model = AutoModel.from_pretrained(f"LongSafari/{model_name}", trust_remote_code=True)
            
            self.model.to(self.device)
            self.model.eval()
            
            self.logger.info("Successfully loaded HyenaDNA model")
            
        except Exception as e:
            self.logger.warning(f"Could not load HyenaDNA model: {e}")
            self.logger.info("Using custom HyenaDNA-inspired implementation")
            self._create_custom_model()
    
    def _create_custom_model(self):
        """Create custom HyenaDNA-inspired model"""
        
        class CustomDNATokenizer:
            def __init__(self):
                self.char_to_idx = {'A': 0, 'T': 1, 'G': 2, 'C': 3, 'N': 4}
            
            def __call__(self, sequence):
                sequence = sequence.upper()
                input_ids = [self.char_to_idx.get(char, 4) for char in sequence]
                return {'input_ids': input_ids}
        
        class CustomHyenaDNAModel(nn.Module):
            def __init__(self, vocab_size=5, embed_dim=256, max_length=15000):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, embed_dim)
                
                # Multi-scale CNN layers (simulating HyenaDNA's attention patterns)
                self.conv1 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1)
                self.conv2 = nn.Conv1d(embed_dim, embed_dim, kernel_size=7, padding=3)
                self.conv3 = nn.Conv1d(embed_dim, embed_dim, kernel_size=15, padding=7)
                
                self.pool = nn.AdaptiveAvgPool1d(1)
                self.dropout = nn.Dropout(0.1)
                
            def forward(self, input_ids):
                if isinstance(input_ids, list):
                    input_ids = torch.tensor(input_ids, device=self.embedding.weight.device)
                
                if input_ids.dim() == 1:
                    input_ids = input_ids.unsqueeze(0)
                
                # Embeddings
                x = self.embedding(input_ids)  # (batch, seq_len, embed_dim)
                x = x.transpose(1, 2)  # (batch, embed_dim, seq_len)
                
                # Multi-scale convolutions
                x1 = torch.relu(self.conv1(x))
                x2 = torch.relu(self.conv2(x))
                x3 = torch.relu(self.conv3(x))
                
                # Combine features
                x = (x1 + x2 + x3) / 3
                x = self.dropout(x)
                
                # Global average pooling
                x = self.pool(x)  # (batch, embed_dim, 1)
                x = x.squeeze(-1)  # (batch, embed_dim)
                
                return x
        
        self.tokenizer = CustomDNATokenizer()
        self.model = CustomHyenaDNAModel(embed_dim=self.embedding_dim, max_length=self.max_length)
        self.model.to(self.device)
        self.model.eval()
        
        self.logger.info("Created custom HyenaDNA-inspired model")
    
    def extract_features(self, sequences: List[str]) -> np.ndarray:
        """
        Extract deep learning features using HyenaDNA
        
        Args:
            sequences: List of DNA sequences
            
        Returns:
            Feature matrix of shape (n_sequences, embedding_dim)
        """
        if self.model is None:
            self.logger.warning("No model available, returning zeros")
            return np.zeros((len(sequences), self.embedding_dim))
        
        self.logger.info(f"Extracting HyenaDNA features from {len(sequences)} sequences")
        
        all_features = []
        progress = ProgressLogger(self.logger, len(sequences), "Extracting HyenaDNA features")
        
        # Process in batches
        for i in range(0, len(sequences), self.batch_size):
            batch_sequences = sequences[i:i + self.batch_size]
            batch_features = self._extract_batch_features(batch_sequences)
            all_features.append(batch_features)
            
            progress.update(len(batch_sequences))
        
        progress.finish()
        
        features = np.vstack(all_features)
        self.logger.info(f"Extracted HyenaDNA features with shape: {features.shape}")
        
        return features
    
    def _extract_batch_features(self, sequences: List[str]) -> np.ndarray:
        """Extract features for a batch of sequences"""
        batch_inputs = []
        
        for sequence in sequences:
            # Truncate or pad sequence
            if len(sequence) > self.max_length:
                sequence = sequence[:self.max_length]
            
            # Tokenize
            tokens = self.tokenizer(sequence)
            batch_inputs.append(tokens['input_ids'])
        
        # Pad sequences in batch to same length
        max_len = max(len(seq) for seq in batch_inputs)
        padded_batch = []
        
        for seq in batch_inputs:
            if len(seq) < max_len:
                seq.extend([4] * (max_len - len(seq)))  # Pad with N tokens
            padded_batch.append(seq)
        
        # Convert to tensor
        batch_tensor = torch.tensor(padded_batch, device=self.device, dtype=torch.long)
        
        # Extract features
        with torch.no_grad():
            features = self.model(batch_tensor)
        
        return features.cpu().numpy()


class FeatureExtractor:
    """Combined feature extractor for traditional and deep learning features"""
    
    def __init__(self, config: dict, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or setup_logger("combined_extractor")
        
        # Initialize extractors
        self.traditional_extractor = TraditionalFeatureExtractor(config, logger)
        
        if PYTORCH_AVAILABLE and config.get('features', {}).get('deep_learning', {}).get('enabled', True):
            self.hyenadna_extractor = HyenaDNAFeatureExtractor(config, logger)
        else:
            self.hyenadna_extractor = None
            self.logger.warning("Deep learning features disabled or PyTorch not available")
    
    def extract_all_features(self, sequences: List[str]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Extract both traditional and deep learning features
        
        Returns:
            Tuple of (traditional_features, deep_features, feature_names)
        """
        self.logger.info("Starting comprehensive feature extraction")
        
        # Extract traditional features
        traditional_features = self.traditional_extractor.extract_features(sequences)
        
        # Extract deep features
        if self.hyenadna_extractor is not None:
            deep_features = self.hyenadna_extractor.extract_features(sequences)
        else:
            deep_features = np.zeros((len(sequences), 256))  # Placeholder
        
        # Get feature names
        feature_names = self.traditional_extractor.get_feature_names()
        feature_names.extend([f'hyenadna_{i}' for i in range(deep_features.shape[1])])
        
        self.logger.info(f"Feature extraction complete:")
        self.logger.info(f"  Traditional features: {traditional_features.shape}")
        self.logger.info(f"  Deep features: {deep_features.shape}")
        self.logger.info(f"  Total features: {len(feature_names)}")
        
        return traditional_features, deep_features, feature_names