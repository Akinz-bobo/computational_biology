"""
Data loading and preprocessing for WNV genome analysis
"""

import pandas as pd
import numpy as np
from Bio import SeqIO
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import re
from collections import Counter
import logging

from src.utils.logger import setup_logger, ProgressLogger


class WNVDataLoader:
    """
    Efficient data loader for West Nile Virus genome sequences
    Handles FASTA parsing, metadata extraction, and quality control
    """
    
    def __init__(self, config: dict, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or setup_logger("data_loader")
        
        # Quality thresholds
        self.min_length = config.get('sequence', {}).get('min_length', 8000)
        self.max_length = config.get('sequence', {}).get('max_length', 15000)
        self.min_quality = config.get('sequence', {}).get('min_quality', 0.95)
        
        self.sequences = []
        self.metadata = []
    
    def load_fasta(self, fasta_path: str, sample_size: Optional[int] = None) -> Tuple[List[str], List[Dict]]:
        """
        Load and process FASTA file with comprehensive metadata extraction
        
        Args:
            fasta_path: Path to FASTA file
            sample_size: If provided, randomly sample this many sequences
            
        Returns:
            Tuple of (sequences, metadata_list)
        """
        fasta_path = Path(fasta_path)
        if not fasta_path.exists():
            raise FileNotFoundError(f"FASTA file not found: {fasta_path}")
        
        self.logger.info(f"Loading FASTA file: {fasta_path}")
        
        # Count total sequences for progress tracking
        with open(fasta_path, 'r') as f:
            total_seqs = sum(1 for line in f if line.startswith('>'))
        
        self.logger.info(f"Found {total_seqs:,} sequences in FASTA file")
        
        sequences = []
        metadata = []
        
        progress = ProgressLogger(self.logger, total_seqs, "Loading sequences")
        
        for i, record in enumerate(SeqIO.parse(fasta_path, "fasta")):
            # Basic sequence info
            sequence = str(record.seq).upper()
            header = record.description
            
            # Quality control
            if not self._passes_quality_control(sequence):
                continue
            
            # Extract comprehensive metadata
            seq_metadata = self._extract_metadata(record.id, header, sequence)
            seq_metadata['sequence_index'] = len(sequences)  # Track position in final dataset
            
            sequences.append(sequence)
            metadata.append(seq_metadata)
            
            progress.update()
            
            # Stop if we've reached sample size
            if sample_size and len(sequences) >= sample_size:
                break
        
        progress.finish()
        
        self.logger.info(f"Loaded {len(sequences):,} high-quality sequences")
        if sample_size and len(sequences) >= sample_size:
            self.logger.info(f"Sampled {sample_size:,} sequences as requested")
        
        self.sequences = sequences
        self.metadata = metadata
        
        return sequences, metadata
    
    def _passes_quality_control(self, sequence: str) -> bool:
        """Check if sequence passes quality control filters"""
        length = len(sequence)
        n_content = sequence.count('N') / length if length > 0 else 1.0
        
        # Length check
        if length < self.min_length or length > self.max_length:
            return False
        
        # Quality check (N content)
        if n_content > (1.0 - self.min_quality):
            return False
        
        return True
    
    def _extract_metadata(self, seq_id: str, header: str, sequence: str) -> Dict:
        """Extract comprehensive metadata from sequence header and content"""
        metadata = {
            'sequence_id': seq_id,
            'header': header,
            'length': len(sequence)
        }
        
        # Sequence composition
        metadata.update(self._calculate_composition(sequence))
        
        # Header parsing
        metadata.update(self._parse_header(header))
        
        return metadata
    
    def _calculate_composition(self, sequence: str) -> Dict:
        """Calculate sequence composition metrics"""
        length = len(sequence)
        if length == 0:
            return {}
        
        base_counts = {
            'A': sequence.count('A'),
            'T': sequence.count('T'),
            'C': sequence.count('C'),
            'G': sequence.count('G'),
            'N': sequence.count('N')
        }
        
        return {
            'gc_content': (base_counts['G'] + base_counts['C']) / length * 100,
            'at_content': (base_counts['A'] + base_counts['T']) / length * 100,
            'n_content': base_counts['N'] / length * 100,
            'gc_skew': (base_counts['G'] - base_counts['C']) / (base_counts['G'] + base_counts['C']) 
                      if (base_counts['G'] + base_counts['C']) > 0 else 0,
            'at_skew': (base_counts['A'] - base_counts['T']) / (base_counts['A'] + base_counts['T']) 
                      if (base_counts['A'] + base_counts['T']) > 0 else 0,
        }
    
    def _parse_header(self, header: str) -> Dict:
        """Parse metadata from FASTA header"""
        header_lower = header.lower()
        
        metadata = {}
        
        # Extract strain/isolate
        metadata['strain'] = self._extract_strain(header)
        
        # Extract year
        metadata['year'] = self._extract_year(header)
        
        # Extract geographic information
        metadata['country'] = self._extract_country(header_lower)
        metadata['continent'] = self._get_continent(metadata['country'])
        
        # Extract host information
        metadata['host'] = self._extract_host(header_lower)
        
        # Extract lineage if mentioned
        metadata['lineage'] = self._extract_lineage(header_lower)
        
        return metadata
    
    def _extract_strain(self, header: str) -> str:
        """Extract strain information from header"""
        strain_patterns = [
            r'strain\s+([^,;\n\|]+)',
            r'isolate\s+([^,;\n\|]+)',
            r'/([^/\s]+)/\d{4}',  # Pattern like /NY99/1999
            r'WNV\s*[-_]?\s*([^,;\n\|]+)'
        ]
        
        for pattern in strain_patterns:
            match = re.search(pattern, header, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return 'Unknown'
    
    def _extract_year(self, header: str) -> Optional[int]:
        """Extract year from header"""
        year_match = re.search(r'\b(19[3-9][0-9]|20[0-2][0-9])\b', header)
        return int(year_match.group(1)) if year_match else None
    
    def _extract_country(self, header_lower: str) -> str:
        """Extract country from header"""
        countries = {
            'usa': 'USA', 'united states': 'USA', 'america': 'USA', 'us': 'USA',
            'canada': 'Canada', 'mexico': 'Mexico',
            'italy': 'Italy', 'greece': 'Greece', 'spain': 'Spain', 'france': 'France',
            'germany': 'Germany', 'romania': 'Romania', 'hungary': 'Hungary',
            'czech republic': 'Czech Republic', 'austria': 'Austria',
            'russia': 'Russia', 'ukraine': 'Ukraine', 'poland': 'Poland',
            'egypt': 'Egypt', 'israel': 'Israel', 'morocco': 'Morocco',
            'turkey': 'Turkey', 'iran': 'Iran', 'india': 'India',
            'australia': 'Australia', 'china': 'China', 'japan': 'Japan'
        }
        
        for country_key, country_name in countries.items():
            if country_key in header_lower:
                return country_name
        
        return 'Unknown'
    
    def _get_continent(self, country: str) -> str:
        """Map country to continent"""
        continent_map = {
            'USA': 'North America', 'Canada': 'North America', 'Mexico': 'North America',
            'Italy': 'Europe', 'Greece': 'Europe', 'Spain': 'Europe', 'France': 'Europe',
            'Germany': 'Europe', 'Romania': 'Europe', 'Hungary': 'Europe',
            'Czech Republic': 'Europe', 'Austria': 'Europe', 'Russia': 'Europe',
            'Ukraine': 'Europe', 'Poland': 'Europe',
            'Egypt': 'Africa', 'Morocco': 'Africa',
            'Turkey': 'Asia', 'Iran': 'Asia', 'Israel': 'Asia',
            'India': 'Asia', 'China': 'Asia', 'Japan': 'Asia',
            'Australia': 'Oceania'
        }
        
        return continent_map.get(country, 'Unknown')
    
    def _extract_host(self, header_lower: str) -> str:
        """Extract host information from header"""
        host_patterns = [
            r'host[:\s]+([^,;\n\|]+)',
            r'from\s+([a-zA-Z]+\s+[a-zA-Z]+)',
            r'\b(human|mosquito|bird|crow|horse|culex|aedes)\b'
        ]
        
        for pattern in host_patterns:
            match = re.search(pattern, header_lower)
            if match:
                return match.group(1).strip().title()
        
        return 'Unknown'
    
    def _extract_lineage(self, header_lower: str) -> Optional[str]:
        """Extract lineage information from header"""
        lineage_patterns = [
            r'lineage\s+(\w+)',
            r'clade\s+(\w+)',
            r'genotype\s+(\w+)'
        ]
        
        for pattern in lineage_patterns:
            match = re.search(pattern, header_lower)
            if match:
                return match.group(1).upper()
        
        return None
    
    def create_dataframe(self) -> pd.DataFrame:
        """Create pandas DataFrame from loaded data"""
        if not self.metadata:
            raise ValueError("No data loaded. Call load_fasta() first.")
        
        df = pd.DataFrame(self.metadata)
        df['sequence'] = self.sequences
        
        self.logger.info(f"Created DataFrame with shape: {df.shape}")
        
        return df
    
    def save_processed_data(self, df: pd.DataFrame, output_path: str):
        """Save processed data to CSV and HDF5 formats"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to CSV (without sequences for readability)
        csv_path = output_path.with_suffix('.csv')
        df_no_seq = df.drop('sequence', axis=1)
        df_no_seq.to_csv(csv_path, index=False)
        self.logger.info(f"Saved metadata to: {csv_path}")
        
        # Save to HDF5 (with sequences) OR fallback to Parquet if PyTables not available
        h5_path = output_path.with_suffix('.h5')
        try:
            df.to_hdf(h5_path, key='wnv_data', mode='w', complevel=9)
            self.logger.info(f"Saved complete data (HDF5) to: {h5_path}")
        except Exception as e:
            self.logger.warning(f"HDF5 save failed ({e}); falling back to Parquet")
            parquet_path = output_path.with_suffix('.parquet')
            try:
                df.to_parquet(parquet_path, index=False)
                self.logger.info(f"Saved complete data (Parquet) to: {parquet_path}")
            except Exception as e2:
                self.logger.error(f"Failed to save data in any binary format: {e2}")
                parquet_path = None
            return csv_path, parquet_path
        
        return csv_path, h5_path