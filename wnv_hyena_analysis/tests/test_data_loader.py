"""
Tests for WNV data loader functionality
"""

import pytest
import tempfile
import os
from pathlib import Path
import pandas as pd

from src.analysis.data_loader import WNVDataLoader
from src.utils.config import Config


@pytest.fixture
def sample_fasta():
    """Create a sample FASTA file for testing"""
    fasta_content = """>seq1 West Nile virus strain NY99 isolate from USA 1999
ATGAAAAACCCAAAAAAGAAATCCGGAGGATTCCGGATTGTCAATATGCTAAAACGCGGAGTAGCCCGTGTGAGCCCCTTTGGGGGCTTGAAGAGGCTTACCCTAGAGTGGAAGG
>seq2 West Nile virus strain Greece/2010 isolate from Greece host:Culex 2010
ATGAAAAACCCAAAAAAGAAATCCGGAGGATTCCGGATTGTCAATATGCTAAAACGCGGAGTAGCCCGTGTGAGCCCCTTTGGGGGCTTGAAGAGGCTTACCCTAGAGTGGAAGG
>seq3 West Nile virus lineage 1 Italy/2008
ATGAAAAACCCAAAAAAGAAATCCGGAGGATTCCGGATTGTCAATATGCTAAAACGCGGAGTAGCCCGTGTGAGCCCCTTTGGGGGCTTGAAGAGGCTTACCCTAGAGTGGAAGG
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
        f.write(fasta_content)
        f.flush()
        yield f.name
    
    # Cleanup
    os.unlink(f.name)


@pytest.fixture
def test_config():
    """Create test configuration"""
    config_dict = {
        'sequence': {
            'min_length': 50,
            'max_length': 20000,
            'min_quality': 0.8
        }
    }
    return config_dict


class TestWNVDataLoader:
    """Test WNV data loader"""
    
    def test_load_fasta(self, sample_fasta, test_config):
        """Test FASTA file loading"""
        loader = WNVDataLoader(test_config)
        sequences, metadata = loader.load_fasta(sample_fasta)
        
        assert len(sequences) == 3
        assert len(metadata) == 3
        assert all(len(seq) > 0 for seq in sequences)
    
    def test_metadata_extraction(self, sample_fasta, test_config):
        """Test metadata extraction from headers"""
        loader = WNVDataLoader(test_config)
        sequences, metadata = loader.load_fasta(sample_fasta)
        
        # Check first sequence metadata
        meta1 = metadata[0]
        assert 'NY99' in meta1['strain']
        assert meta1['year'] == 1999
        assert meta1['country'] == 'USA'
        
        # Check second sequence metadata  
        meta2 = metadata[1]
        assert 'Greece' in meta2['strain']
        assert meta2['year'] == 2010
        assert meta2['country'] == 'Greece'
        assert meta2['host'] == 'Culex'
    
    def test_quality_control(self, test_config):
        """Test sequence quality control"""
        # Create sequence with too many Ns
        fasta_content = """>low_quality
ATGAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
            f.write(fasta_content)
            f.flush()
            
            loader = WNVDataLoader(test_config)
            sequences, metadata = loader.load_fasta(f.name)
            
            # Should be filtered out due to high N content
            assert len(sequences) == 0
        
        os.unlink(f.name)
    
    def test_sample_size_limit(self, sample_fasta, test_config):
        """Test sampling functionality"""
        loader = WNVDataLoader(test_config)
        sequences, metadata = loader.load_fasta(sample_fasta, sample_size=2)
        
        assert len(sequences) == 2
        assert len(metadata) == 2
    
    def test_create_dataframe(self, sample_fasta, test_config):
        """Test DataFrame creation"""
        loader = WNVDataLoader(test_config)
        sequences, metadata = loader.load_fasta(sample_fasta)
        df = loader.create_dataframe()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert 'sequence' in df.columns
        assert 'country' in df.columns
        assert 'year' in df.columns
    
    def test_composition_calculation(self, test_config):
        """Test sequence composition calculation"""
        loader = WNVDataLoader(test_config)
        
        # Test sequence with known composition
        sequence = "AAAATTTTGGGGCCCC"  # 25% each base
        composition = loader._calculate_composition(sequence)
        
        assert abs(composition['gc_content'] - 50.0) < 0.1
        assert abs(composition['at_content'] - 50.0) < 0.1
        assert composition['n_content'] == 0.0
    
    def test_geographic_parsing(self, test_config):
        """Test geographic information parsing"""
        loader = WNVDataLoader(test_config)
        
        # Test various header formats
        test_cases = [
            ("West Nile virus from USA 2020", "USA"),
            ("WNV isolate Italy/Milan/2018", "Italy"), 
            ("strain Greece-2019", "Greece"),
            ("virus from unknown location", "Unknown")
        ]
        
        for header, expected_country in test_cases:
            metadata = loader._parse_header(header)
            assert metadata['country'] == expected_country
    
    def test_year_extraction(self, test_config):
        """Test year extraction from headers"""
        loader = WNVDataLoader(test_config)
        
        test_cases = [
            ("strain NY99 1999", 1999),
            ("isolate 2020", 2020),
            ("virus/2018/location", 2018),
            ("no year mentioned", None)
        ]
        
        for header, expected_year in test_cases:
            year = loader._extract_year(header)
            assert year == expected_year