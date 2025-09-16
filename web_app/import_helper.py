"""
Import Helper - Tự động xử lý import paths cho cả local và Streamlit Cloud
"""

import sys
import os

def setup_imports():
    """Setup import paths for both local and Streamlit Cloud environments"""
    # Get project root
    current_file = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(current_file))
    
    # Add project root to path
    if project_root not in sys.path:
        sys.path.append(project_root)
    
    # Add src directory to path for fallback imports
    src_path = os.path.join(project_root, 'src')
    if src_path not in sys.path:
        sys.path.append(src_path)

def safe_import(module_path, fallback_path=None):
    """
    Safely import module with fallback for Streamlit Cloud
    
    Args:
        module_path: Primary import path (e.g., 'src.data.data_provider')
        fallback_path: Fallback import path (e.g., 'data.data_provider')
    
    Returns:
        Imported module
    """
    try:
        parts = module_path.split('.')
        module = __import__(module_path, fromlist=[parts[-1]])
        return getattr(module, parts[-1])
    except ImportError:
        if fallback_path:
            try:
                parts = fallback_path.split('.')
                module = __import__(fallback_path, fromlist=[parts[-1]])
                return getattr(module, parts[-1])
            except ImportError:
                pass
        raise ImportError(f"Could not import {module_path}")

# Auto setup when imported
setup_imports()
