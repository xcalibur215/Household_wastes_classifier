#!/usr/bin/env python3
"""
Script to fix Unicode encoding issues in Jupyter notebooks
Run this script to clean up problematic characters before committing to GitHub
"""

import json
import re
import sys

def clean_unicode_string(text):
    """Clean problematic Unicode characters from text"""
    if not isinstance(text, str):
        return text
    
    # Replace problematic Unicode characters with safe alternatives
    replacements = {
        '🚀': '[ROCKET]',
        '🎯': '[TARGET]',
        '📊': '[CHART]',
        '🏗️': '[CONSTRUCTION]', 
        '🔧': '[WRENCH]',
        '✅': '[CHECK]',
        '❌': '[X]',
        '🎉': '[PARTY]',
        '📦': '[PACKAGE]',
        '📂': '[FOLDER]',
        '🏷️': '[TAG]',
        '🔄': '[REFRESH]',
        '🖼️': '[IMAGE]',
        '🤖': '[ROBOT]',
        '🧠': '[BRAIN]',
        '🏃': '[RUNNER]',
        '🧹': '[BROOM]',
        # Add more as needed
    }
    
    for emoji, replacement in replacements.items():
        text = text.replace(emoji, replacement)
    
    # Remove any remaining problematic Unicode characters
    text = re.sub(r'[\udcca-\udcff]', '', text)
    
    return text

def clean_notebook(notebook_path):
    """Clean a Jupyter notebook of problematic Unicode characters"""
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        
        # Clean all cell sources
        for cell in notebook.get('cells', []):
            if 'source' in cell:
                if isinstance(cell['source'], list):
                    cell['source'] = [clean_unicode_string(line) for line in cell['source']]
                else:
                    cell['source'] = clean_unicode_string(cell['source'])
            
            # Clean outputs if they exist
            if 'outputs' in cell:
                for output in cell['outputs']:
                    if 'text' in output:
                        if isinstance(output['text'], list):
                            output['text'] = [clean_unicode_string(line) for line in output['text']]
                        else:
                            output['text'] = clean_unicode_string(output['text'])
        
        # Save the cleaned notebook
        clean_path = notebook_path.replace('.ipynb', '_clean.ipynb')
        with open(clean_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, ensure_ascii=True, indent=2)
        
        print(f"✓ Cleaned notebook saved as: {clean_path}")
        return clean_path
        
    except Exception as e:
        print(f"Error cleaning notebook: {e}")
        return None

if __name__ == "__main__":
    notebook_path = "waste_classifier_DN201.ipynb"
    clean_notebook(notebook_path)
