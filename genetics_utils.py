import json
from pandas import Series
import numpy as np

CODE = json.load(open('genetic_code.json'))

def _translate_str(sequence: str, ):
    """
    Translate a DNA sequence into a protein sequence.
    """
    protein = ''
    for i in range(0, len(sequence), 3):
        codon = sequence[i:i+3]
        protein += CODE[codon]
    return protein

def _translate_series(series: Series):
    """
    Translate a Series of DNA sequences into a Series of protein sequences.
    """
    codon = np.arange(0, len(series)) // 3
    codon = Series(codon, index=series.index)
    return series.groupby(codon).apply(lambda s: CODE[s.str.cat()])

def translate(sequence):
    """
    Translate a DNA sequence into a protein sequence.
    """
    if isinstance(sequence, Series):
        return _translate_series(sequence)
    elif isinstance(sequence, str):
        return _translate_str(sequence) 
    else:
        try:
            return _translate_str(''.join(sequence))
        except Exception as e:
            raise ValueError('Invalid input') from e