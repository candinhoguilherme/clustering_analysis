import numpy as np
import pandas as pd

spreadsheet = 'analise.xlsx'

columns = [
    'ID_PRODUTO',
    'NOMEPRODUTO',
    'UNIDADE',
    'CUSTOEST',
    'QTD',
    'PRECONORMAL',
    'TOTALPRECONORMAL',
    'CUSTO',
    'VENDA',
    'TOTALPROMOCAO',
    'PART_NORMAL',
    'PART_PROMO',
    'PART_VENDA',
    'CUSTONORMAL',
    'CUSTOPROMOCIONAL',
    'MVB',
    'MVN',
    'MVP',
    'MVPTOT',
    'MVPINDIV',
    'MARGEM',
    'MARGEM_PROMOCAO',
    'MARGEM_NORMAL'
]

df = pd.read_excel(
    spreadsheet,
    engine='openpyxl',
    usecols=columns
)

df.to_excel(
    "clean_analise.xlsx",
    index=False
)