import json
import os
import pandas as pd
import sqlite3


def get_dataset_filtered_commits() -> pd.DataFrame:
    patterns_EN = ['fix','bug','patch','repair','refactor','address','error','conflict','revert']
    patterns_CN = ['修正','错误','补丁','修理','重构','地址','错误','冲突','还原']
    patterns_RU = ['исправить', 'ошибка', 'исправление', 'исправить', 'рефакторинг', 'адрес', 'ошибка', 'конфликт', 'отменить']
    patterns_JP = ['修正する','バグ','パッチ','修理する','リファクタリング','アドレス','エラー','競合','復元する']
    patterns_SP = ['arreglar','error','parche','reparar','refactorizar','direccionar','error','conflicto','revertir']
    patterns_FR = ['corriger','erreur','pilote','réparer','refactoriser','adresse','erreur','conflit','inverser']
    patterns_DE = ['korrigieren','Fehler','Patch','reparieren','refaktorisieren','Adresse','Fehler','Konflikt','zurücksetzen']
    patterns = patterns_EN + patterns_CN + patterns_RU + patterns_JP + patterns_SP + patterns_FR + patterns_DE

    query = """
    SELECT * FROM commitpackft
    WHERE message like '%fix%'
    OR message like '%bug%'
    OR message like '%patch%'
    OR message like '%repair%'
    OR message like '%refactor%'
    OR message like '%address%'
    OR message like '%error%'
    OR message like '%conflict%'
    OR message like '%revert%'
    OR message like '%修正%'
    OR message like '%错误%'
    OR message like '%补丁%'
    OR message like '%修理%'
    OR message like '%重构%'
    OR message like '%地址%'
    OR message like '%错误%'
    OR message like '%冲突%'
    OR message like'%还原%'
    OR message like '%исправить%'
    OR message like'%ошибка%'
    OR message like '%исправление%'
    OR message like'%исправить%'
    OR message like '%рефакторинг%'
    OR message like'%адрес%'
    OR message like '%ошибка%'
    OR message like'%конфликт%'
    OR message like '%отменить%'
    OR message like '%修正する%'
    OR message like '%バグ%'
    OR message like'%パッチ%'
    OR message like '%修理する%'
    OR message like'%リファクタリング%'
    OR message like '%アドレス%'
    OR message like'%エラー%'
    OR message like '%競合%'
    OR message like'%復元する%'
    OR message like '%arreglar%'
    OR message like'%error%'
    OR message like'%parche%'
    OR message like '%reparar%'
    OR message like '%refactorizar%'
    OR message like '%direccionar%'
    OR message like '%error%'
    OR message like '%conflicto%'
    OR message like'%revertir%'
    OR message like'%corriger%'
    OR message like'%erreur%'
    OR message like '%pilote%'
    OR message like '%réparer%'
    OR message like '%refactoriser%'
    OR message like '%adresse%'
    OR message like'%erreur%'
    OR message like '%conflit%'
    OR message like '%inverser%'
    OR message like '%korrigieren%'
    OR message like '%Fehler%'
    OR message like'%Patch%'
    OR message like '%reparieren%'
    OR message like '%refaktorisieren%'
    OR message like '%Adresse%'
    OR message like '%Fehler%'
    OR message like '%Konflikt%'
    """

    if os.path.exists('/content/drive/MyDrive/Thesis/commipack_datasets.sql'):
        con = sqlite3.connect('/content/drive/MyDrive/Thesis/commipack_datasets.sql')
    else:
        con = sqlite3.connect('commipack_datasets.sql')
    dataset_df = pd.read_sql_query(query, con)
    return dataset_df

def writeJS(path: str, contents: str):
    if os.path.exists(path):
        os.remove(path)
    with open(path,'w') as f:
        f.write(contents)
        
def filter_eslint(output: dict):
    output_new = output[0]['messages']
    output_old = output[1]['messages']
    return 1 if len(output_old) < len(output_new) else 0