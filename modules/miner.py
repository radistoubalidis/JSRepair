import pandas as pd

def get_fix_commits(dataset_res: dict) -> pd.DataFrame:
    dataset_list = []
    for item in dataset_res['rows']:
        dataset_list.append(item['row'])
    df = pd.DataFrame(dataset_list)
    patterns_EN = ['fix','bug','patch','repair','refactor','address','error','conflict','revert']
    patterns_CN = ['修正','错误','补丁','修理','重构','地址','错误','冲突','还原']
    patterns_RU = ['исправить', 'ошибка', 'исправление', 'исправить', 'рефакторинг', 'адрес', 'ошибка', 'конфликт', 'отменить']
    patterns_JP = ['修正する','バグ','パッチ','修理する','リファクタリング','アドレス','エラー','競合','復元する']
    patterns_SP = ['arreglar','error','parche','reparar','refactorizar','direccionar','error','conflicto','revertir']
    patterns_FR = ['corriger','erreur','pilote','réparer','refactoriser','adresse','erreur','conflit','inverser']
    patterns_DE = ['korrigieren','Fehler','Patch','reparieren','refaktorisieren','Adresse','Fehler','Konflikt','zurücksetzen']
    patterns = patterns_EN + patterns_CN + patterns_RU + patterns_JP + patterns_SP + patterns_FR + patterns_DE
    buggy_rows =  df[df['message'].str.contains('|'.join(patterns), case=False, na=False)]
    only_code = buggy_rows[['old_contents','new_contents']].copy()
    return only_code
        
