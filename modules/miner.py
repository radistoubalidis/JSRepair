import pandas as pd

def get_fix_commits(dataset_res: dict) -> pd.DataFrame:
    dataset_list = []
    for item in dataset_res['rows']:
        dataset_list.append(item['row'])
    df = pd.DataFrame(dataset_list)
    pattern = ['fix','bug','patch', 'repair', 'refactor','address']
    buggy_rows =  df[df['message'].str.contains(pattern, case=False, na=False)]
    only_code = buggy_rows[['old_contents','new_contents']].copy()
    return only_code
        
