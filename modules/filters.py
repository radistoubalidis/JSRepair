import os
import sqlite3
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from collections import Counter

from df2sql import sqlite2postgres
from sklearn.model_selection import train_test_split
from difflib import SequenceMatcher
from typing import List
from transformers import RobertaTokenizer
import random


def compute_diffs(sample: dict, dmp):
    # Compute the differences
    diffs = dmp.diff_main(sample['old_contents'], sample['new_contents'])
    dmp.diff_cleanupSemantic(diffs)
    # Count the changes
    return sum(1 for diff in diffs if diff[0] == 1)

def mask(buggy_code: str, correct_code: str, tokenizer: RobertaTokenizer) -> str:
    """Η συναρτηση χρησιμοποιεί τον tokenizer του μοντελου,
    για να μετατρέψει τον κωδικα σε μια λιστα απο word tokens
    πανω στις οποιες εφαρμοζεται συγκριση χαρακτηρα προς χαρακτηρα
    για να βρεθουν τα σημεία στην λιστα που βρισκονταιοι διαφορες τους,
    ώστε να χρησιμοποιηθουν τα συγκεκριμενα στοιχεια της λιστας στον μηχανισμο
    αποκρυψης.
    Σεναριο 1: Αν το συνολο των word tokens που αλλαξαν δεν ειναι
    μεγαλύτερο απο το 1/4 του συνολου των word tokens τοτε εφαρμοζεται
    το στοιχειο μασκα του tokenizer στα word token που αλλαξαν.
    Σεναριο 2: Αν το συνολο των word tokens που αλλαξαν ειναι
    μεγαλύτερο απο το 1/4 του συνολου των word tokens τοτε εφαρμοζεται η μασκα στα word tokens με τυχαιο τροπο.
    Επιστρέφεται ο κώδικας σε μορφη string αφοτου εφαρμόστηκε η τεχνικη
    αποκρυψης στα επιλεγμενα στοιχεια.

    Αν σε ολα τα δειγματα εφαρμοζοταν η τεχνικη αποκρυψης
    με βαση τις διαφορες στον κωδικα θα δημιουργοταν θορυβος
    στο dataset με πολλα outlier δειγματα (π.χ. σε ενα δειγμα που
    υπήρχαν πολλές γραμμές με σχόλια και αφαιρεθηκαν, θα δημιουργοταν
    ενα δειγμα που το συνολο των word token του θα ηταν το στοιχείο αποκρυψης)

    Args:
        buggy_code (str): code before commit
        correct_code (str): code after commit
        tokenizer (RobertaTokenizer): codebert's tokenizer

    Returns:
        _type_: str
    """
    buggy_tokens = tokenizer.tokenize(buggy_code)
    correct_tokens = tokenizer.tokenize(correct_code)
    indices = get_changed_token_indices(buggy_tokens, correct_tokens)
    masked_buggy_tokens = buggy_tokens
    # if len(indices) <= len(buggy_tokens) / 4:
    #     for i1, i2 in indices:
    #         if abs(i2-i1) == 0:
    #             masked_buggy_tokens[i1-1] = tokenizer.mask_token
    #         if abs(i2-i1) == 1:
    #             masked_buggy_tokens[i1] = tokenizer.mask_token
    #         else:
    #             for idx in range(i1,i2):
    #                 masked_buggy_tokens[idx] = tokenizer.mask_token
    # else:
    #     num_random_masks = random.randint(1, int(len(buggy_tokens) / 4))
    #     random_indices = random.sample(range(1,len(buggy_tokens)), num_random_masks)
    #     for ri in random_indices:
    #         masked_buggy_tokens[ri] = tokenizer.mask_token
    if len(indices) <= 1:
        try:
            masked_buggy_tokens[indices[0][0]] = tokenizer.mask_token
        except Exception as e:
            masked_buggy_tokens[indices[0][0]-1] = tokenizer.mask_token
    else:
        random_change_index = random.randint(1, len(indices) - 1)
        masked_buggy_tokens[random_change_index] = tokenizer.mask_token
    return tokenizer.convert_tokens_to_string(masked_buggy_tokens)


def count_comment_lines(sample: str) -> int:
    comment_blocks = []
    start_index = -1
    for i, line in enumerate(sample.splitlines()):
        if line.strip().startswith('/*'):
            start_index = i
        elif line.strip().endswith('*/'):
            comment_blocks.append([start_index, i])
            start_index = -1

    comment_lines_count = sum([c[1]-c[0] for c in comment_blocks])

    for i, line in enumerate(sample.splitlines()):
        if line.strip().startswith('//'):
            comment_lines_count += 1
    return comment_lines_count

def preprocess_text(text: str, get_tokens: bool = True):
    # Convert to lowercase
    text = text.lower()
    
    # remove punctuation marks
    punctMarks = [".", ",", "?", "!", ":", ";" ,"\'", "\"", "`", "-", "(", ")", "[", "]", "{", "}", "...", "/", "\\", "•", "*", "^", "_", "<", ">",]
    for pM in punctMarks:
        if pM in text:
            text = text.replace(pM, '')
            
    # Tokenization
    tokens = word_tokenize(text)
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Stemming
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(word) for word in tokens]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in stemmed_tokens]
    
    if get_tokens:
        return lemmatized_tokens
    return " ".join(lemmatized_tokens)
    
def check_bug(text: str):
    bug_keywords = [
        "fix","bug","error","issue","debug","repair","correct","patch","resolve","adjust","handle",
        "defect","broken","crash","problem","exception","restore","sanitize","refactor","stabilize"
    ]
    if any(word in text for word in bug_keywords):
        return True
    return False

def create_word_frequency(commit_messages):
    all_words = []
    for message in commit_messages:
        processed_words = preprocess_text(message)
        all_words.extend(processed_words)
    word_counts = Counter(all_words)
    return word_counts

def classify_message(msg: str, bugTypes: pd.DataFrame):
    bTypes = []
    bugTypes = bugTypes.to_dict(orient='records')
    for bT in bugTypes:
        if any(word in msg for word in bT['keywords']):
            bTypes.append(bT['type'])
    if len(bTypes) == 0:
        return "general"
    return ",".join(bTypes)

def split(df: pd.DataFrame, sqlite_con, test_size = 0.2, psql_convert = False):
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
    train_df.to_sql('commitpackft_classified_train', con=sqlite_con, if_exists='replace')
    test_df.to_sql('commitpackft_classified_test', con=sqlite_con, if_exists='replace')
    if psql_convert:
        sqlite2postgres(train_df, 'commitpackft_classified_train')
        sqlite2postgres(test_df, 'commitpackft_classified_test')

def identify(query: str, sqlite_path: str, psql_convert = False) -> pd.DataFrame:
    if not os.path.exists(sqlite_path):
        raise RuntimeError('Invalid Sqlite path.')
    con = sqlite3.connect(sqlite_path)
    df = pd.read_sql_query(query, con)

    print("# Create Word Count Dictionary")
    wordsFreq = create_word_frequency(df['message'].tolist())
    wordsFreqDict = dict(wordsFreq)
    wordsCount = []
    for key in wordsFreqDict:
        wordsCount.append({
            'word': key,
            'count': wordsFreqDict[key]
        })
    wordCount = pd.DataFrame(wordsCount)
    
    if psql_convert:
        wordCount.to_sql('commitpackft_word_count', con, if_exists='replace')
        sqlite2postgres(wordCount, 'commitpackft_word_count')

    # Process Commits and store 
    print("# Processing Messages")
    df['processed_message'] = df['message'].apply(lambda msg: preprocess_text(text=msg, get_tokens=False))
    
    print("# Detecting bugs")
    df["is_bug"] = df["processed_message"].apply(lambda msg: check_bug(text=msg))
    df = df[df['is_bug'] == True]
    if psql_convert:
        df.to_sql(name='commitpackft_bugs', con=con, if_exists='replace')
        sqlite2postgres(df, 'commitpackft_bugs')

    print("# Classifying samples")
    bugTypes = pd.read_sql("select * from bug_types", con)
    bugTypes['keywords'] = bugTypes['keywords'].str.split(',')
    df['bug_type'] = df["processed_message"].apply(lambda msg: classify_message(msg=msg, bugTypes=bugTypes))
    
    if psql_convert:
        df.to_sql(name='commitpackft_classified' ,con=con, if_exists='replace')
        sqlite2postgres(df, 'commitpackft_classified')
        
    # split the dataset into train and test
    # only for the samples that are bug identified
    
    split(df[df['bug_type'].str.len() > 0], sqlite_con=con, psql_convert=True)
    con.close()

def mask_code_diff(code_before: str, code_after: str, tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base-mlm')):
    """
    Αν η αλλαγη στον κωδικα ειναι πανω απο ενα token
    θα πρεπει να εισαχθουν mask token ισα με το συνολο 
    token που θα εβγαζε ο roberta tokenizer
    """
    tokens_before = tokenizer.tokenize(code_before)
    tokens_after = tokenizer.tokenize(code_after)

    token_ids_before = tokenizer.convert_tokens_to_ids(tokens_before)
    token_ids_after = tokenizer.convert_tokens_to_ids(tokens_after)

    mask_token_id = tokenizer.mask_token_id
    
    changed_token_indices = get_changed_token_indices(token_ids_before, token_ids_after)

    masked_token_ids_before = token_ids_before.copy() 
    for i1, i2 in changed_token_indices:
        for idx in range(i1, i2):
            masked_token_ids_before[idx] = mask_token_id
            
    # Print the masked token IDs for the "before" sequence
    masked_tokens = tokenizer.convert_ids_to_tokens(masked_token_ids_before)
    masked_seq = tokenizer.convert_tokens_to_string(masked_tokens)
    return masked_seq
    

def get_changed_token_indices(token_ids_before, token_ids_after):
    sm = SequenceMatcher(None, token_ids_before, token_ids_after)
    changed_indices = []

    for opcode, i1, i2, j1, j2 in sm.get_opcodes():
        if opcode != 'equal':
            changed_indices.append([i1, i2])

    return changed_indices

def add_labels(bType: List[str], classLabels: dict):
    sample_labels = {}
    for key in classLabels.keys():
        if key in bType:
            sample_labels[key] = 1.
        else:
            sample_labels[key] = 0.
    return list(sample_labels.values())

def bug_type_dist_query(with_mobile: bool, table: str) -> str:
    if with_mobile:
        query = f"""select count(*) ,'general' as "bug_type"
        from {table}
        where bug_type like '%general%'
        union 
        select count(*), 'mobile' as "bug_type" 
        from {table}
        where bug_type like '%mobile%'
        union
        select count(*), 'functionality' as "bug_type" 
        from {table}
        where bug_type like '%functionality%'
        union
        select count(*), 'ui-ux' as "bug_type" 
        from {table}
        where bug_type like '%ui-ux%'
        union
        select count(*), 'compatibility-performance' as "bug_type" 
        from {table}
        where bug_type like '%compatibility-performance%'
        union
        select count(*), 'network-security' as "bug_type" 
        from {table}
        where bug_type like '%network-security%';"""
    else:
        query = f"""select count(*) ,'general' as "bug_type"
        from {table}
        where bug_type like '%general%'
        union 
        select count(*), 'functionality' as "bug_type" 
        from {table}
        where bug_type like '%functionality%'
        union
        select count(*), 'ui-ux' as "bug_type" 
        from {table}
        where bug_type like '%ui-ux%'
        union
        select count(*), 'compatibility-performance' as "bug_type" 
        from {table}
        where bug_type like '%compatibility-performance%'
        union
        select count(*), 'network-security' as "bug_type" 
        from {table}
        where bug_type like '%network-security%';"""
    return query