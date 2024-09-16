import json
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
import re
from transformers import RobertaTokenizer


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
    

def create_word_frequency(commit_messages):
    all_words = []
    for message in commit_messages:
        processed_words = preprocess_text(message)
        all_words.extend(processed_words)
    word_counts = Counter(all_words)
    return word_counts

def classify_message(msg: str, bugTypes: pd.DataFrame):
    bTypes = []
    for i,bType in bugTypes.iterrows():
        bTD = bType.to_dict()
        if any(word in msg for word in bTD['keywords']):
            bTypes.append(bTD['type'])
    return ",".join(bTypes)

def split(df: pd.DataFrame, sqlite_con, test_size = 0.2, psql_convert = False):
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
    train_df.to_sql('commitpackft_classified_train', con=sqlite_con)
    test_df.to_sql('commitpackft_classified_test', con=sqlite_con)
    if psql_convert:
        sqlite2postgres(train_df, 'commitpackft_classified_train')
        sqlite2postgres(test_df, 'commitpackft_classified_test')

def identify(query: str, sqlite_path: str, psql_convert = False) -> pd.DataFrame:
    if not os.path.exists(sqlite_path):
        raise RuntimeError('Invalid Sqlite path.')
    con = sqlite3.connect(sqlite_path)
    df = pd.read_sql_query(query, con).set_index('index')

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
        con.cursor().execute("drop table commitpackft_word_count")
        wordCount.to_sql('commitpackft_word_count', con)
        sqlite2postgres(wordCount, 'commitpackft_word_count')

    # Process Commits and store 
    print("# Processing Commits")
    df["processed_messages"] = df["message"].apply(lambda msg: preprocess_text(text=msg, get_tokens=False))
    if psql_convert:
        con.cursor().execute('drop table commitpackft_processed_commits')
        df.to_sql(name='commitpackft_processed_commits', con=con)
        sqlite2postgres(df, 'commitpackft_processed_commits')

    print("# Classifying samples")
    bugTypes = pd.read_sql("select * from bug_types", con)
    bugTypes['keywords'] = bugTypes['keywords'].str.split(',')
    df['bug_type'] = df["processed_messages"].apply(lambda msg: classify_message(msg=msg, bugTypes=bugTypes))
    
    if psql_convert:
        con.cursor().execute("drop table commitpackft_classified")
        df.to_sql(name='commitpackft_classified' ,con=con)
        sqlite2postgres(df, 'commitpackft_classified')
    
    # split the dataset into train and test
    # only for the samples that are bug identified
    
    split(df[df['bug_type'].str.len() > 0], sqlite_con=con)
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
            changed_indices.append((i1, i2))  # Store only the indices of changes in the before sequence

    return changed_indices