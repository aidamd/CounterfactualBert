import re


def preprocess(df):
    print(df.shape[0], "datapoints in dataset")
    df = clean(df, "text")
    df = remove_empty(df, "text")
    print(df.shape[0], "datapoints after removing empty strings")
    return df

def remove_empty(corpus, col):
    drop = list()
    for i, row in corpus.iterrows():
        if row[col] == "":
            drop.append(i)
    return corpus.drop(drop)

def clean(df, col):
    link_re = re.compile(r"(http(s)?[^\s]*)|(pic\.[s]*)")
    hashtag_re = re.compile(r"#[a-zA-Z0-9_]+")
    mention_re = re.compile(r"@[a-zA-Z0-9_]+")
    # nonalph_re = re.complie(r"")

    for i, row in df.iterrows():
        text = row[col].lower()
        text = link_re.sub("", text)
        text = hashtag_re.sub("", text)
        text = mention_re.sub("", text)
        df.at[i, col] = text
    return df

def get_batches(data, data_idx, batch_size, tokenizer, counter=[], hate=None):
    batches = []
    for s in range(0, len(data), batch_size):
        e = s + batch_size #if s + batch_size < len(data) else len(data)
        data_batch = data[s: e]
        idx_batch = data_idx[s: e]
        hate_batch = hate[s: e] if hate else None

        data_info = batch_to_info(data_batch, hate_batch, idx_batch, tokenizer, counter)
        batches.append(data_info)
    return batches

def batch_to_info(batch, hate, idx, tokenizer, cf):
    max_len = 128
    batch_info = list()
    
    tokens = tokenizer(batch,
            #return_tensors='pt',
            padding="max_length",
            max_length=max_len,
            truncation=True)
    inputs = tokens["input_ids"]
    attentions = tokens["attention_mask"]
    """
    max_len = len(inputs[0])
    if max_len > 150:
        max_len = 150
        tokens = tokenizer(batch,
            #return_tensors='pt',
            padding="max_length",
            max_length=max_len,
            truncation=True)

        inputs = tokens["input_ids"]
        attentions = tokens["attention_mask"]
    """
    for i, sent in enumerate(batch):
        if idx[i] not in cf:
            counter_inputs = [inputs[i]]
            counter_attention = [attentions[i]]
        else:
            counter_tokens = tokenizer(cf[idx[i]],
                    #return_tensors='pt',
                    padding="max_length",
                    max_length=max_len,
                    truncation=True)
            counter_inputs = counter_tokens["input_ids"]
            counter_attention = counter_tokens["attention_mask"]
        sentence = {
            "input": inputs[i],
            "attention": attentions[i],
            "counter_input": counter_inputs,
            "counter_attention": counter_attention,
            "length": len(sent),
            "hate": hate[i] if hate else None,
        }
        
        batch_info.append(sentence)
    return batch_info
