import os
import pandas as pd
from tqdm import tqdm
from keras.preprocessing.sequence import pad_sequences
import re
import torch


class Preprocessing():

    def __init__(self, predict_dir, sgt_path, result_dir, found_sgts=[]):
        self.predict_dir = predict_dir
        self.sgt_file_path = sgt_path
        with open(sgt_path) as f:
            content = f.readlines()
            self.sgts = [item.replace("\n", "") for item in content]
        self.result_dir = result_dir
        self.tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'gpt2')
        self.model = torch.hub.load('huggingface/pytorch-transformers', 'modelWithLMHead', 'gpt2')
        self.found_sgts = found_sgts
        self.device = "cuda"
        self.model = self.model.to(self.device)

    def extract_posts_with_sgt(self):

        for filename in os.listdir(self.predict_dir):
            print("--------------------------------")
            print(os.path.join(self.predict_dir, filename))
            print("--------------------------------")

            dataset = pd.read_csv(os.path.join(self.predict_dir, filename))
            dataset = dataset.rename(columns={"body": "text"})
            detected = []
            drop = list()
            for index, row in tqdm(dataset.iterrows()):
                text = row.text.lower()
                for sgt in self.sgts:
                     if re.findall("((^|[^\w])(" + sgt + ")([^\w]|$|s))", text):
                    #if sgt in text:
                        #row_dict["detected_sgt"] = sgt
                        detected.append(sgt)
                        break
                else:
                    detected.append("")
                    drop.append(index)
            dataset["detected_sgt"] = pd.Series(detected)
            dataset = dataset.drop(drop)
            result_path = os.path.join(self.result_dir, "SGT_" + filename)
            print("saving to", result_path)
            dataset.to_csv(result_path)

    def extract_all_sgts(self):
        for filename in os.listdir(self.result_dir):
            if not filename.startswith("SGT_"):
                continue
            print("--------------------------------")
            print(os.path.join(self.result_dir, filename))
            print("--------------------------------")
            dataset = pd.read_csv(os.path.join(self.result_dir, filename))
            #rows_list = []
            all_sgts = list()
            for index, row in tqdm(dataset.iterrows()):
                text = row.text.lower()
                present_sgts = [sgt for sgt in self.sgts if
                                re.findall("((^|[^\w])(" + sgt + ")([^\w]|$|s))", text)]

                for sgt in present_sgts:
                    if sgt not in self.found_sgts:
                        self.found_sgts.append(sgt)

                present_sgts_str = ','.join(present_sgts)
                all_sgts.append(present_sgts_str)

            #sgt_data_df = pd.DataFrame(rows_list)
            dataset["sgts"] = pd.Series(all_sgts)
            result_path = os.path.join(self.result_dir, "Populated_" + filename)
            print("saving to", result_path)
            dataset.to_csv(result_path)

    def merge_preprocessed_chunks(directory):
        os.chdir(directory)
        all_filenames = [i for i in os.listdir()]
        combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames])
        combined_csv.to_csv("SGT-Gab.csv", index=False, encoding='utf-8-sig')


    def get_perplexity(self, all_sentences, MAX_LEN=128, chosen_batch_size=64):
        # TODO: discuss MAX_LEN with Aida
        perplexities = []

        for w in range(0, len(all_sentences), chosen_batch_size):
            input_ids = []

            sentences = all_sentences[w:w + chosen_batch_size]

            for sen in sentences:
                token_ids = self.tokenizer.encode(sen, add_special_tokens=True)
                input_ids.append(token_ids)

            input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long",
                                      value=0, truncating="post", padding="post")
            input_ids = torch.tensor(input_ids)
            input_ids = input_ids.to(self.device)

            outputs = self.model(input_ids)
            last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
            batch_size = last_hidden_states.shape[0]

            # print("shape of last hid", last_hidden_states.shape)
            for k in range(len(sentences)):
                perplexity = 0
                for i in range(MAX_LEN):
                    index = input_ids[k][i].item()
                    probs = torch.nn.functional.softmax(last_hidden_states[k, i, :], dim=0)
                    prob = probs[index]
                    perplexity += torch.log(prob.data)

                perplexities.append(perplexity)

        return perplexities


    def generate_counterfactuals(self, filename):
        # TODO: Comment the following line
        # TODO: Change to work over batches (a convenient choice is using all sgt-substitued texts for each record as a batch)
        print("reading", filename)
        dataset = pd.read_csv(filename)
        print("finished reading ", filename)
        results = {
            "text": list(),
            "id": list(),
            "original_sgt": list(),
            "new_sgt": list(),
            "perplexity": list()
        }
        result_path = os.path.join(self.result_dir, "Counterfactuals_" + filename.split("/")[-1])

        print("saving counterfactuals to", result_path)
        pd.DataFrame(results).to_csv(result_path, index=False)

        for index, row in tqdm(dataset.iterrows()):
            text = row.text.lower()
            current_sgts = row.sgts.split(",")
            sentences = []
            for from_sgt in current_sgts:
                for to_sgt in self.found_sgts:
                    results["id"].append(row.id)
                    results["original_sgt"].append(from_sgt)
                    results["new_sgt"].append(to_sgt)
                    new_text = re.sub("((^|[^\w])(%s)([^\w]|$|s))" % from_sgt, " "+to_sgt+" ", text)
                    sentences.append(new_text)
                    results["text"].append(new_text)
            try:
                results["perplexity"].extend(self.get_perplexity(sentences, chosen_batch_size=16))
                
            except Exception:
                results["perplexity"].extend(self.get_perplexity(sentences, chosen_batch_size=4))

            if index % 100 == 99:
                pd.DataFrame.from_dict(results).to_csv(result_path, index=False, header=False, mode="a")

                results = {
                    "text": list(),
                    "id": list(),
                    "original_sgt": list(),
                    "new_sgt": list(),
                    "perplexity": list()
                }

        pd.DataFrame.from_dict(results).to_csv(result_path, index=False, header=False, mode="a")


    def chunk(self, filepath, num_of_rows_per_chunk=2000):
        filename = filepath.split("/")[-1]
        dataset = pd.read_csv(filepath)
        self.found_sgts = list(set(dataset["sgts"]))
        list_df = [dataset[i:i + num_of_rows_per_chunk] for i in range(0, dataset.shape[0], num_of_rows_per_chunk)]
        for i in range(len(list_df)):
            result_path = "dump/Chunk_" + str(i) + "_" + filename
            print("saving to", result_path)
            list_df[i].to_csv(result_path)
        return len(list_df)
