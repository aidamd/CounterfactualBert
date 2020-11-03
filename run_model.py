import argparse
from CounterfactualBert import *
import json


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus")
    parser.add_argument("--params")
    #parser.add_argument("--counter")

    args = parser.parse_args()
    params = json.load(open(args.params, 'r'))
    
    train_file = os.path.join(args.corpus, "train.csv")
    counter_file = os.path.join(args.corpus, "counter.csv")
    data = pd.read_csv(train_file)
    counter = pd.read_csv(counter_file)
    
    model = CounterfactualBert(params, data, counter, args.corpus)
    model.CV()


