"""Usage:
python passage_retrieve.py 0 0
The first parameter is the group id. The second parameter is the passage id within that group.
"""
import sys
import pickle

def main():
    group = int(sys.argv[1])
    idx = int(sys.argv[2])
    with open("answer_mapping.pickle", "rb") as file:
        answer_mapping = pickle.load(file)
    with open("doc_list.pickle", "rb") as file:
        doc_list = pickle.load(file)
    print(doc_list[answer_mapping[group][idx]])

    
if __name__ == "__main__":
    main()
