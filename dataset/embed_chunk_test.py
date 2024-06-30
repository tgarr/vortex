from FlagEmbedding import BGEM3FlagModel
from datasets import load_dataset
import numpy as np
import pickle
import pdb
import time

group2id = {
    'validation' : '0',
    'train' : '1',
    'test' : '2'
}

log_file = None


def download_msmarco(version='v1.1'):
    ds = load_dataset('microsoft/ms_marco', version)
    return ds




def load_pickle(name):
    with open(name, 'rb') as file:
        loaded_data = pickle.load(file)
    return loaded_data




def main():
    global log_file
    log_file = open("embed.log", 'w', 1)
    t1 = time.time()
    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=False)
    t2 = time.time()
    chunk_len = 17004
    total_len = 102023
    dimension = 1024

    ds = download_msmarco("v1.1")

    q_count = 0
    d_count = 0
    total_count = 0
    d_count_list = list()
    q_count_list = list()
    for group_name, group_id in group2id.items():
        for index, data in enumerate(ds[group_name]):
            total_count += 1
            q_count += 1
            d_count += len(data['passages']['passage_text'])
            if q_count % chunk_len == 0:
                d_count_list.append(d_count)
                q_count_list.append(q_count)
                d_count = 0
                q_count = 0
    d_count_list.append(d_count)
    q_count_list.append(q_count)

    
    total_processed = 0
    q_count = 0
    start_iteration = 5
    start_q = sum(q_count_list[:start_iteration])
    iteration = start_iteration
    query_ebd_list = np.zeros((q_count_list[iteration], dimension), dtype=np.float32)
    doc_ebd_list = np.zeros((d_count_list[iteration],
                    dimension), dtype=np.float32)
    doc_pos = 0
    
    t1 = time.time()

    for group_name, group_id in group2id.items():
        for index, data in enumerate(ds[group_name]):
            total_processed += 1
            if total_processed <= start_q:
                continue
            context = " ".join(data['passages']['passage_text'])
            passage_list = [context] +\
                [p for p in data['passages']['passage_text']] +\
                [data['query']]
            passage_count = len(data['passages']['passage_text'])
            ebd = np.ones((passage_count+2, 1024), dtype=np.float32)
            # ebd = embed(passage_list, model)['dense_vecs']
            # Put the passages' embedding in the doc embedding list.
            try:
                doc_ebd_list[doc_pos : doc_pos + passage_count] = \
                    ebd[1 : -1]
            except:
                print(f"Error: {group_name}, {index}, {iteration}, {q_count}, {total_processed}")
                pdb.set_trace()
            doc_pos += passage_count
            # Put the query's embedding in the query embedding list.
            query_ebd_list[q_count] = ebd[-1]
            q_count += 1
            if total_processed % 1000 == 0:
                print(f"Finish {total_processed}/102023", flush=True)
            if q_count % chunk_len == 0 or total_processed == total_count:
                t2 = time.time()
                #write_log(f'Doc embedding Chunk {iteration} time: {t2 - t1}')

                #store_pickle(doc_ebd_list, f"ebd_doc_{iteration}.pickle")
                #store_pickle(query_ebd_list, f"ebd_query_{iteration}.pickle")
                iteration += 1
                if iteration != 6:
                    q_count = 0
                    query_ebd_list = np.zeros((q_count_list[iteration], dimension), dtype=np.float32)
                    doc_ebd_list = np.zeros((d_count_list[iteration],
                                    dimension), dtype=np.float32)
                    doc_pos = 0
                    t1 = time.time()
                else:
                    break
    print("Complete!")
    log_file.close()


if __name__ == "__main__":
    main()
