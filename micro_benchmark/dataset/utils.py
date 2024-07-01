import faiss
import numpy as np
from collections import defaultdict
import pickle
import pdb

group2id = {
    'validation' : '0',
    'train' : '1',
    'test' : '2'
}

id2group = {
    '0' : 'validation',
    '1' : 'train',
    '2' : 'test'
}

desc2id = {
    'description' : '1',
    'numeric' : '2',
    'entity' : '3',
    'location' : '4',
    'person' : '5'
}

ANS_QUERY_IDX = 0
ANS_ANSWER_IDX = 1
ANS_PASSAGE_START = 2
ANS_PASSAGE_END = 3
ANS_GROUP = 4
ANS_GROUP_IDX = 5

answer_list = None
query_list = None
doc_list = None

"""Load a pickle file.
@return
The content of the pickle file.
"""
def load_pickle(file_name):
    with open(file_name, "rb") as file:
        d = pickle.load(file)
    return d


def store_pickle(data, name):
    with open(name, 'wb') as file:
        pickle.dump(data, file)

"""Load the reference answers.
@return
The answer list, query list and doc list.
"""
def load_answer():
    global answer_list, query_list, doc_list
    if answer_list is None:
        answer_list = load_pickle("answer_list.pickle")
        query_list = load_pickle("query_list.pickle")
        doc_list = load_pickle("doc_list.pickle")
    return {
        'answer_list' : answer_list,
        'query_list' : query_list,
        'doc_list' : doc_list
    }

"""Load the query and document embeddings. All the queries and embeddings are
merged together.

@input
total_query: the total number of query embeddings to be loaded.
total_passage: the total number of passage embeddings to be loaded.
dimension: the dimension of the embedding.
chunk_start: the starting index of the embedding chunk file.
chunk_end: the ending index of the embedding chunk file, inclusive.
@return
q_ebd: the concatenated numpy array of the query embeddings.
       Its shape is (total_query, dimension)
q_doc: the concatednated numpy array of the passage embeddings.
       Its shape is (total_passage, dimension)
"""
def load_merged_embedding(total_query=102023, total_passage=837729,
        dimension=1024, chunk_start=0, chunk_end=5):
    q_ebd = np.zeros((total_query, dimension), dtype=np.float32)
    d_ebd = np.zeros((total_passage, dimension), dtype=np.float32)
    q_pos, d_pos = 0, 0
    for i in range(chunk_start, chunk_end + 1):
        q = load_pickle(f"ebd_query_{i}.pickle")
        d = load_pickle(f"ebd_doc_{i}.pickle")
        q_ebd[q_pos : q_pos + q.shape[0]] = q
        d_ebd[d_pos : d_pos + d.shape[0]] = d
        q_pos += q.shape[0]
        d_pos += d.shape[0]
    return q_ebd, d_ebd

query_limit_dict, doc_limit_dict = None, None


def seperate_dataset():
    global query_limit_dict, doc_limit_dict
    if query_limit_dict is not None:
        return (query_limit_dict, doc_limit_dict)
    load_answer()
    # First, we need to know the start and end index of each group. 
    # All indice are inclusive.
    group_start_idx = -1
    group_end_idx = -1
    query_limit_dict = dict()
    for group_id, group_name in id2group.items():
        for idx, answer in enumerate(answer_list):
            if answer[ANS_GROUP] == group_id:
                if group_start_idx == -1:
                    group_start_idx = idx
                group_end_idx = idx
            elif group_start_idx != -1:
                break
        assert(group_start_idx >=0)
        assert(group_end_idx >=0)
        assert(group_end_idx > group_start_idx)
        assert(answer_list[group_end_idx][ANS_GROUP] == group_id)
        query_limit_dict[group_name] = (group_start_idx, group_end_idx)
        group_start_idx = -1
        group_end_idx = -1
    print(query_limit_dict)

    # Then, we need the boundaries of documents for each group.
    #  All indice are inclusive.
    doc_limit_dict = dict()
    for group_name, limit in query_limit_dict.items():
        doc_start_idx = answer_list[limit[0]][ANS_PASSAGE_START]
        doc_end_idx = answer_list[limit[1]][ANS_PASSAGE_END]
        doc_limit_dict[group_name] = (doc_start_idx, doc_end_idx)
    print(doc_limit_dict)
    return query_limit_dict, doc_limit_dict


"""Load the query and document embeddings. The queries and embeddings are
seperated according to their groups, i.e., validation, train, and test.
@input
total_query: the total number of query embeddings to be loaded.
total_passage: the total number of passage embeddings to be loaded.
dimension: the dimension of the embedding.
chunk_start: the starting index of the embedding chunk file.
chunk_end: the ending index of the embedding chunk file, inclusive.
@return
q_ebd: the concatenated numpy array of the query embeddings.
       Its shape is (total_query, dimension)
q_doc: the concatednated numpy array of the passage embeddings.
       Its shape is (total_passage, dimension)
"""
def load_seperated_embedding(total_query=102023, total_passage=837729,
        dimension=1024, chunk_start=0, chunk_end=5):
    
    query_limit_dict, doc_limit_dict = seperate_dataset()
    q_ebd, d_ebd = load_merged_embedding(total_passage, total_passage,
                                         dimension, chunk_start, chunk_end)
    # Finally, let's split the embeddings.
    result = defaultdict(dict)
    for _, group_name in id2group.items():
        q_limit = query_limit_dict[group_name]
        d_limit = doc_limit_dict[group_name]
        result[group_name]['query'] = q_ebd[q_limit[0] : q_limit[1] + 1]
        result[group_name]['doc'] = d_ebd[d_limit[0] : d_limit[1] + 1]
    return result


def search(query_ebd, doc_ebd, k=10):
    index_cpu = faiss.IndexFlatL2(1024)
    index_gpu = faiss.index_cpu_to_all_gpus(index_cpu)
    index_gpu.add(doc_ebd)
    # For KNN, top k.
    distance, index = index_gpu.search(query_ebd, k)
    return distance, index


def evaluate_profile(index, group_name):
    load_answer()
    print(group_name)
    query_limits, doc_limits = seperate_dataset()
    q_limit = query_limits[group_name]
    d_limit = doc_limits[group_name]
    answer = answer_list
    success_count = 0
    ranking = float(0)
    total_count = q_limit[1] - q_limit[0] + 1
    assert(total_count == index.shape[0])
    for idx, search_result in enumerate(index):
        #pdb.set_trace()
        correct_answer_list = answer_list[idx + q_limit[0]][ANS_ANSWER_IDX]
        if correct_answer_list[0] == -1:
            # No correct answer as labeled in msmarco.
            total_count -= 1
            continue
        answer_rank_list = list()
        success_found_flag = False
        for correct_idx in correct_answer_list:
            try:
                assert(correct_idx >= d_limit[0])
                assert(correct_idx <= d_limit[1])
            except:
                pdb.set_trace()
            in_group_idx = correct_idx - d_limit[0]
            if in_group_idx in search_result:
                if not success_found_flag:
                    success_found_flag = True
                    success_count += 1
                # 9 is the maximum index since I look for 10 answers.
                # The best case is that the index is 0 and ranking is 0, worst ranking is 1 for successful search.
                try:
                    answer_rank_list.append(float(np.where(search_result == in_group_idx)[0][0]) / 9)
                except:
                    pdb.set_trace()
        if not not answer_rank_list:
            ranking += min(answer_rank_list)
    # Get the average ranking.
    ranking = ranking / success_count
    print(f"Success Rate 1: {success_count}/{total_count}({success_count / total_count})")
    print(f"Average ranking is {ranking}")

    success_count = 0
    for idx, search_result in enumerate(index):
        correct_answer_list = answer_list[idx + q_limit[0]][ANS_ANSWER_IDX]
        if correct_answer_list[0] == -1:
            # No correct answer as labeled in msmarco.
            continue
        in_group_id_list = [a - d_limit[0] for a in correct_answer_list]
        if search_result[0] in in_group_id_list:
            success_count += 1
    print(f"First Hit Rate: {success_count}/{total_count}({success_count / total_count})")

def write_report(group_name, idx, query, passages, correct_answer, knn_result, search_result):
    with open(f"{group_name}-{idx}.report", 'w') as f:
        f.write(f"ID: {group_name}-{idx}\n\n")
        f.write(f"#########################\n")
        f.write(f"Query: {query}\n\n")
        f.write(f"#########################\n")
        f.write("Passages:\n\n")
        for p in passages:
            f.write(p+'\n\n')
        f.write(f"#########################\n")
        f.write(f"Correct Answer: \n\n")
        for p in correct_answer:
            f.write(p+'\n')
        f.write(f"#########################\n")
        f.write("Top k:\n\n")
        for idx, p in enumerate(knn_result):
            context = get_knn_context(search_result[idx])
            f.write(f"{context}\n")
            f.write(p+'\n\n')

def get_knn_context(doc_id):
    load_answer()
    for a in answer_list:
        if a[ANS_PASSAGE_START] <= doc_id and a[ANS_PASSAGE_END] >= doc_id:
            try:
                return {
                    'query': query_list[a[ANS_QUERY_IDX]],
                    'answer id': a[ANS_ANSWER_IDX],
                    'passages start': a[ANS_PASSAGE_START],
                    'passage end' : a[ANS_PASSAGE_END],
                    'Group ID': a[ANS_GROUP]
                }
            except:
                pdb.set_trace()

def evaluate_item(index, group_name):
    load_answer()
    query_limits, doc_limits = seperate_dataset()
    q_limit = query_limits[group_name]
    d_limit = doc_limits[group_name]

    for idx, search_result in enumerate(index):
        answer = answer_list[idx + q_limit[0]]
        correct_answer_list = answer[ANS_ANSWER_IDX]
        if correct_answer_list[0] == -1:
            # No correct answer as labeled in msmarco.
            continue
        success_found_flag = False
        for correct_idx in correct_answer_list:
            in_group_idx = correct_idx - d_limit[0]
            if in_group_idx in search_result:
                success_found_flag = True
                break
        if not success_found_flag:
            passages = doc_list[answer[ANS_PASSAGE_START] :
                                answer[ANS_PASSAGE_END] + 1]
            query = query_list[answer[ANS_QUERY_IDX]]
            correct_answer = [doc_list[d] for d in answer[ANS_ANSWER_IDX]]
            knn_result_list = [s + d_limit[0] for s in search_result]
            passages_found = [doc_list[i] for i in knn_result_list]
            write_report(group_name, idx, query, passages, correct_answer,
                         passages_found, knn_result_list)
            break


def main():
    
    ebd_dict = load_seperated_embedding()
    #q_old, d_old = load_old_embedding()
    
    distance, index = search(ebd_dict['train']['query'], ebd_dict['train']['doc'])
    #evaluate_profile(index, 'train')
    

    for group_name, ebd in ebd_dict.items():
        distance, index = search(ebd['query'], ebd['doc'])
        evaluate_profile(index, group_name)
    evaluate_item(index, 'train')
    #distance_old, index_old = search(q_old, d_old)
    #evaluate_profile(index_old)
    #evaluate_item(index_new)


if __name__ == "__main__":
    main()
       
