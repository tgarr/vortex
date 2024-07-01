from datasets import load_dataset
import pdb
import pickle
import random

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


def download_msmarco(version='v1.1'):
    ds = load_dataset('microsoft/ms_marco', version)
    return ds


def test(ds, query_list, doc_list, answer_list):
    total_query = 0
    total_doc = 0
    for group_name, group_id in group2id.items():
        for index, data in enumerate(ds[group_name]):
            total_query += 1
            total_doc += len(data['passages']['is_selected'])
    assert(total_query == len(query_list))
    assert(total_query == len(answer_list))
    assert(total_doc == len(doc_list))
    lower_bound = 0
    upper_bound = total_query
    random_idx_list = [random.randint(lower_bound, upper_bound) for _ in range(100)]
    for order, idx in enumerate(random_idx_list):
        answer = answer_list[idx]
        group_name = id2group[answer[ANS_GROUP]]
        idx_in_group = answer[ANS_GROUP_IDX]
        data = ds[group_name][idx_in_group]
        assert(query_list[idx] == data['query'])
        select_count = 0
        for answer_idx, selected in enumerate(data['passages']['is_selected']):
            if selected == 1:
                find = doc_list[answer[1][select_count]]
                ref = data['passages']['passage_text'][answer_idx]
                assert(find == ref)
                select_count += 1
        if select_count == 0:
            assert(len(answer[1]) == 1 and answer[1][0] == -1)
        if (order + 1) % 10 == 0:
            print(f"{order + 1} is done.")


def convert_format(ds):
    query_list = list()
    doc_list = list()
    answer_list = list()
    query_idx, doc_idx = 0, 0
    non_selected_count = 0
    multiple_selected_count = 0
    for group_name, group_id in group2id.items():
        for index, data in enumerate(ds[group_name]):
            query_list.append(data['query'])
            doc_list += data['passages']['passage_text']
            answer_idx_list = list()
            for answer_idx, sel in enumerate(data['passages']['is_selected']):
                if sel == 1:
                    answer_idx_list.append(doc_idx + answer_idx)
            if not answer_idx_list:
                # The is_selected list only contains 0s, none of the
                # passages are selected.
                answer_idx_list = [-1]
                non_selected_count += 1
            answer_sum = sum(data['passages']['is_selected'])
            if answer_sum > 1:
                multiple_selected_count += 1
            answer = (
                query_idx, #'query_idx' 
                answer_idx_list,
                doc_idx, # The start of the answer.
                doc_idx + len(data['passages']['passage_text']) - 1, # The end of the answer, inclusive.
                group_id, # 'group_id'
                index, #'idx_within_group'
                desc2id[data['query_type']]
            )
            answer_list.append(answer)
            query_idx += 1
            doc_idx += len(data['passages']['passage_text'])
    print(f"Non-selected: {non_selected_count}, Multiple-selected: {multiple_selected_count}\n")
    return query_list, doc_list, answer_list


def store_pickle(data, name):
    with open(name, 'wb') as file:
        pickle.dump(data, file)


def main():
    ds = download_msmarco()
    query_list, doc_list, answer_list = convert_format(ds)
    test(ds, query_list, doc_list, answer_list)
    store_pickle(query_list, 'query_list.pickle')
    store_pickle(doc_list, 'doc_list.pickle')
    store_pickle(answer_list, 'answer_list.pickle')

    
if __name__ == "__main__":
    main()
