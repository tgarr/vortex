import numpy as np
import pickle
import pdb
import time
import faiss

d = 1024                           # dimension
log_name = "test_gpu.log"

def write_log(l):
    with open(log_name, "a") as file:
        file.write(l+'\n')


def cpu_test(nb=100000, nq=100, k=10):
    np.random.seed(1234)             # make reproducible
    xb = np.random.random((nb, d)).astype('float32')
    xb[:, 0] += np.arange(nb) / 1000.
    xq = np.random.random((nq, d)).astype('float32')
    xq[:, 0] += np.arange(nq) / 1000.
    nlist = 1000
    m = 8
    quantizer = faiss.IndexFlatL2(d)  # this remains the same
    # 8 specifies that each sub-vector is encoded as 8 bits
    index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)
    t1 = time.time()
    index.train(xb)
    t2 = time.time()
    write_log(f"CPU Test, nb:{nb}, nq:{nq}, k:{k}, Training time:{t2-t1}s.")
    t1 = time.time()
    index.add(xb)
    t2 = time.time()
    write_log(f"CPU Test, nb:{nb}, nq:{nq}, k:{k}, Loading time:{t2-t1}s.")
    t1 = time.time()
    index.nprobe = 10
    D, I = index.search(xq, k)
    t2 = time.time()
    write_log(f"CPU Test, nb:{nb}, nq:{nq}, k:{k}, Searching time:{t2-t1}s.")

def gpu_test(nb=100000, nq=100, k=10):
    np.random.seed(1234)             # make reproducible
    xb = np.random.random((nb, d)).astype('float32')
    xb[:, 0] += np.arange(nb) / 1000.
    xq = np.random.random((nq, d)).astype('float32')
    xq[:, 0] += np.arange(nq) / 1000.
    nlist = 1000
    m = 8
    quantizer = faiss.IndexFlatL2(d)  # this remains the same
    # 8 specifies that each sub-vector is encoded as 8 bits
    cpu_index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)
    gpu_index = faiss.index_cpu_to_all_gpus(cpu_index)

    t1 = time.time()
    gpu_index.train(xb)
    t2 = time.time()
    write_log(f"GPU Test, nb:{nb}, nq:{nq}, k:{k}, Training time:{t2-t1}s.")
    t1 = time.time()
    gpu_index.add(xb)
    t2 = time.time()
    write_log(f"GPU Test, nb:{nb}, nq:{nq}, k:{k}, Loading time:{t2-t1}s.")

    t1 = time.time()
    gpu_index.nprobe = 10
    D, I = gpu_index.search(xq, k)
    t2 = time.time()
    write_log(f"GPU Test, nb:{nb}, nq:{nq}, k:{k}, Searching time:{t2-t1}s.")


def main():
    for nb in [100000, 1000000, 10000000]:
        print(f"nb={nb}\n")
        cpu_test(nb)
        gpu_test(nb)
 
if __name__ == "__main__":
    main()
