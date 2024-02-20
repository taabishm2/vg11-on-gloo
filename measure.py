from collections import defaultdict
import hashlib
import psutil
import time
import csv
import os

MEASUREMENTS = defaultdict(list)

def measure_iters(source, iter, start_time, iter_loss, 
                  total_loss, batch_size, sync_time):
    # print("Running iteration: ", iter)
    if iter == 0 or iter > 41: return
    
    total_io = psutil.net_io_counters()
    process = psutil.Process(os.getpid())
    
    # print(f"Total Bytes Sent: {total_io.bytes_sent / 1024 ** 2} MB")
    # print(f"Total Bytes Received: {total_io.bytes_recv / 1024 ** 2} MB")
    # print(f"Current memory usage: {process.memory_info().rss / 1024 ** 2} MB")  # RSS (Resident Set Size)
    
    MEASUREMENTS[source].append([
        "source", "iter", "time", "iter_loss", 
        "total_loss", "batch_size", "sync_time",
        "bytes_sent", "bytes_recv", "memory"])
    
    MEASUREMENTS[source].append([
        source, iter, time.time()-start_time, 
        iter_loss, total_loss, batch_size, sync_time,
        total_io.bytes_sent / 1024 ** 2,
        total_io.bytes_recv / 1024 ** 2,
        process.memory_info().rss / 1024 ** 2
    ])
    
    if iter == 41:
        csv.writer(open(f"results/{source}.csv", "w")).writerows(MEASUREMENTS[source])
        print(f"Results saved to results/{source}.csv")
        
def save_params(file, model):
    file = f"data/{file}.params"
    with open(file, 'w') as f:
        for name, param in model.named_parameters():
            f.write(f'Layer: {name}\n')
            f.write(f'Parameters: {param.size()}\n')
            f.write(f'Values:\n{param.data.numpy()}\n\n')
    
    sha256_hash = hashlib.sha256()
    with open(file, 'rb') as f:
        # Read and update hash in chunks to avoid loading large files into memory
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)
    
    print(f"\n\nSHA(params): {sha256_hash.hexdigest()}")