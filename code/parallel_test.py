import multiprocessing
import numpy as np
import time

def task_1(data, lock, output_file, fileno, queue):
    # Process the data (example: calculate squares)
    processed_data = [x**3 for x in data]

    # Write to the output file with locking
    with lock:
        with open(output_file, 'a') as f:
            f.write(f"Task 1 output from file {fileno}: \n")
    queue.put(processed_data)
    print("Task 1 done")

def task_2(data, lock, output_file, fileno, queue):
    # Process the data (example: calculate cubes)
    processed_data = [x**(1./3.) for x in data]

    # Write to the output file with locking
    with lock:
        with open(output_file, 'a') as f:
            f.write(f"Task 2 output from file {fileno}: \n")
    queue.put(processed_data)
    print("Task 2 done")

def process_file(input_file, output_file, lock, fileno):
    # Load data from input file
    data = np.loadtxt(input_file, dtype=int)

    queue = multiprocessing.Queue()

    # Create processes for task_1 and task_2
    p1 = multiprocessing.Process(target=task_1, args=(data, lock, output_file, fileno, queue))
    p2 = multiprocessing.Process(target=task_2, args=(data, lock, output_file, fileno, queue))

    # Start both processes
    p1.start()
    p2.start()

    print("Before get")
    result_1 = queue.get()
    print("After get 1")
    result_2 = queue.get()
    print("After get 2")

    print("Before join")
    # Wait for both processes to finish
    p1.join()
    print("After join 1")
    p2.join()

    print(f"Results from file {fileno}: Task 1 -> Completed, Task 2 -> Completed")
    return

if __name__ == "__main__":
    input_files = ['input_data1.txt', 'input_data2.txt', 'input_data3.txt']  # List of input files
    fileno = ['1', '2', '3']  # Identifiers for each file
    output_file = 'output_data2.txt'

    # Create a lock object
    lock = multiprocessing.Lock()

    start_time = time.time()
    # Process each file in parallel
    processes = []
    for i in range(len(input_files)):
        p = multiprocessing.Process(target=process_file, args=(input_files[i], output_file, lock, fileno[i]))
        processes.append(p)
        p.start()

    # Wait for all processes to finish
    for p in processes:
        p.join()
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")

