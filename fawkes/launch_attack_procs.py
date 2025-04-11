import multiprocessing
import importlib

def main():
    processes = []

    for i in range(1):
        module = importlib.import_module(f"run_attacks{i}")
        p = multiprocessing.Process(target=module.run) 
        processes.append(p)

    for p in processes:
        p.start()

    for p in processes:
        p.join()

if __name__ == "__main__":
    main()