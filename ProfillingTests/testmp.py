import multiprocessing as mp

def f(l, i,turn):
    l.acquire()
    try:
        print('hello world', 170*turn+i)
    finally:
        l.release()

def task(value):  
    print(f"task : {value}")

if __name__ == '__main__':
    #CASE 1
    # lock = mp.Lock()
    # for i in range(170):
    #     for num in range(100):
    #         mp.Process(target=f, args=(lock, num,i)).start()

    #CASE 2

    tasks=[]
    t_sz=3823
    n_processes=123
    for i in range(t_sz):
        tasks.append([task,i,0])

    j=0
    
    while j<t_sz:
        processes=[]
        for i in range(n_processes):
            if j+i>t_sz-1:
                 break
            # print(j+i)
            p=mp.Process(target=tasks[j+i][0],args=(tasks[j+i][1],))
            p.start()
            processes.append(p)
            tasks[j+i][2]=1
        for p in processes:
             p.join()
        j+=n_processes
    for task in range(len(tasks)):
         if tasks[task][3]!=1:
              print(task)