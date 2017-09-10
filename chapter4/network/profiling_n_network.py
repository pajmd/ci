from chapter4.network.n_network import Network
from cProfile import Profile
from pstats import Stats

def train_urls_for_profiling():
    mynet = Network(3,14,3)
    mynet.create_connections()
    #mynet.feed_forward([1,1,1])
    #print(mynet.output_ouputs)
    print(mynet.matrix_wih)
    print(mynet.matrix_who)
    for i in range(1000):
        mynet.train([1,0,1],[1,0,0])
        mynet.train([0, 1, 1], [0, 1, 0])
        mynet.train([1, 0, 0], [0, 0, 1])

if __name__ == '__main__':
    profiler =Profile()
    profiler.runcall(train_urls_for_profiling)
    stats =Stats(profiler)
    stats.strip_dirs()
    stats.sort_stats('cumulative')
    stats.print_stats()