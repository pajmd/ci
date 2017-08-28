from chapter4.my_nn import Searchnet

def test_training():
    mynet = Searchnet('nn.db')
    # mynet.maketables()
    mynet.del_db()
    wWorld, wRiver, wBank = 101, 102, 103
    uWorldBank, uRiver, uEarth = 201, 202, 203
    allurls = [uWorldBank, uRiver, uEarth]
    for i in range(30):
        print('#'*20)
        mynet.trainquery([wWorld, wBank], allurls, uWorldBank)
        print('[wWorld, wBank]: ', mynet.wi)
        print('[wWorld, wBank]: ',mynet.wo)
        mynet.trainquery([wRiver, wBank], allurls, uRiver)
        print('[wRiver, wBank]: ',mynet.wi)
        print('[wRiver, wBank]: ',mynet.wo)
        mynet.trainquery([wWorld], allurls, uEarth)
        print('[wWorld]: ',mynet.wi)
        print('[wWorld]: ',mynet.wo)
    mynet.getresult([wWorld, wBank], allurls)
    print(mynet.ao)
    mynet.getresult([wRiver, wBank], allurls)
    print(mynet.ao)
    mynet.getresult([wBank], allurls)
    print(mynet.ao)

