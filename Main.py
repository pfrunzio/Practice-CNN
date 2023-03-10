from CNN import load_data, test_print_data, make_cnn, save_cnn, load_cnn, test_print_results, accuracy_overall, accuracy_perclass
    
def main():
    
    trainloader, testloader, classes = load_data()

    # test_print_data(trainloader, classes, batch_size)
    
    PATH = './models/cifar_net.pth'
    
    # net = make_cnn(trainloader)
    # save_cnn(net, PATH)
    net = load_cnn(PATH)

    # test_print_results(net, testloader, classes)
    
    accuracy_overall(net, testloader)
    
    accuracy_perclass(net, testloader, classes)

if __name__ == "__main__":
    main()