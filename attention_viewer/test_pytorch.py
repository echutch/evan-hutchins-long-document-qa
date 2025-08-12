from torch import cuda, version

def main():
    print(f"CUDA Available: {cuda.is_available()}")

    # This should print '12.1'
    print(f"PyTorch CUDA Version: {version.cuda}")

    # This should print 'NVIDIA GeForce RTX 3050 ...'
    if cuda.is_available():
        print(f"Device Name: {cuda.get_device_name(0)}")

if __name__ == "__main__":
    main()
