import torch
def main():
    print(f"CUDA Available: {torch.cuda.is_available()}")

    # This should print '12.1'
    print(f"PyTorch CUDA Version: {torch.version.cuda}")

    # This should print 'NVIDIA GeForce RTX 3050 ...'
    if torch.cuda.is_available():
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
        
if __name__ == "__main__":
    main()
