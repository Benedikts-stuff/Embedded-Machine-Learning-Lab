import torch
import torchvision.datasets


def get_client_training_data(client_id, total_clients):
    assert 0 <= client_id < total_clients
    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor()
    ])

    dataset = torchvision.datasets.CIFAR10('data/', train=True, download=True, transform=transform)

    length = len(dataset)
    l = length // total_clients
    lengths = [l + 1 if i < length % total_clients else l for i in range(total_clients)]
    splits = torch.utils.data.random_split(dataset, lengths, generator=torch.Generator().manual_seed(1))
    return splits[client_id]


def get_test_data():
    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor()
    ])
    return torchvision.datasets.CIFAR10('data/', train=False, download=True, transform=transform)