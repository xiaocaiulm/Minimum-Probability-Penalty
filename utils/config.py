def get_dataset_config(dataset_name):
    assert dataset_name in ['bird', 'car',
                            'aircraft','dog'], 'No dataset named %s!' % dataset_name
    dataset_dict = {
        'bird': {'train_root': 'data/bird/images',
                 'val_root': 'data/bird/images',

                 'train': 'data/bird/bird_train.txt',
                 'val': 'data/bird/bird_test.txt'},
        'car': {'train_root': 'data/car/cars_train',
                'val_root': 'data/car/cars_test',
                'train': 'data/car/car_train.txt',
                'val': 'data/car/car_test.txt'},
        'aircraft': {'train_root': 'data/aircraft/images',
                     'val_root': 'data/aircraft/images',
                     'train': 'data/aircraft/aircraft_train.txt',
                     'val': 'data/aircraft/aircraft_test.txt'},
        'dog': {'train_root': 'Fine-grained/dogs/Images',
                'val_root': 'Fine-grained/dogs/Images',
                'train': 'Fine-grained/dogs/dog_train.txt',
                'val': 'Fine-grained/dogs/dog_test.txt'},
    }
    return dataset_dict[dataset_name]
