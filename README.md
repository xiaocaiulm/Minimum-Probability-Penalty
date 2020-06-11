# Minimum-Probability-Penalty
## Usage

1. Download [CUB-200](http://www.vision.caltech.edu/visipedia/CUB-200.html),
[FGVC-Aircrafts](http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/) and
[Stanford Cars](https://ai.stanford.edu/~jkrause/cars/car_dataset.html).
2. Convert the dataset to the designated format using utils/convert_data.py and place them in data/bird(aircraft/car)
3. Run the following code to get the reported results:
#### CUB
```bash
python train.py --model-name=densenet161 --scheduler=cos --lr=0.008 --loss=mpp --smoothing=0.1 -b=16 --image-size=600
```
#### Aircrafts
```bash
python train.py --model-name=densenet161 --lr=0.005 --loss=mpp --smoothing=0.1 -b=4
```
#### Stanford Cars
```bash
python train.py --model-name=densenet161 --lr=0.001 --loss=mpp --smoothing=0.1 -b=4
```
