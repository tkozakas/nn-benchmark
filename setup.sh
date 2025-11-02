mkdir dataset
cd dataset

curl -L -o . \
https://www.kaggle.com/api/v1/datasets/download/akash2sharma/tiny-imagenet
unzip tiny-imagenet.zip

cd ..