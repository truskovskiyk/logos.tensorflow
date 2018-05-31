#!/usr/bin/env bash
wget http://image.ntua.gr/iva/datasets/flickr_logos/flickr_logos_27_dataset.tar.gz
tar -zxvf flickr_logos_27_dataset.tar.gz
tar -zxvf ./flickr_logos_27_dataset/flickr_logos_27_dataset_images.tar.gz
mkdir data
mv flickr_logos_27_dataset ./data/
mv flickr_logos_27_dataset_images ./data/
rm -f flickr_logos_27_dataset.tar.gz
