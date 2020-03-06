#!/bin/bash

# target='target'
read -p "Enter the directory: " target

if [ -d $target ]
then
echo "Directory Already Exists"
else
mkdir $target
figs=$(find . -maxdepth 1 -name 'fig*.png')
# echo $all
for fig in $figs
do 
cp $fig $target
done

meas=$(find . -maxdepth 1 -name 'meas*')
for mea in $meas
do 
cp $mea $target
done

distributions=$(find . -maxdepth 1 -name 'distributions*')
for distribution in $distributions
do 
cp $distribution $target
done

results=$(find . -maxdepth 1 -name 'results*')
for result in $results
do 
cp $result $target
done

cp 'config.py' $target
cp -r 'chains' $target
fi
