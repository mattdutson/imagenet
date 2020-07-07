#!/usr/bin/env bash

mkdir tmp
tar -xv -f train.tar -C tmp
mkdir train
for ARCHIVE in $(ls tmp/*.tar)
do
    SUBDIR=train/$(basename $ARCHIVE .tar)
    mkdir $SUBDIR
    tar -xv -f $ARCHIVE -C $SUBDIR
    rm $ARCHIVE
done
rmdir tmp

mkdir val
tar -xv -f val.tar -C val
