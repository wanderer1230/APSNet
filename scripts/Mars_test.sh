#!/bin/scripts
#change test_path to test
python Test.py  --arch 'PSTA'\
                --dataset 'mars'\
                --test_sampler 'Begin_interval'\
                --triplet_distance 'cosine'\
                --test_distance 'cosine'\
                --test_path ''
