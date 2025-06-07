#!/bin/bash

mkdir -p test_out/

function testA {
    python run.py FF0000 0000FF -o test_out/im1 --testSrip
    python run.py FF0000 FF00FF -o test_out/im2 --testSrip
    python run.py e0a21d 1de0a2 -o test_out/im3 --testSrip
    python run.py e0a21d a21de0 -o test_out/im4 --testSrip
    python run.py e0a21d 8fd6bf -o test_out/im5 --testSrip
}

function genForBlog {
    # For blog post specifically
    python run.py e0a21d 1de0a2 -o test_out/test_img1 --testSrip
    python run.py FF0000 0000FF -o test_out/test_img2 --testSrip


    python run.py e0a21d 1de0a2 -o test_out/test_a_img1 -a
    python run.py FF0000 0000FF -o test_out/test_a_img2 -a
}

# testA()
genForBlog