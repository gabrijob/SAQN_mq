#!/bin/bash

python3 setup.py build_ext --inplace
gcc -I/usr/include/python3.8 -c DummyMQ.c
gcc -o DummyMQ DummyMQ.o build/temp.linux-x86_64-3.8/SAQNAgent.o  -I/usr/include/python3.8 -L/usr/lib/python3.8/config-3.8-x86_64-linux-gnu -L/usr/lib -lpython3.8 -lcrypt -lpthread -ldl  -lutil -lm
