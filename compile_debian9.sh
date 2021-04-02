#!/bin/bash

python3 setup.py build_ext --inplace
gcc -I/usr/include/python3.5m -c DummyMQ.c
#gcc -o DummyMQ DummyMQ.o -L/home/ggrabher/build/lib.linux-x86_64-3.5 -I/usr/include/python3.5m -L/usr/lib/python3.5/config-3.5m-x86_64-linux-gnu -L/usr/lib -lSAQNAgent -lpython3.5m -lpthread -ldl  -lutil -lm -lcrypt
gcc -o DummyMQ DummyMQ.o build/temp.linux-x86_64-3.5/SAQNAgent.o -I/usr/include/python3.5m -L/usr/lib/python3.5/config-3.5m-x86_64-linux-gnu -L/usr/lib -lpython3.5m -lpthread -ldl  -lutil -lm -lcrypt
