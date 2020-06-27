## Code
```
# Single image on CIFAR
time perf stat -e cache-misses,cache-references,instructions,cycles python cifar.py

# Single image on MNIST
time perf stat -e cache-misses,cache-references,instructions,cycles python mnist.py 
```
  
