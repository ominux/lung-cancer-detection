import time
print 't1'
t1 = time.time()
for i in range(10000000):
	print i

print 't2'
t2 = time.time()

diff = t2-t1
print diff