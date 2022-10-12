all: producer-consumer.o rsa-sample.o

producer-consumer.o: producer-consumer.c
	gcc producer-consumer.c -o producer-consumer.o -lpthread -Wall

rsa-sample.o: rsa-sample.c
	gcc rsa-sample.c -o rsa-sample.o -lm -Wall

clean:
	rm *.o

