all: producer-consumer rsa-sample

producer-consumer: producer-consumer.c
	gcc producer-consumer.c -o producer-consumer -lpthread -Wall

rsa-sample: rsa-sample.c
	gcc rsa-sample.c -o rsa-sample -lm -Wall

clean:
	rm rsa-sample producer-consumer

