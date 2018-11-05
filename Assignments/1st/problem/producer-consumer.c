#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define BUF_SIZE 5

int buf[BUF_SIZE]; // the buffer
int len = 0;
int head = 0;
int tail = 0;
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER; // needed to add/remove data from the buffer
pthread_cond_t can_produce = PTHREAD_COND_INITIALIZER; // signaled when items are removed
pthread_cond_t can_consume = PTHREAD_COND_INITIALIZER; // signaled when items are added

// produce random numbers
void* producer(void *arg) {
	long int id = (long int) arg;
    while(1) {
#ifdef UNDERFLOW
        // used to show that if the producer is somewhat "slow"
        // the consumer will not fail (i.e. it'll just wait
        // for new items to consume)
        sleep(rand() % 3);
#endif

        pthread_mutex_lock(&mutex);

        while(len == BUF_SIZE) { // full
            // wait until some elements are consumed
            pthread_cond_wait(&can_produce, &mutex);
        }

        // in real life it may be some data fetched from
        // sensors, the web, or just some I/O
        int t = rand();
        printf("p%ld, index %d: Produced: %d\n", id, head, t);

        // append data to the buffer
        buf[head] = t;
        //++buffer->len;
		head = (head + 1)%BUF_SIZE;
		len++;
        // signal the fact that new items may be consumed
        pthread_cond_signal(&can_consume);
        pthread_mutex_unlock(&mutex);
    }

    // never reached
    return NULL;
}

// consume random numbers
void* consumer(void *arg) {

	long int id = (long int) arg;

    while(1) {
#ifdef OVERFLOW
        // show that the buffer won't overflow if the consumer
        // is slow (i.e. the producer will wait)
        sleep(rand() % 3);
#endif
        pthread_mutex_lock(&mutex);

        while(len == 0) { // empty
            // wait for new items to be appended to the buffer
            pthread_cond_wait(&can_consume, &mutex);
        }

        // grab data
        printf("q%ld, index %d: Consumed: %d\n", id, tail, buf[tail]);
		
        tail = (tail + 1)%BUF_SIZE;
		len--;
        // signal the fact that new items may be produced
        pthread_cond_signal(&can_produce);
        pthread_mutex_unlock(&mutex);
    }

    // never reached
    return NULL;
}

int main(int argc, char *argv[]) {
    
    pthread_t prod_1, cons_1;
	pthread_t prod_2, cons_2;
    
	pthread_create(&prod_1, NULL, producer, (void*)0);
    pthread_create(&prod_2, NULL, producer, (void*)1);
    pthread_create(&cons_1, NULL, consumer, (void*)0);
    pthread_create(&cons_2, NULL, consumer, (void*)1);

    pthread_join(prod_1, NULL);
    pthread_join(prod_2, NULL);
    pthread_join(cons_1, NULL);
    pthread_join(cons_2, NULL);

    return 0;
}
