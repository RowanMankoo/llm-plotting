# Define variables
CC = gcc
CFLAGS = -Wall -Werror

# Define targets and dependencies
all: main

main: main.o
	$(CC) $(CFLAGS) -o main main.o

main.o: main.c
	$(CC) $(CFLAGS) -c main.c

clean:
	rm -f main main.o

