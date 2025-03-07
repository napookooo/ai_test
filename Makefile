main: src/main.c
	gcc -o main src/main.c -lm

clean:
	rm main
