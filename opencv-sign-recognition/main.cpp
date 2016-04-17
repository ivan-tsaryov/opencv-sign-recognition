#include "Recognition.h"

int main(int argc, char** argv) {
	VideoCapture cap(0);
	Recognition r;
	r.detectSign(cap, true, 20);

	return 0;
}