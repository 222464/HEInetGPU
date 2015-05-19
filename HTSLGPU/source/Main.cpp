#include <system/ComputeSystem.h>

#include <time.h>
#include <iostream>

int main() {
	std::mt19937 generator(time(nullptr));

	sys::ComputeSystem cs;

	cs.create(sys::ComputeSystem::_gpu);

	return 0;
}