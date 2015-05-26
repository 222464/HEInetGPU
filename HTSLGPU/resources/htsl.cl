float randFloat(uint2* state) {
	const float invMaxInt = 1.0f / 4294967296.0f;
	uint x = (*state).x * 17 + (*state).y * 13123;
	(*state).x = (x << 13) ^ x;
	(*state).y ^= (x << 7);

	uint tmp = x * (x * x * 15731 + 74323) + 871483;

	return convert_float(tmp) * invMaxInt;
}

void kernel rscInitialize(write_only image3d_t hiddenVisibleWeights,
	write_only image3d_t hiddenHiddenPrevWeights,
	write_only image3d_t hiddenHiddenWeights,
	int receptiveSize, int recurrentSize, int inhibitionSize, float ffWeight, float lWeight, uint2 seed)
{
	uint2 seedValue = seed + (uint2)(get_global_id(0) * 29 + 12, get_global_id(1) * 16 + 23) * 36;

	int2 position = (int2)(get_global_id(0), get_global_id(1));

	for (int wi = 0; wi < receptiveSize; wi++) {
		int4 weightPosition = (int4)(position.x, position.y, wi, 0);

		float weight = ffWeight * (randFloat(&seedValue) * 2.0f - 1.0f);

		write_imagef(hiddenVisibleWeights, weightPosition, (float4)(weight));
	}

	for (int wi = 0; wi < recurrentSize; wi++) {
		int4 weightPosition = (int4)(position.x, position.y, wi, 0);

		float weight = ffWeight * (randFloat(&seedValue) * 2.0f - 1.0f);

		write_imagef(hiddenHiddenPrevWeights, weightPosition, (float4)(weight));
	}

	for (int wi = 0; wi < inhibitionSize; wi++) {
		int4 weightPosition = (int4)(position.x, position.y, wi, 0);

		float weight = lWeight * randFloat(&seedValue);

		write_imagef(hiddenHiddenWeights, weightPosition, (float4)(weight));
	}
}

void kernel rscExcitation(read_only image2d_t inputs, read_only image2d_t spikesRecurrentPrev,
	read_only image3d_t hiddenVisibleWeightsPrev, read_only image3d_t hiddenHiddenPrevWeightsPrev,
	write_only image2d_t excitations,
	int2 inputDims, int2 dims, float2 dimsToInputDims,
	int receptiveRadius, int recurrentRadius)
{
	int2 position = (int2)(get_global_id(0), get_global_id(1));

	int2 inputCenterPosition = (int2)((position.x + 0.5f) * dimsToInputDims.x + 0.5f, (position.y + 0.5f) * dimsToInputDims.y + 0.5f);

	float excitation = 0.0f;

	int wi = 0;

	for (int dx = -receptiveRadius; dx <= receptiveRadius; dx++)
		for (int dy = -receptiveRadius; dy <= receptiveRadius; dy++) {
			int2 inputPosition = (int2)(inputCenterPosition.x + dx, inputCenterPosition.y + dy);

			if (inputPosition.x >= 0 && inputPosition.x < inputDims.x && inputPosition.y >= 0 && inputPosition.y < inputDims.y) {
				float input = read_imagef(inputs, inputPosition).x;

				float weight = read_imagef(hiddenVisibleWeightsPrev, (int4)(position.x, position.y, wi, 0)).x;

				excitation += input * weight;
			}

			wi++;
		}

	wi = 0;

	for (int dx = -recurrentRadius; dx <= recurrentRadius; dx++)
		for (int dy = -recurrentRadius; dy <= recurrentRadius; dy++) {
			int2 inputPosition = (int2)(position.x + dx, position.y + dy);

			if (inputPosition.x >= 0 && inputPosition.x < dims.x && inputPosition.y >= 0 && inputPosition.y < dims.y) {
				float input = read_imagef(spikesRecurrentPrev, inputPosition).x;

				float weight = read_imagef(hiddenHiddenPrevWeightsPrev, (int4)(position.x, position.y, wi, 0)).x;

				excitation += input * weight;
			}

			wi++;
		}

	write_imagef(excitations, position, (float4)(excitation));
}

void kernel rscActivate(read_only image2d_t excitations, read_only image2d_t statesPrev,
	read_only image2d_t activationsPrev, read_only image2d_t spikesPrev,
	read_only image3d_t hiddenHiddenWeightsPrev, read_only image2d_t biasesPrev,
	write_only image2d_t activations, write_only image2d_t spikes, write_only image2d_t states,
	int2 dims,
	int inhibitionRadius, float inhibitionRadiusInv,
	float dt, float iterationsInv)
{
	int2 position = (int2)(get_global_id(0), get_global_id(1));

	float excitation = read_imagef(excitations, position).x;

	float inhibition = 0.0f;

	int wi = 0;

	for (int dx = -inhibitionRadius; dx <= inhibitionRadius; dx++)
		for (int dy = -inhibitionRadius; dy <= inhibitionRadius; dy++) {
			if (dx == 0 && dy == 0)
				continue;

			int2 inputPosition = (int2)(position.x + dx, position.y + dy);

			if (inputPosition.x >= 0 && inputPosition.x < dims.x && inputPosition.y >= 0 && inputPosition.y < dims.y) {
				float input = read_imagef(statesPrev, inputPosition).x;

				float weight = read_imagef(hiddenHiddenWeightsPrev, (int4)(position.x, position.y, wi, 0)).x;

				float falloff = fmax(0.0f, 1.0f - (abs(dx) + abs(dy)) * inhibitionRadiusInv);

				inhibition += falloff * weight * input;
			}

			wi++;
		}

	float activationPrev = read_imagef(activationsPrev, position).x;

	float activation = (1.0f - dt) * activationPrev + dt * (excitation - inhibition);

	float biasPrev = read_imagef(biasesPrev, position).x;

	float spike = 0.0f;

	if (activation > biasPrev) {
		spike = 1.0f;
		activation = 0.0f;
	}

	write_imagef(states, position, (float4)(spike));

	write_imagef(activations, position, (float4)(activation));

	float spikeAccumPrev = read_imagef(spikesPrev, position).x;

	float spikeAccum = spikeAccumPrev + spike * iterationsInv;

	write_imagef(spikes, position, (float4)(spikeAccum));
}

void kernel rscLearn(read_only image2d_t inputs, read_only image2d_t spikes, read_only image2d_t spikesRecurrentPrev,
	read_only image3d_t hiddenVisibleWeightsPrev, read_only image3d_t hiddenHiddenPrevWeightsPrev, read_only image3d_t hiddenHiddenWeightsPrev, read_only image2d_t biasesPrev,
	write_only image3d_t hiddenVisibleWeights, write_only image3d_t hiddenHiddenPrevWeights, write_only image3d_t hiddenHiddenWeights, write_only image2d_t biases,
	int2 inputDims, int2 dims, float2 dimsToInputDims,
	int receptiveRadius, int recurrentRadius, int inhibitionRadius,
	float4 learningRates, float sparsity, float sparsitySquared)
{
	int2 position = (int2)(get_global_id(0), get_global_id(1));

	int2 inputCenterPosition = (int2)((position.x + 0.5f) * dimsToInputDims.x + 0.5f, (position.y + 0.5f) * dimsToInputDims.y + 0.5f);

	float spike = read_imagef(spikes, position).x;

	int wi = 0;

	for (int dx = -receptiveRadius; dx <= receptiveRadius; dx++)
		for (int dy = -receptiveRadius; dy <= receptiveRadius; dy++) {
			int2 inputPosition = (int2)(inputCenterPosition.x + dx, inputCenterPosition.y + dy);

			if (inputPosition.x >= 0 && inputPosition.x < inputDims.x && inputPosition.y >= 0 && inputPosition.y < inputDims.y) {
				float weightPrev = read_imagef(hiddenVisibleWeightsPrev, (int4)(position.x, position.y, wi, 0)).x;

				float input = read_imagef(inputs, inputPosition).x;
	
				float weight = weightPrev + learningRates.x * spike * (input - spike * weightPrev);

				write_imagef(hiddenVisibleWeights, (int4)(position.x, position.y, wi, 0), (float4)(weight));
			}

			wi++;
		}

	wi = 0;

	for (int dx = -recurrentRadius; dx <= recurrentRadius; dx++)
		for (int dy = -recurrentRadius; dy <= recurrentRadius; dy++) {
			int2 inputPosition = (int2)(position.x + dx, position.y + dy);

			if (inputPosition.x >= 0 && inputPosition.x < dims.x && inputPosition.y >= 0 && inputPosition.y < dims.y) {
				float weightPrev = read_imagef(hiddenHiddenPrevWeightsPrev, (int4)(position.x, position.y, wi, 0)).x;

				float input = read_imagef(spikesRecurrentPrev, inputPosition).x;
			
				float weight = weightPrev + learningRates.x * spike * (input - spike * weightPrev);

				write_imagef(hiddenHiddenPrevWeights, (int4)(position.x, position.y, wi, 0), (float4)(weight));
			}

			wi++;
		}

	wi = 0;

	// Inhibitory connections
	for (int dx = -inhibitionRadius; dx <= inhibitionRadius; dx++)
		for (int dy = -inhibitionRadius; dy <= inhibitionRadius; dy++) {
			int2 inputPosition = (int2)(position.x + dx, position.y + dy);

			if (inputPosition.x >= 0 && inputPosition.x < dims.x && inputPosition.y >= 0 && inputPosition.y < dims.y) {		
				float weightPrev = read_imagef(hiddenHiddenWeightsPrev, (int4)(position.x, position.y, wi, 0)).x;
				
				float input = read_imagef(spikes, inputPosition).x;

				float weight = fmax(0.0f, weightPrev + learningRates.z * (input * spike - sparsitySquared));

				write_imagef(hiddenHiddenWeights, (int4)(position.x, position.y, wi, 0), (float4)(weight));
			}

			wi++;
		}

	// Bias
	float biasPrev = read_imagef(biasesPrev, position).x;

	float bias = biasPrev + learningRates.w * (spike - sparsity);

	write_imagef(biases, position, (float4)(bias));
}